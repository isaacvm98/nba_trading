"""
Retroactive FAV-only backtest for Apr 6 -> Apr 13 2026.

Covers the window between when live V2 was paused (Apr 5) and the end of
the regular season. Simulates what the strategy WOULD HAVE DONE if it had
kept running, to test whether the pause decision looks justified and to
extend our late-regular-season sample.

Pipeline:
  1. Fetch closed NBA game events from PM gamma API (already resolved).
  2. Extract home/away teams, both token IDs, and game outcomes from
     the moneyline market (outcomePrices gives us the winner).
  3. Get pregame price for each token:
       - If token is in nba_game_snapshots.parquet: last tick before game start
       - Else fall back to live PriceHistoryProvider (30-day PM API window)
  4. Run XGBoost + TeamData (with date-fallback for missing snapshots).
  5. Apply V2 FAV filter: model_prob >= 0.60 AND edge >= 7%.
  6. Simulate hold-to-resolution with flat 2% sizing from $1000 bankroll.
"""

import json
import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.DataProviders.PriceHistoryProvider import PriceHistoryProvider
from build_backtest_dataset import (
    load_model, TEAM_INDEX, TEAM_NAME_MAP, calculate_days_rest, load_schedule
)

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS = BASE_DIR / "nba_game_snapshots.parquet"
TEAM_DB = BASE_DIR / "Data" / "TeamData.sqlite"
OUT_CSV = BASE_DIR / "Data" / "backtest" / "analysis" / "retro_late_season_trades.csv"
OUT_REPORT = BASE_DIR / "Data" / "backtest" / "analysis" / "retro_late_season_report.md"

GAMMA_URL = "https://gamma-api.polymarket.com"
NBA_SERIES_ID = "10345"
GAMES_TAG_ID = "100639"

# V2 FAV-leg filter parameters
MIN_EDGE = 0.07
FAV_MIN_CONF = 0.60
FLAT_SIZE_PCT = 0.02
STARTING_BANKROLL = 1000.0

WINDOW_START = "2026-04-06"
WINDOW_END = "2026-04-13"


# ────────────────────────────────────────────────────────────────────────────
# PM event fetch
# ────────────────────────────────────────────────────────────────────────────

def fetch_closed_events(start_date, end_date):
    """Fetch closed NBA game events in [start_date, end_date] by paginating
    all closed events and filtering client-side (gamma API's server-side
    start_date_min/max is unreliable)."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    all_events = []
    offset = 0
    print(f"  Paginating closed events (filtering client-side for {start_date} -> {end_date})...")
    while True:
        r = requests.get(
            f"{GAMMA_URL}/events",
            params={
                "series_id": NBA_SERIES_ID,
                "tag_id": GAMES_TAG_ID,
                "closed": "true",
                "limit": 100,
                "offset": offset,
                "order": "startTime",
                "ascending": "false",  # newest first, we stop when we pass start_date
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break

        in_range_batch = []
        past_window = False
        for e in batch:
            s = e.get("startTime", "")
            if not s:
                continue
            try:
                ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if ts >= end_dt:
                continue  # too new, skip
            if ts < start_dt:
                past_window = True  # we've gone past the start
                continue
            in_range_batch.append(e)

        all_events.extend(in_range_batch)
        offset += 100
        print(f"    offset={offset}: {len(in_range_batch)} in window, total {len(all_events)}")
        if past_window or len(batch) < 100:
            break
    return all_events


def extract_game_info(event):
    """Return a dict with home/away teams, tokens, outcome — or None on failure."""
    slug = event.get("slug", "")
    title = event.get("title", "")
    start_str = event.get("startTime", "")
    if not start_str:
        return None
    try:
        start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    for market in event.get("markets", []):
        question = market.get("question", "")
        if any(x in question for x in ("O/U", "Spread", "Over", "1H")):
            continue

        outcomes = market.get("outcomes", "[]")
        prices = market.get("outcomePrices", "[]")
        tokens = market.get("clobTokenIds", "[]")

        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                continue
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except json.JSONDecodeError:
                continue

        if len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2:
            continue

        # Convention: outcomes[0] = Away, outcomes[1] = Home (per slug nba-{away}-{home}-{date})
        away_name = TEAM_NAME_MAP.get(outcomes[0].strip(), outcomes[0].strip())
        home_name = TEAM_NAME_MAP.get(outcomes[1].strip(), outcomes[1].strip())

        try:
            away_won = float(prices[0]) > 0.5
        except ValueError:
            continue

        return {
            "slug": slug,
            "title": title,
            "home_team": home_name,
            "away_team": away_name,
            "home_token": str(tokens[1]),
            "away_token": str(tokens[0]),
            "winner_side": "away" if away_won else "home",
            "game_start_utc": start_utc,
            "game_date": start_utc.astimezone(
                __import__("zoneinfo").ZoneInfo("America/New_York")
            ).date(),
        }
    return None


# ────────────────────────────────────────────────────────────────────────────
# Pregame price extraction
# ────────────────────────────────────────────────────────────────────────────

def load_snapshots():
    if SNAPSHOTS.exists():
        df = pd.read_parquet(str(SNAPSHOTS))
        df["token_id"] = df["token_id"].astype(str)
        return df
    return pd.DataFrame(columns=["token_id", "ts", "price"])


def get_pregame_price(token_id, game_start_utc, snap_df, provider_cache):
    """Get last price before game start — try snapshots first, then PM API."""
    # Try snapshots
    sub = snap_df[snap_df["token_id"] == token_id]
    if len(sub) > 0:
        cutoff = pd.Timestamp(game_start_utc).tz_localize(None) - pd.Timedelta(minutes=10)
        pre = sub[sub["ts"] <= cutoff]
        if len(pre) > 0:
            return float(pre.sort_values("ts").iloc[-1]["price"]), "snapshots"

    # Fall back to PM live API (30-day retention)
    if token_id in provider_cache:
        history = provider_cache[token_id]
    else:
        history = PriceHistoryProvider().get_price_history(token_id)
        provider_cache[token_id] = history

    if not history:
        return None, "none"

    cutoff_ts = int(game_start_utc.timestamp()) - 600  # 10 min before start
    pre_game = [h for h in history if h.get("t", 0) <= cutoff_ts and "p" in h]
    if not pre_game:
        return None, "none"
    last = max(pre_game, key=lambda h: h["t"])
    return float(last["p"]), "pm_api"


# ────────────────────────────────────────────────────────────────────────────
# Model predictions using TeamData
# ────────────────────────────────────────────────────────────────────────────

def get_available_dates(con):
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return set(r[0] for r in rows)


def find_nearest_date(target_date, available_dates, max_back=45):
    if target_date in available_dates:
        return target_date
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    for offset in range(1, max_back + 1):
        alt = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        if alt in available_dates:
            return alt
    return None


def predict_game(con, available_dates, schedule_df,
                 home_team, away_team, game_date_str, model, calibrator):
    """Return (home_prob, away_prob, stats_date) or None on failure."""
    if home_team not in TEAM_INDEX or away_team not in TEAM_INDEX:
        return None

    stats_date = find_nearest_date(game_date_str, available_dates)
    if stats_date is None:
        return None

    try:
        team_df = pd.read_sql_query(f'SELECT * FROM "{stats_date}"', con)
        home_idx = TEAM_INDEX[home_team]
        away_idx = TEAM_INDEX[away_team]
        if home_idx >= len(team_df) or away_idx >= len(team_df):
            return None
        home_series = team_df.iloc[home_idx]
        away_series = team_df.iloc[away_idx]

        stats = pd.concat([home_series, away_series])
        stats = stats.drop(labels=["TEAM_ID", "TEAM_NAME", "Date"], errors="ignore")

        rest_home = calculate_days_rest(schedule_df, home_team, game_date_str)
        rest_away = calculate_days_rest(schedule_df, away_team, game_date_str)
        stats["Days-Rest-Home"] = rest_home
        stats["Days-Rest-Away"] = rest_away

        feature_data = stats.values.astype(float).reshape(1, -1)

        if calibrator is not None:
            probs = calibrator.predict_proba(feature_data)[0]
        else:
            import xgboost as xgb
            probs = model.predict(xgb.DMatrix(feature_data))[0]

        if isinstance(probs, np.ndarray) and len(probs) >= 2:
            home_prob = float(probs[1])
            away_prob = float(probs[0])
        else:
            home_prob = float(probs) if not isinstance(probs, np.ndarray) else float(probs[0])
            away_prob = 1 - home_prob
        return home_prob, away_prob, stats_date
    except Exception as e:
        print(f"    predict error: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Fetching closed PM events {WINDOW_START} -> {WINDOW_END}...")
    events = fetch_closed_events(WINDOW_START, WINDOW_END)
    print(f"  {len(events)} events")

    print("Parsing game info + outcomes from moneyline markets...")
    games = [g for g in (extract_game_info(e) for e in events) if g is not None]
    print(f"  {len(games)} games with parsable ML markets")

    print("Loading snapshots for pregame pricing...")
    snap_df = load_snapshots()
    print(f"  {len(snap_df):,} snapshot rows")

    print("Loading model + schedule...")
    model, calibrator = load_model()
    teams_con = sqlite3.connect(str(TEAM_DB))
    available_dates = get_available_dates(teams_con)
    print(f"  TeamData has {len(available_dates)} date snapshots (latest: "
          f"{max(available_dates) if available_dates else 'none'})")
    schedule_df = load_schedule()

    # Pipeline
    print("\nBuilding per-game records...")
    records = []
    provider_cache = {}
    for i, g in enumerate(games, 1):
        if i % 10 == 0 or i == len(games):
            print(f"  [{i}/{len(games)}] {g['away_team']} @ {g['home_team']} ({g['game_date']})")

        home_price, hsrc = get_pregame_price(
            g["home_token"], g["game_start_utc"], snap_df, provider_cache
        )
        away_price, asrc = get_pregame_price(
            g["away_token"], g["game_start_utc"], snap_df, provider_cache
        )
        if home_price is None or away_price is None:
            continue

        # Normalize (sometimes prices sum slightly above/below 1 — typically minor)
        total = home_price + away_price
        if total <= 0:
            continue

        pred = predict_game(
            teams_con, available_dates, schedule_df,
            g["home_team"], g["away_team"],
            g["game_date"].strftime("%Y-%m-%d"),
            model, calibrator,
        )
        if pred is None:
            continue
        home_prob, away_prob, stats_date = pred

        home_edge = home_prob - home_price
        away_edge = away_prob - away_price

        records.append({
            "game_date": g["game_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_price": round(home_price, 4),
            "away_price": round(away_price, 4),
            "model_home_prob": round(home_prob, 4),
            "model_away_prob": round(away_prob, 4),
            "edge_home_pm": round(home_edge, 4),
            "edge_away_pm": round(away_edge, 4),
            "winner_side": g["winner_side"],
            "stats_as_of": stats_date,
            "price_src_home": hsrc,
            "price_src_away": asrc,
        })

    teams_con.close()
    df = pd.DataFrame(records)
    print(f"\n{len(df)} games with full predictions + pricing")

    # Apply FAV-leg filter
    print("\nApplying FAV-leg filter (model_prob >= 60%, edge >= 7%)...")
    bets = []
    for _, r in df.iterrows():
        # Pick bet side: highest edge above MIN_EDGE with conf >= FAV_MIN_CONF
        candidates = []
        if r["edge_home_pm"] >= MIN_EDGE and r["model_home_prob"] >= FAV_MIN_CONF:
            candidates.append(("home", r["edge_home_pm"], r["model_home_prob"],
                              r["home_price"]))
        if r["edge_away_pm"] >= MIN_EDGE and r["model_away_prob"] >= FAV_MIN_CONF:
            candidates.append(("away", r["edge_away_pm"], r["model_away_prob"],
                              r["away_price"]))
        if not candidates:
            continue
        side, edge, prob, entry = max(candidates, key=lambda x: x[1])
        bets.append({
            **r.to_dict(),
            "bet_side": side,
            "bet_edge": round(edge, 4),
            "bet_model_prob": round(prob, 4),
            "entry_price": round(entry, 4),
            "bet_won": (side == r["winner_side"]),
        })

    bets_df = pd.DataFrame(bets)
    print(f"  {len(bets_df)} FAV bets would have been placed")

    # Simulate flat 2%
    bankroll = STARTING_BANKROLL
    pnls = []
    for _, b in bets_df.iterrows():
        stake = bankroll * FLAT_SIZE_PCT
        if b["bet_won"]:
            pnl = stake * (1.0 / b["entry_price"] - 1)
        else:
            pnl = -stake
        bankroll += pnl
        pnls.append(pnl)
    if not bets_df.empty:
        bets_df["pnl"] = pnls
        bets_df["bankroll_after"] = [STARTING_BANKROLL + sum(pnls[:i+1]) for i in range(len(pnls))]

    # Results
    wins = int(bets_df["bet_won"].sum()) if not bets_df.empty else 0
    losses = len(bets_df) - wins
    wr = wins / len(bets_df) if len(bets_df) else 0.0
    total_pnl = sum(pnls)
    total_staked = len(pnls) * STARTING_BANKROLL * FLAT_SIZE_PCT
    roi = total_pnl / total_staked if total_staked > 0 else 0

    print("\n" + "=" * 70)
    print(f"RETRO FAV BACKTEST — {WINDOW_START} -> {WINDOW_END}")
    print("=" * 70)
    print(f"Bets placed:       {len(bets_df)}")
    print(f"Wins / Losses:     {wins} / {losses}")
    print(f"Win rate:          {wr:.1%}")
    print(f"Total P&L:         ${total_pnl:+.2f}")
    print(f"Final bankroll:    ${bankroll:.2f}")
    print(f"ROI:               {roi:+.2%}")

    if not bets_df.empty:
        print(f"\nPer-bet summary:")
        for _, b in bets_df.iterrows():
            outcome = "WIN " if b["bet_won"] else "LOSS"
            print(f"  {b['game_date']}  {b['away_team']} @ {b['home_team']}  "
                  f"bet {b['bet_side']} @ ${b['entry_price']:.3f}  "
                  f"edge {b['bet_edge']:+.1%}  {outcome}  pnl ${b['pnl']:+.2f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    bets_df.to_csv(str(OUT_CSV), index=False)

    # Report
    lines = [
        f"# Retroactive FAV Backtest ({WINDOW_START} -> {WINDOW_END})",
        "",
        "**Goal:** Simulate what V2 would have done if it had kept running after being "
        "paused on 2026-04-05, using actual game outcomes and PM pregame pricing.",
        "",
        "**Scope:** FAV leg only (hold-to-resolution, no ESPN WP exits required).",
        "",
        "## Pipeline",
        "",
        f"- Fetched {len(events)} closed NBA game events from PM gamma API",
        f"- Parsed {len(games)} games with usable ML markets",
        f"- {len(df)} games with full pricing + model prediction",
        f"- {len(bets_df)} FAV bets passed filter (edge >= 7%, conf >= 60%)",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Bets | {len(bets_df)} |",
        f"| W / L | {wins} / {losses} |",
        f"| Win rate | {wr:.1%} |",
        f"| Total P&L | ${total_pnl:+.2f} |",
        f"| Final bankroll | ${bankroll:.2f} |",
        f"| ROI | {roi:+.2%} |",
    ]
    if not bets_df.empty:
        lines += [
            "",
            "## Per-bet detail",
            "",
            "| Date | Game | Side | Entry | Edge | Outcome | P&L |",
            "|------|------|------|-------|------|---------|-----|",
        ]
        for _, b in bets_df.iterrows():
            lines.append(
                f"| {b['game_date']} | {b['away_team']} @ {b['home_team']} | "
                f"{b['bet_side']} | ${b['entry_price']:.3f} | "
                f"{b['bet_edge']:+.1%} | "
                f"{'WIN' if b['bet_won'] else 'LOSS'} | ${b['pnl']:+.2f} |"
            )
    lines += [
        "",
        "## Context",
        "",
        "Compare to live V2 FAV results over Mar 16 -> Apr 5 (N=5, WR=40%, ROI=-63%).",
        "A similar or worse result here confirms the regime hypothesis. A much better "
        "result suggests the Apr 5 pause was premature.",
        "",
        "### Caveats",
        "- TeamData snapshots only go through Mar 15, so predictions use stale stats "
        "(same as what V2 live was doing in its final weeks, so the comparison is fair).",
        "- Pregame prices pulled from tick snapshots where available (Apr 6-7), "
        "falling back to PM API for later games (still within 30-day retention).",
        "- Sample is small (<10-20 FAV bets expected); directional only.",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {OUT_REPORT}")
    print(f"Trades CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
