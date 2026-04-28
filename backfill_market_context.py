"""
Backfill market-context features for the 25 closed live V2 trades.

PM retains 30 days of price history, so trades placed Mar 18 onwards are
still queryable (today = 2026-04-17). We fetch each bet-side token's
price history, keep only ticks BEFORE the entry_time (so the computed
features match what would have been logged at bet time), and extract:
  - open_price, price_max, price_min, price_range, open_move

`rest_diff` is pulled from Data/OddsData.sqlite (2025-26 table) which
already stores Days_Rest_Home/Away for every scheduled game.

Writes a self-contained enriched CSV for analysis; does NOT modify
positions.json or trades.json (both are source-of-truth for live state).
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.DataProviders.PriceHistoryProvider import PriceHistoryProvider

BASE_DIR = Path(__file__).resolve().parent
POSITIONS_FILE = BASE_DIR / "Data" / "paper_trading_v2" / "positions.json"
TRADES_FILE = BASE_DIR / "Data" / "paper_trading_v2" / "trades.json"
SCHEDULE_CSV = BASE_DIR / "Data" / "nba-2025-UTC.csv"
OUT_CSV = BASE_DIR / "Data" / "backtest" / "analysis" / "live_trades_enriched.csv"


def build_rest_lookup():
    """Build a per-team dict of sorted game dates for rest-day computation.

    Uses the same schedule file the live paper_trader reads (nba-2025-UTC.csv).
    Returns {team_name: [sorted datetime list]}.
    """
    df = pd.read_csv(str(SCHEDULE_CSV), parse_dates=["Date"], dayfirst=True)
    team_games = {}
    for _, row in df.iterrows():
        for t in (row["Home Team"], row["Away Team"]):
            if pd.notna(t):
                team_games.setdefault(t, []).append(row["Date"])
    for t in team_games:
        team_games[t] = sorted(team_games[t])
    return team_games


def days_rest_for(team_games, team, bet_dt):
    """Days between bet_dt and the team's most recent prior game.

    Mirrors the paper_trader logic: if no prior game, defaults to 7 days.
    """
    games = team_games.get(team, [])
    prior = [g for g in games if g.date() < bet_dt.date()]
    if not prior:
        return 7
    last = prior[-1]
    return (bet_dt.date() - last.date()).days


def lookup_rest(team_games, home_team, away_team, bet_time):
    """Return (days_rest_home, days_rest_away) using schedule history."""
    rest_h = days_rest_for(team_games, home_team, bet_time)
    rest_a = days_rest_for(team_games, away_team, bet_time)
    return rest_h, rest_a


def fetch_pre_entry_history(token_id, entry_time):
    """Fetch PM price history and keep only ticks BEFORE entry_time."""
    if not token_id:
        return []
    try:
        hist = PriceHistoryProvider().get_price_history(token_id)
    except Exception as e:
        print(f"    Fetch failed: {e}")
        return []
    if not hist:
        return []
    entry_ts = int(entry_time.timestamp())
    pre = [h for h in hist if h.get("t", 0) <= entry_ts and "p" in h]
    return pre


def backfill_one(pos, entry_log, team_games):
    """Build an enriched record for one closed position."""
    home_team = pos["home_team"]
    away_team = pos["away_team"]
    bet_side = pos["bet_side"]
    entry_time = datetime.fromisoformat(pos["entry_time"])
    bet_date = entry_time.date()

    token = pos["home_token_id"] if bet_side == "home" else pos["away_token_id"]
    hist = fetch_pre_entry_history(token, entry_time)
    if hist:
        prices = [h["p"] for h in hist]
        open_price = round(prices[0], 4)
        p_max = round(max(prices), 4)
        p_min = round(min(prices), 4)
        p_range = round(p_max - p_min, 4)
        open_move = round(pos["entry_price"] - open_price, 4)
        n_ticks = len(prices)
    else:
        open_price = p_max = p_min = p_range = open_move = None
        n_ticks = 0

    rest_h, rest_a = lookup_rest(team_games, home_team, away_team, entry_time)
    rest_diff = (rest_h - rest_a) if bet_side == "home" else (rest_a - rest_h)

    return {
        "position_id": pos.get("game_key", "") + f"_{bet_date.strftime('%Y%m%d')}",
        "entry_time": pos["entry_time"],
        "game": f"{away_team} @ {home_team}",
        "home_team": home_team,
        "away_team": away_team,
        "bet_side": bet_side,
        "leg": "FAV" if pos["is_favorite"] else "DOG",
        "model_prob": pos["model_prob"],
        "entry_price": pos["entry_price"],
        "bet_edge": pos["bet_edge"],
        "bet_amount": pos["bet_amount"],
        "pnl": pos.get("pnl"),
        "bet_won": (pos.get("pnl") or 0) > 0,
        "exit_type": pos.get("exit_type") or "resolved",
        # Backfilled market-context features
        "open_price": open_price,
        "price_max": p_max,
        "price_min": p_min,
        "price_range": p_range,
        "open_move": open_move,
        "n_ticks": n_ticks,
        "days_rest_home": rest_h,
        "days_rest_away": rest_a,
        "rest_diff": rest_diff,
    }


def main():
    print("Loading live positions...")
    with open(POSITIONS_FILE) as f:
        positions = json.load(f)

    # Only keep closed positions (resolved or exited with pnl known)
    closed = {k: v for k, v in positions.items()
              if v.get("status") != "open" or v.get("pnl") is not None}
    print(f"  {len(closed)} closed positions (of {len(positions)} total)")

    print("Loading schedule + rest lookup from nba-2025-UTC.csv...")
    team_games = build_rest_lookup()
    print(f"  {len(team_games)} teams indexed")

    # Map entries -> enriched records
    print("\nBackfilling PM price history (rate-limited)...")
    records = []
    for i, (pid, pos) in enumerate(sorted(closed.items(),
                                          key=lambda kv: kv[1]["entry_time"]), 1):
        print(f"  [{i}/{len(closed)}] {pos['entry_time'][:10]} "
              f"{pos['away_team']} @ {pos['home_team']} ({pos['bet_side']})")
        rec = backfill_one(pos, None, team_games)
        records.append(rec)
        time.sleep(0.2)  # gentle rate-limit

    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(OUT_CSV), index=False)

    # Summary
    have_open = df["open_price"].notna().sum()
    have_rest = df["rest_diff"].notna().sum()
    print(f"\nBackfill summary:")
    print(f"  Total closed trades:  {len(df)}")
    print(f"  With PM price history: {have_open} / {len(df)}")
    print(f"  With rest_diff:       {have_rest} / {len(df)}")
    print(f"\nSaved: {OUT_CSV}")

    # Quick look at the feature distributions
    print("\nEnriched feature distributions (where available):")
    for col in ["open_move", "price_range", "rest_diff", "n_ticks"]:
        s = df[col].dropna()
        if len(s) > 0:
            print(f"  {col:<14s} n={len(s)} min={s.min():.3f} "
                  f"med={s.median():.3f} max={s.max():.3f}")


if __name__ == "__main__":
    main()
