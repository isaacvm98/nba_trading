"""
Compute microstructure features from 10-min PM price snapshots.

Inputs:
  - nba_game_snapshots.parquet         (token_id, ts, price)
  - Data/backtest/nba_backtest_dataset.csv  (token -> game mapping, backtest)
  - Data/paper_trading_v2/positions.json   (token -> game mapping, live)
  - Data/nba-2025-UTC.csv              (game start times)

Outputs:
  - Data/backtest/analysis/tick_features_by_token.csv
      one row per (token_id, game_date, home_team, away_team, bet_side):
      { open_price, close_price, price_max, price_min, price_range,
        open_move, vol_30m, vol_2h, vol_6h, momentum_30m, momentum_2h,
        n_large_moves, n_direction_changes, n_ticks, hours_of_data }

Features are computed on the PRE-GAME window only: all ticks strictly before
(game_start - 10min).  This prevents in-game price leakage into features.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS = BASE_DIR / "nba_game_snapshots.parquet"
BACKTEST_DATASET = BASE_DIR / "Data" / "backtest" / "nba_backtest_dataset.csv"
POSITIONS_FILE = BASE_DIR / "Data" / "paper_trading_v2" / "positions.json"
SCHEDULE_CSV = BASE_DIR / "Data" / "nba-2025-UTC.csv"
OUT_CSV = BASE_DIR / "Data" / "backtest" / "analysis" / "tick_features_by_token.csv"

# Cut features off this many minutes before game start to avoid in-game leakage
PREGAME_CUTOFF_MIN = 10


# ────────────────────────────────────────────────────────────────────────────
# Token -> game mapping (combines backtest + live positions)
# ────────────────────────────────────────────────────────────────────────────

def build_token_game_map():
    """Return a DataFrame: token_id, side, home_team, away_team, game_date, source."""
    rows = []

    # Backtest: every eligible bet has home_token + away_token
    bt = pd.read_csv(str(BACKTEST_DATASET))
    bt = bt[bt["pm_home_token"].notna() & bt["pm_away_token"].notna()].copy()
    bt["game_date"] = pd.to_datetime(bt["game_date"])
    for _, r in bt.iterrows():
        rows.append({
            "token_id": str(r["pm_home_token"]),
            "side": "home",
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "game_date": r["game_date"].date(),
            "source": "backtest",
        })
        rows.append({
            "token_id": str(r["pm_away_token"]),
            "side": "away",
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "game_date": r["game_date"].date(),
            "source": "backtest",
        })

    # Live positions
    with open(POSITIONS_FILE) as f:
        pos = json.load(f)
    for p in pos.values():
        game_date = pd.to_datetime(p["entry_time"][:10]).date()
        if p.get("home_token_id"):
            rows.append({
                "token_id": str(p["home_token_id"]),
                "side": "home",
                "home_team": p["home_team"],
                "away_team": p["away_team"],
                "game_date": game_date,
                "source": "live",
            })
        if p.get("away_token_id"):
            rows.append({
                "token_id": str(p["away_token_id"]),
                "side": "away",
                "home_team": p["home_team"],
                "away_team": p["away_team"],
                "game_date": game_date,
                "source": "live",
            })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["token_id", "home_team", "away_team", "game_date"]
    )
    return df


# ────────────────────────────────────────────────────────────────────────────
# Game start time lookup from the schedule CSV
# ────────────────────────────────────────────────────────────────────────────

def build_schedule():
    """Return DataFrame with game_start_utc per (home, away) indexed by local date.

    The CSV timestamps are UTC; NBA backtest dates are local-calendar. Games
    starting after ~midnight UTC (evening ET) map to the PRIOR UTC date in
    local-calendar terms, so we index by BOTH UTC-date and (UTC-date minus 1
    day) and let the merge hit either.
    """
    df = pd.read_csv(str(SCHEDULE_CSV), parse_dates=["Date"], dayfirst=True)
    df = df.rename(columns={"Home Team": "home_team", "Away Team": "away_team"})
    df["game_start_utc"] = df["Date"]

    # Create two candidate local-dates per game: the UTC date and UTC date - 1
    rows = []
    for _, r in df.iterrows():
        for delta in (0, -1):
            rows.append({
                "game_date": (r["Date"] + pd.Timedelta(days=delta)).date(),
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "game_start_utc": r["game_start_utc"],
            })
    expanded = pd.DataFrame(rows)
    return expanded


def attach_game_start(token_map, schedule):
    """Join schedule game_start_utc onto the token map.

    Uses the expanded schedule (each game appears at UTC date and UTC date - 1)
    so either matches the local game_date in the backtest/live data.  If a
    token matches multiple candidates, keep the closest game_start_utc to the
    game_date's noon (disambiguates double-headers / scheduling anomalies).
    """
    merged = token_map.merge(
        schedule,
        on=["game_date", "home_team", "away_team"],
        how="left",
    )
    # Deduplicate: one row per token. Keep the schedule row whose game_start_utc
    # is closest to noon of the game_date (simple disambiguation).
    merged["_gd_noon"] = pd.to_datetime(merged["game_date"]) + pd.Timedelta(hours=12)
    merged["_offset"] = (merged["game_start_utc"] - merged["_gd_noon"]).abs()
    merged = (merged.sort_values("_offset")
                     .drop_duplicates(subset=["token_id", "game_date",
                                              "home_team", "away_team"],
                                      keep="first")
                     .drop(columns=["_gd_noon", "_offset"]))
    missing = merged["game_start_utc"].isna().sum()
    if missing:
        print(f"  Warning: {missing} tokens missing game_start_utc (schedule mismatch)")
    return merged


# ────────────────────────────────────────────────────────────────────────────
# Feature computation
# ────────────────────────────────────────────────────────────────────────────

def compute_features(ticks, cutoff_ts):
    """Compute microstructure features from pre-cutoff ticks only.

    ticks: DataFrame with columns [ts (datetime), price (float)], sorted.
    cutoff_ts: pd.Timestamp; ticks with ts >= cutoff_ts are excluded.

    Returns dict or None if < 2 usable ticks.
    """
    pre = ticks[ticks["ts"] < cutoff_ts]
    if len(pre) < 2:
        return None

    prices = pre["price"].to_numpy()
    ts = pre["ts"]

    open_price = float(prices[0])
    close_price = float(prices[-1])
    price_max = float(prices.max())
    price_min = float(prices.min())
    price_range = price_max - price_min
    open_move = close_price - open_price

    def window_prices(minutes):
        start = cutoff_ts - pd.Timedelta(minutes=minutes)
        mask = (pre["ts"] >= start) & (pre["ts"] < cutoff_ts)
        return pre.loc[mask, "price"].to_numpy()

    def returns_std(arr):
        if len(arr) < 2:
            return 0.0
        return float(np.std(np.diff(arr), ddof=0))

    def cum_move(arr):
        if len(arr) < 2:
            return 0.0
        return float(arr[-1] - arr[0])

    last_30 = window_prices(30)
    last_2h = window_prices(120)
    last_6h = window_prices(360)

    returns = np.diff(prices)
    n_large_moves = int(np.sum(np.abs(returns) >= 0.01))
    # direction changes: count of sign flips in non-zero returns
    nonzero = returns[returns != 0]
    if len(nonzero) >= 2:
        n_direction_changes = int(np.sum(np.diff(np.sign(nonzero)) != 0))
    else:
        n_direction_changes = 0

    hours = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0

    return {
        "open_price": round(open_price, 4),
        "close_price": round(close_price, 4),
        "price_max": round(price_max, 4),
        "price_min": round(price_min, 4),
        "price_range": round(price_range, 4),
        "open_move": round(open_move, 4),
        "vol_30m": round(returns_std(last_30), 5),
        "vol_2h": round(returns_std(last_2h), 5),
        "vol_6h": round(returns_std(last_6h), 5),
        "momentum_30m": round(cum_move(last_30), 4),
        "momentum_2h": round(cum_move(last_2h), 4),
        "n_large_moves": n_large_moves,
        "n_direction_changes": n_direction_changes,
        "n_ticks": len(pre),
        "hours_of_data": round(hours, 2),
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading snapshots parquet...")
    snap = pd.read_parquet(str(SNAPSHOTS))
    snap["token_id"] = snap["token_id"].astype(str)
    snap = snap.sort_values(["token_id", "ts"]).reset_index(drop=True)
    print(f"  {len(snap):,} ticks, {snap['token_id'].nunique():,} tokens")

    print("\nBuilding token -> game map (backtest + live)...")
    token_map = build_token_game_map()
    print(f"  {len(token_map)} token records (both sides)")

    print("Loading schedule (game start times)...")
    schedule = build_schedule()
    print(f"  {len(schedule)} games in schedule")

    token_map = attach_game_start(token_map, schedule)

    # Filter token_map to tokens that actually appear in snapshots
    snap_tokens = set(snap["token_id"].unique())
    covered = token_map[token_map["token_id"].isin(snap_tokens)].copy()
    print(f"\nTokens with tick coverage: {len(covered)} of {len(token_map)}")

    # Group snapshots by token for fast lookup
    print("Grouping snapshots by token...")
    snap_by_tok = {tok: g for tok, g in snap.groupby("token_id", sort=False)}

    # Compute features per token
    print("Computing microstructure features...")
    results = []
    skipped_no_schedule = skipped_no_ticks = 0
    for i, row in enumerate(covered.itertuples(index=False), 1):
        if i % 100 == 0:
            print(f"  {i}/{len(covered)}...")

        if pd.isna(row.game_start_utc):
            skipped_no_schedule += 1
            continue

        cutoff = pd.Timestamp(row.game_start_utc) - pd.Timedelta(minutes=PREGAME_CUTOFF_MIN)
        ticks = snap_by_tok.get(row.token_id)
        if ticks is None or len(ticks) == 0:
            skipped_no_ticks += 1
            continue

        feats = compute_features(ticks, cutoff)
        if feats is None:
            continue

        results.append({
            "token_id": row.token_id,
            "game_date": row.game_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "side": row.side,
            "source": row.source,
            "game_start_utc": row.game_start_utc,
            **feats,
        })

    print(f"\nFeature rows: {len(results)}")
    print(f"Skipped (no schedule match): {skipped_no_schedule}")
    print(f"Skipped (no ticks or <2 pre-game): {skipped_no_ticks}")

    out = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(OUT_CSV), index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Quick summary
    print("\nFeature summary (over all computed rows):")
    for col in ["open_move", "price_range", "vol_2h", "vol_6h",
                "momentum_30m", "momentum_2h",
                "n_large_moves", "n_direction_changes", "n_ticks", "hours_of_data"]:
        s = out[col]
        print(f"  {col:<22s} n={s.notna().sum():4d}  "
              f"min={s.min():>+8.4f}  med={s.median():>+8.4f}  "
              f"max={s.max():>+8.4f}")


if __name__ == "__main__":
    main()
