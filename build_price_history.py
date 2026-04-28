"""
Fetch full Polymarket price evolution for every NBA game with available history.

Output: Data/backtest/nba_price_history.csv
  Columns: game_date, home_team, away_team, timestamp, time_utc, home_price, away_price, minutes_to_start

Uses cached polymarket_events_cache.json from build_backtest_dataset.py.
Only games from ~Jan 28, 2026 onward have price history available.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"
CACHE_EVENTS = BACKTEST_DIR / "polymarket_events_cache.json"
CACHE_RAW_HISTORIES = BACKTEST_DIR / "price_history_raw_cache.json"
OUTPUT_CSV = BACKTEST_DIR / "nba_price_history.csv"

CLOB_URL = "https://clob.polymarket.com"
# Price history only available from ~Jan 27 2026
CUTOFF_TS = int(datetime(2026, 1, 27, tzinfo=timezone.utc).timestamp())


def main():
    # Load events cache
    if not CACHE_EVENTS.exists():
        print("Run build_backtest_dataset.py first to generate event cache.")
        return

    with open(CACHE_EVENTS) as f:
        events = json.load(f)

    # Filter to events with tokens and after cutoff
    eligible = [
        e for e in events
        if e.get("home_token") and e.get("start_ts", 0) >= CUTOFF_TS
    ]
    print(f"Eligible games with price history: {len(eligible)}")

    # Fetch raw histories (with caching)
    raw_histories = {}
    if CACHE_RAW_HISTORIES.exists():
        with open(CACHE_RAW_HISTORIES) as f:
            raw_histories = json.load(f)
        print(f"Loaded {len(raw_histories)} cached raw histories")

    to_fetch = [e for e in eligible if e["home_token"] not in raw_histories]
    print(f"Need to fetch: {len(to_fetch)} new histories")

    for i, event in enumerate(to_fetch):
        token = event["home_token"]
        try:
            resp = requests.get(
                f"{CLOB_URL}/prices-history",
                params={"market": token, "interval": "max"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            history = data if isinstance(data, list) else data.get("history", [])
            if history:
                raw_histories[token] = history
        except Exception as e:
            print(f"  Error: {event['home_team']} vs {event['away_team']}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(to_fetch)}...")
        time.sleep(0.3)

    # Save raw cache
    with open(CACHE_RAW_HISTORIES, "w") as f:
        json.dump(raw_histories, f)
    print(f"Cached {len(raw_histories)} raw histories")

    # Build flat dataset
    print("Building price history dataset...")
    rows = []
    for event in eligible:
        token = event["home_token"]
        history = raw_histories.get(token, [])
        if not history:
            continue

        game_date = event["game_date_et"]
        home = event["home_team"]
        away = event["away_team"]
        start_ts = event["start_ts"]

        for point in history:
            t = point["t"]
            p = float(point["p"])
            minutes_to_start = (start_ts - t) / 60  # positive = before game

            rows.append({
                "game_date": game_date,
                "home_team": home,
                "away_team": away,
                "timestamp": t,
                "time_utc": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "home_price": round(p, 4),
                "away_price": round(1 - p, 4),
                "minutes_to_start": round(minutes_to_start, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(str(OUTPUT_CSV), index=False)

    games_covered = df.groupby(["game_date", "home_team", "away_team"]).ngroups if len(df) > 0 else 0
    print(f"\nSaved {len(df):,} price points across {games_covered} games to {OUTPUT_CSV.name}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}" if len(df) > 0 else "")

    # Quick stats
    if len(df) > 0:
        per_game = df.groupby(["game_date", "home_team"]).size()
        print(f"Avg data points per game: {per_game.mean():.0f}")
        print(f"Min: {per_game.min()}, Max: {per_game.max()}")


if __name__ == "__main__":
    main()
