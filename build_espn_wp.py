"""
Fetch ESPN win probability data for all games in the backtest dataset.

For each completed game, fetches:
- Pregame home WP (ESPN's statistical model prediction before tipoff)
- Full play-by-play WP curve (for live exit signal backtesting)
- Quarter-end WP snapshots (Q1, Q2, Q3, Q4 end probabilities)

Output: Data/backtest/espn_wp_cache.json  (raw cache, keyed by event_id)
        Data/backtest/espn_wp_backtest.csv (flat file merged with backtest games)

Usage:
    python build_espn_wp.py              # Fetch all (uses cache)
    python build_espn_wp.py --no-cache   # Re-fetch everything
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"
DATASET_CSV = BACKTEST_DIR / "nba_backtest_dataset.csv"
ESPN_CACHE = BACKTEST_DIR / "espn_wp_cache.json"
OUTPUT_CSV = BACKTEST_DIR / "espn_wp_backtest.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# Team abbreviation -> full name (from ESPNProvider)
ABBR_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'GS': 'Golden State Warriors',
    'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans',
    'NO': 'New Orleans Pelicans', 'NY': 'New York Knicks', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs', 'SA': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'UTAH': 'Utah Jazz',
    'WAS': 'Washington Wizards', 'WSH': 'Washington Wizards',
}

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("https://", adapter)


def fetch_scoreboard(date_str):
    """Fetch ESPN scoreboard for a date (YYYYMMDD format)."""
    resp = session.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_str},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_game_summary(event_id):
    """Fetch full game summary including win probability."""
    resp = session.get(
        f"{ESPN_BASE}/summary",
        params={"event": event_id},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def parse_scoreboard_games(scoreboard_data):
    """Parse scoreboard into list of {event_id, home_team, away_team}."""
    games = []
    for event in scoreboard_data.get("events", []):
        event_id = event.get("id")
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])

        home_team = away_team = None
        for comp in competitors:
            team_data = comp.get("team", {})
            abbr = team_data.get("abbreviation", "")
            name = ABBR_TO_NAME.get(abbr, team_data.get("displayName", ""))
            if comp.get("homeAway") == "home":
                home_team = name
            else:
                away_team = name

        if event_id and home_team and away_team:
            games.append({
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
            })
    return games


def extract_wp_data(summary):
    """Extract WP snapshots from a game summary."""
    win_prob = summary.get("winprobability", [])
    plays = summary.get("plays", [])

    if not win_prob:
        return None

    play_details = {str(p.get("id")): p for p in plays}

    # Pregame WP (first data point)
    pregame_wp = win_prob[0].get("homeWinPercentage", 0.5)

    # Final WP
    final_wp = win_prob[-1].get("homeWinPercentage", 0.5)

    # Quarter-end snapshots
    quarter_end = {}
    for wp in win_prob:
        play_id = str(wp.get("playId", ""))
        play = play_details.get(play_id, {})
        period = play.get("period", {}).get("number", 0)
        home_wp = wp.get("homeWinPercentage", 0.5)
        home_score = play.get("homeScore", 0)
        away_score = play.get("awayScore", 0)

        # Track last entry per period as "end of quarter"
        quarter_end[period] = {
            "home_wp": home_wp,
            "home_score": home_score,
            "away_score": away_score,
        }

    # Min/max WP during game (volatility measure)
    all_wp = [wp.get("homeWinPercentage", 0.5) for wp in win_prob]
    wp_min = min(all_wp)
    wp_max = max(all_wp)

    # Lead changes: count times WP crosses 0.5
    lead_changes = 0
    for i in range(1, len(all_wp)):
        if (all_wp[i - 1] < 0.5 and all_wp[i] >= 0.5) or (all_wp[i - 1] >= 0.5 and all_wp[i] < 0.5):
            lead_changes += 1

    return {
        "pregame_home_wp": round(pregame_wp, 4),
        "final_home_wp": round(final_wp, 4),
        "wp_min": round(wp_min, 4),
        "wp_max": round(wp_max, 4),
        "lead_changes": lead_changes,
        "total_plays": len(win_prob),
        "q1_end": quarter_end.get(1, {}),
        "q2_end": quarter_end.get(2, {}),
        "q3_end": quarter_end.get(3, {}),
        "q4_end": quarter_end.get(4, {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch ESPN WP for backtest games")
    parser.add_argument("--no-cache", action="store_true", help="Re-fetch all data")
    args = parser.parse_args()

    # Load backtest dataset
    df = pd.read_csv(str(DATASET_CSV))
    game_dates = sorted(df["game_date"].unique())
    print(f"Backtest dataset: {len(df)} games across {len(game_dates)} dates")

    # Load existing cache
    cache = {}
    if not args.no_cache and ESPN_CACHE.exists():
        with open(ESPN_CACHE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached ESPN WP entries")

    # Build set of (date, home, away) we need
    needed_games = set()
    for _, row in df.iterrows():
        key = f"{row['game_date']}_{row['home_team']}_{row['away_team']}"
        if key not in cache:
            needed_games.add(row["game_date"])

    needed_dates = sorted(needed_games)
    print(f"Dates to fetch: {len(needed_dates)}")

    # Fetch by date
    errors = 0
    fetched = 0
    for i, date in enumerate(needed_dates):
        date_espn = date.replace("-", "")  # YYYYMMDD

        try:
            scoreboard = fetch_scoreboard(date_espn)
            espn_games = parse_scoreboard_games(scoreboard)

            for game in espn_games:
                # Match to our dataset
                matching = df[
                    (df["game_date"] == date)
                    & (df["home_team"] == game["home_team"])
                    & (df["away_team"] == game["away_team"])
                ]
                if matching.empty:
                    continue

                key = f"{date}_{game['home_team']}_{game['away_team']}"
                if key in cache:
                    continue

                # Fetch full summary
                try:
                    summary = fetch_game_summary(game["event_id"])
                    wp_data = extract_wp_data(summary)
                    if wp_data:
                        wp_data["event_id"] = game["event_id"]
                        cache[key] = wp_data
                        fetched += 1
                    time.sleep(0.5)  # Rate limit
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  Error fetching {key}: {e}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error fetching scoreboard for {date}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(needed_dates)} dates, {fetched} games fetched, {errors} errors")
            # Save intermediate cache
            with open(ESPN_CACHE, "w") as f:
                json.dump(cache, f)

        time.sleep(0.3)

    # Save final cache
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(ESPN_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCached {len(cache)} ESPN WP entries to {ESPN_CACHE.name}")

    # Build flat CSV
    rows = []
    for _, row in df.iterrows():
        key = f"{row['game_date']}_{row['home_team']}_{row['away_team']}"
        wp = cache.get(key, {})

        rows.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win": row["home_win"],
            # ESPN WP
            "espn_pregame_home": wp.get("pregame_home_wp"),
            "espn_final_home": wp.get("final_home_wp"),
            "espn_wp_min": wp.get("wp_min"),
            "espn_wp_max": wp.get("wp_max"),
            "espn_lead_changes": wp.get("lead_changes"),
            "espn_total_plays": wp.get("total_plays"),
            # Quarter-end WPs (for live exit backtesting)
            "espn_q1_end_wp": wp.get("q1_end", {}).get("home_wp"),
            "espn_q1_end_home_score": wp.get("q1_end", {}).get("home_score"),
            "espn_q1_end_away_score": wp.get("q1_end", {}).get("away_score"),
            "espn_q2_end_wp": wp.get("q2_end", {}).get("home_wp"),
            "espn_q2_end_home_score": wp.get("q2_end", {}).get("home_score"),
            "espn_q2_end_away_score": wp.get("q2_end", {}).get("away_score"),
            "espn_q3_end_wp": wp.get("q3_end", {}).get("home_wp"),
            "espn_q3_end_home_score": wp.get("q3_end", {}).get("home_score"),
            "espn_q3_end_away_score": wp.get("q3_end", {}).get("away_score"),
            "espn_q4_end_wp": wp.get("q4_end", {}).get("home_wp"),
            # Model + PM for comparison
            "model_home_prob": row.get("model_home_prob"),
            "pm_pregame_home": row.get("pm_pregame_home"),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(str(OUTPUT_CSV), index=False)

    # Summary
    has_espn = out_df["espn_pregame_home"].notna().sum()
    print(f"Saved {len(out_df)} rows to {OUTPUT_CSV.name}")
    print(f"Games with ESPN WP: {has_espn}/{len(out_df)}")

    if has_espn > 0:
        espn_sub = out_df[out_df["espn_pregame_home"].notna() & out_df["home_win"].notna()]
        espn_sub = espn_sub.copy()
        espn_sub["espn_correct"] = (
            ((espn_sub["espn_pregame_home"] >= 0.5) & (espn_sub["home_win"] == 1))
            | ((espn_sub["espn_pregame_home"] < 0.5) & (espn_sub["home_win"] == 0))
        )
        print(f"\nESPN pregame accuracy: {espn_sub['espn_correct'].mean():.1%} ({int(espn_sub['espn_correct'].sum())}/{len(espn_sub)})")

        if espn_sub["model_home_prob"].notna().sum() > 0:
            model_sub = espn_sub[espn_sub["model_home_prob"].notna()]
            model_sub = model_sub.copy()
            model_sub["model_correct"] = (
                ((model_sub["model_home_prob"] >= 0.5) & (model_sub["home_win"] == 1))
                | ((model_sub["model_home_prob"] < 0.5) & (model_sub["home_win"] == 0))
            )
            print(f"XGBoost model accuracy: {model_sub['model_correct'].mean():.1%} ({int(model_sub['model_correct'].sum())}/{len(model_sub)})")

        if espn_sub["pm_pregame_home"].notna().sum() > 0:
            pm_sub = espn_sub[espn_sub["pm_pregame_home"].notna()]
            pm_sub = pm_sub.copy()
            pm_sub["pm_correct"] = (
                ((pm_sub["pm_pregame_home"] >= 0.5) & (pm_sub["home_win"] == 1))
                | ((pm_sub["pm_pregame_home"] < 0.5) & (pm_sub["home_win"] == 0))
            )
            print(f"Polymarket accuracy:   {pm_sub['pm_correct'].mean():.1%} ({int(pm_sub['pm_correct'].sum())}/{len(pm_sub)})")


if __name__ == "__main__":
    main()
