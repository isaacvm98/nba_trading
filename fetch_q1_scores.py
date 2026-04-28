"""
Fetch Q1 end scores for all NBA regular season games via nba_api.

Uses LeagueGameFinder (1 call/season) to get game IDs, then
BoxScoreSummaryV2 (1 call/game) for quarter-by-quarter scores.

Caches results incrementally so it can be resumed if interrupted.

Output: Data/backtest/q1_scores.csv
Cache:  Data/backtest/q1_scores_cache.json

Usage:
    python fetch_q1_scores.py                  # Fetch all (uses cache)
    python fetch_q1_scores.py --seasons 2024   # Fetch specific season(s)
    python fetch_q1_scores.py --test           # Quick test with 5 games
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, boxscoresummaryv3

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"
CACHE_FILE = BACKTEST_DIR / "q1_scores_cache.json"
OUTPUT_CSV = BACKTEST_DIR / "q1_scores.csv"

# nba_api team abbreviation -> our team names (matching OddsData/TeamData)
ABBR_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'NJN': 'Brooklyn Nets',  # Pre-2012 name
    'CHA': 'Charlotte Hornets', 'CHO': 'Charlotte Hornets', 'CHH': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans',
    'NOH': 'New Orleans Pelicans', 'NOK': 'New Orleans Pelicans',
    'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
    # 2025-26 season aliases
    'SEA': 'Seattle SuperSonics',
}

# Seasons to fetch (nba_api format: YYYY = start year)
ALL_SEASONS = [f'{y}-{str(y+1)[-2:]}' for y in range(2012, 2026)]

DELAY = 0.7  # seconds between API calls (0.6 is the empirical minimum)


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def fetch_season_games(season):
    """Fetch all regular season game IDs + basic info for a season."""
    print(f"  Fetching game list for {season}...")
    for attempt in range(5):
        try:
            gf = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00',
                season_type_nullable='Regular Season',
            )
            time.sleep(DELAY)
            df = gf.get_data_frames()[0]
            break
        except Exception as e:
            wait = (attempt + 1) * 10
            print(f"    Retry {attempt+1}/5 for {season} game list ({e.__class__.__name__}), waiting {wait}s...")
            time.sleep(wait)
    else:
        print(f"    FAILED to fetch game list for {season} after 5 attempts")
        return []

    # Home games have "vs." in MATCHUP, away have "@"
    home = df[df['MATCHUP'].str.contains('vs.')].copy()
    away = df[df['MATCHUP'].str.contains('@')].copy()

    games = []
    for _, h_row in home.iterrows():
        game_id = h_row['GAME_ID']
        # Find matching away row
        a_row = away[away['GAME_ID'] == game_id]
        if a_row.empty:
            continue
        a_row = a_row.iloc[0]

        home_abbr = h_row['TEAM_ABBREVIATION']
        away_abbr = a_row['TEAM_ABBREVIATION']

        games.append({
            'game_id': game_id,
            'game_date': h_row['GAME_DATE'],
            'home_team': ABBR_TO_NAME.get(home_abbr, h_row['TEAM_NAME']),
            'away_team': ABBR_TO_NAME.get(away_abbr, a_row['TEAM_NAME']),
            'home_abbr': home_abbr,
            'away_abbr': away_abbr,
        })

    print(f"    Found {len(games)} games")
    return games


def fetch_q1_score(game_id):
    """Fetch Q1 (and all quarter) scores for a single game."""
    for attempt in range(3):
        try:
            bs = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
            line_score = bs.get_data_frames()[5]

            if len(line_score) < 2:
                return None

            result = {}
            for _, row in line_score.iterrows():
                abbr = row['TEAM_ABBREVIATION']
                result[abbr] = {
                    'q1': row.get('PTS_QTR1'),
                    'q2': row.get('PTS_QTR2'),
                    'q3': row.get('PTS_QTR3'),
                    'q4': row.get('PTS_QTR4'),
                    'pts': row.get('PTS'),
                }
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep((attempt + 1) * 5)
            else:
                return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Fetch Q1 scores via nba_api")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons (e.g., 2024-25)")
    parser.add_argument("--test", action="store_true", help="Quick test with 5 games")
    args = parser.parse_args()

    cache = load_cache()
    print(f"Cache: {len(cache)} games already fetched")

    if args.seasons:
        seasons = args.seasons
    else:
        seasons = ALL_SEASONS

    total_fetched = 0
    total_errors = 0
    all_games = []

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"Season: {season}")
        print(f"{'='*60}")

        games = fetch_season_games(season)
        all_games.extend(games)

        # Filter to uncached games
        to_fetch = [g for g in games if g['game_id'] not in cache]
        print(f"  Need to fetch: {len(to_fetch)} (cached: {len(games) - len(to_fetch)})")

        if args.test:
            to_fetch = to_fetch[:5]
            print(f"  TEST MODE: limiting to {len(to_fetch)} games")

        for i, game in enumerate(to_fetch):
            scores = fetch_q1_score(game['game_id'])
            time.sleep(DELAY)

            if scores and 'error' not in scores:
                cache[game['game_id']] = {
                    'game_date': game['game_date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_abbr': game['home_abbr'],
                    'away_abbr': game['away_abbr'],
                    'scores': scores,
                }
                total_fetched += 1
            else:
                total_errors += 1
                if total_errors <= 10:
                    print(f"    Error: {game['game_id']} {game['home_team']} vs {game['away_team']}: {scores}")

            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(to_fetch)} | fetched: {total_fetched} | errors: {total_errors}")
                save_cache(cache)

        save_cache(cache)
        print(f"  Season complete. Cache now: {len(cache)} games")

        if args.test:
            break

    # Build output CSV
    print(f"\n{'='*60}")
    print(f"Building output CSV...")
    print(f"{'='*60}")

    rows = []
    for game_id, data in cache.items():
        if 'scores' not in data:
            continue

        home_abbr = data['home_abbr']
        away_abbr = data['away_abbr']
        scores = data['scores']

        home_scores = scores.get(home_abbr, {})
        away_scores = scores.get(away_abbr, {})

        home_q1 = home_scores.get('q1')
        away_q1 = away_scores.get('q1')

        if home_q1 is None or away_q1 is None:
            continue

        rows.append({
            'game_id': game_id,
            'game_date': data['game_date'],
            'home_team': data['home_team'],
            'away_team': data['away_team'],
            'home_q1_pts': int(home_q1),
            'away_q1_pts': int(away_q1),
            'home_q2_pts': int(home_scores.get('q2', 0)),
            'away_q2_pts': int(away_scores.get('q2', 0)),
            'home_q3_pts': int(home_scores.get('q3', 0)),
            'away_q3_pts': int(away_scores.get('q3', 0)),
            'home_q4_pts': int(home_scores.get('q4', 0)),
            'away_q4_pts': int(away_scores.get('q4', 0)),
            'home_pts': int(home_scores.get('pts', 0)),
            'away_pts': int(away_scores.get('pts', 0)),
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(['game_date', 'home_team'])
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(str(OUTPUT_CSV), index=False)

    print(f"\nSaved {len(out_df)} games to {OUTPUT_CSV}")
    print(f"Total fetched this run: {total_fetched}")
    print(f"Total errors: {total_errors}")
    print(f"Cache size: {len(cache)}")

    # Quick stats
    if len(out_df) > 0:
        out_df['underdog_leading_q1'] = (
            ((out_df['home_q1_pts'] > out_df['away_q1_pts']) & (out_df['home_pts'] < out_df['away_pts']))
            | ((out_df['away_q1_pts'] > out_df['home_q1_pts']) & (out_df['away_pts'] < out_df['home_pts']))
        )
        fav_comeback = out_df['underdog_leading_q1'].sum()
        print(f"\nQuick stat: Games where Q1 leader lost: {fav_comeback}/{len(out_df)} ({fav_comeback/len(out_df):.1%})")


if __name__ == "__main__":
    main()
