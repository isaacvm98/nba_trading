"""
Build a comprehensive NBA backtesting dataset combining:
- Polymarket prediction market events as primary game source (Gamma API)
- CLOB price history for pre-game price snapshots
- XGBoost model predictions (using historical TeamData)
- Kelly criterion sizing calculations

Output: Data/backtest/nba_backtest_dataset.csv
Cache:  Data/backtest/polymarket_events_cache.json
        Data/backtest/price_history_cache.json

Usage:
    python build_backtest_dataset.py              # Full run (uses cache)
    python build_backtest_dataset.py --no-cache   # Re-fetch all API data
    python build_backtest_dataset.py --no-prices  # Skip CLOB price history
"""

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
BACKTEST_DIR = DATA_DIR / "backtest"
CACHE_EVENTS = BACKTEST_DIR / "polymarket_events_cache.json"
CACHE_PRICES = BACKTEST_DIR / "price_history_cache.json"
CACHE_RAW_HISTORIES = BACKTEST_DIR / "price_history_raw_cache.json"
OUTPUT_CSV = BACKTEST_DIR / "nba_backtest_dataset.csv"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

ET = ZoneInfo("America/New_York")

# Polymarket API
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
NBA_SERIES_ID = "10345"
GAMES_TAG_ID = "100639"

# Regular season start
SEASON_START = datetime(2025, 10, 21, tzinfo=timezone.utc)
# Price history availability cutoff (~Jan 27 2026)
PRICE_HISTORY_CUTOFF = datetime(2026, 1, 27, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Team name normalization (from PolymarketOddsProvider)
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Lakers": "Los Angeles Lakers",
    "Clippers": "LA Clippers",
    "Warriors": "Golden State Warriors",
    "Kings": "Sacramento Kings",
    "Suns": "Phoenix Suns",
    "Nuggets": "Denver Nuggets",
    "Thunder": "Oklahoma City Thunder",
    "Blazers": "Portland Trail Blazers",
    "Trail Blazers": "Portland Trail Blazers",
    "Jazz": "Utah Jazz",
    "Timberwolves": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans",
    "Spurs": "San Antonio Spurs",
    "Rockets": "Houston Rockets",
    "Mavericks": "Dallas Mavericks",
    "Grizzlies": "Memphis Grizzlies",
    "Celtics": "Boston Celtics",
    "Nets": "Brooklyn Nets",
    "Knicks": "New York Knicks",
    "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers",
    "Raptors": "Toronto Raptors",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Cavs": "Cleveland Cavaliers",
    "Pistons": "Detroit Pistons",
    "Pacers": "Indiana Pacers",
    "Bucks": "Milwaukee Bucks",
    "Hawks": "Atlanta Hawks",
    "Hornets": "Charlotte Hornets",
    "Heat": "Miami Heat",
    "Magic": "Orlando Magic",
    "Wizards": "Washington Wizards",
}

# From src/Process-Data/Dictionaries.py
TEAM_INDEX = {
    "Atlanta Hawks": 0, "Boston Celtics": 1, "Brooklyn Nets": 2,
    "Charlotte Hornets": 3, "Chicago Bulls": 4, "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6, "Denver Nuggets": 7, "Detroit Pistons": 8,
    "Golden State Warriors": 9, "Houston Rockets": 10, "Indiana Pacers": 11,
    "Los Angeles Clippers": 12, "LA Clippers": 12,
    "Los Angeles Lakers": 13, "Memphis Grizzlies": 14, "Miami Heat": 15,
    "Milwaukee Bucks": 16, "Minnesota Timberwolves": 17,
    "New Orleans Pelicans": 18, "New York Knicks": 19,
    "Oklahoma City Thunder": 20, "Orlando Magic": 21,
    "Philadelphia 76ers": 22, "Phoenix Suns": 23,
    "Portland Trail Blazers": 24, "Sacramento Kings": 25,
    "San Antonio Spurs": 26, "Toronto Raptors": 27,
    "Utah Jazz": 28, "Washington Wizards": 29,
}


def normalize_team(name):
    """Normalize a Polymarket/odds team name to canonical form."""
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Phase 1: Fetch Polymarket Events
# ---------------------------------------------------------------------------
def fetch_polymarket_events(use_cache=True):
    """Paginate through all closed NBA game events from Polymarket."""
    if use_cache and CACHE_EVENTS.exists():
        print(f"Loading cached Polymarket events from {CACHE_EVENTS.name}")
        with open(CACHE_EVENTS) as f:
            events = json.load(f)
        print(f"  Loaded {len(events)} events from cache")
        return events

    print("Fetching closed NBA events from Polymarket Gamma API...")
    all_events = []
    offset = 0

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)

    while True:
        params = {
            "series_id": NBA_SERIES_ID,
            "tag_id": GAMES_TAG_ID,
            "closed": "true",
            "limit": 100,
            "offset": offset,
            "order": "startTime",
            "ascending": "true",
        }
        resp = session.get(f"{GAMMA_URL}/events", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_events.extend(batch)
        print(f"  Fetched {len(batch)} events (offset={offset}, total={len(all_events)})")
        offset += 100

        if len(batch) < 100:
            break

    # Parse and filter
    parsed = []
    for event in all_events:
        start_str = event.get("startTime", "")
        if not start_str:
            continue
        try:
            start_utc = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        # Skip preseason
        if start_utc < SEASON_START:
            continue

        # Find moneyline market
        ml_market = _find_moneyline_market(event)
        if not ml_market:
            continue

        outcomes = ml_market.get("outcomes", "")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = outcomes.replace("[", "").replace("]", "").replace('"', "").split(",")

        if len(outcomes) < 2:
            continue

        # Polymarket: outcomes[0] = away, outcomes[1] = home
        away_name = normalize_team(outcomes[0].strip())
        home_name = normalize_team(outcomes[1].strip())

        # Get token IDs
        token_ids = ml_market.get("clobTokenIds", "")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []

        # Resolution prices (0 or 1)
        outcome_prices = ml_market.get("outcomePrices", "")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                outcome_prices = []

        # Determine winner from resolution prices
        pm_winner = None
        if len(outcome_prices) >= 2:
            try:
                away_final = float(outcome_prices[0])
                home_final = float(outcome_prices[1])
                if home_final > 0.5:
                    pm_winner = "home"
                elif away_final > 0.5:
                    pm_winner = "away"
            except (ValueError, TypeError):
                pass

        # ET date for matching with OddsData
        start_et = start_utc.astimezone(ET)
        game_date_et = start_et.strftime("%Y-%m-%d")

        parsed.append({
            "home_team": home_name,
            "away_team": away_name,
            "start_utc": start_utc.isoformat(),
            "start_ts": int(start_utc.timestamp()),
            "game_date_et": game_date_et,
            "slug": event.get("slug", ""),
            "pm_winner": pm_winner,
            "home_token": token_ids[1] if len(token_ids) >= 2 else None,
            "away_token": token_ids[0] if len(token_ids) >= 2 else None,
            "market_id": ml_market.get("id", ""),
        })

    print(f"  Parsed {len(parsed)} regular-season events (filtered from {len(all_events)} total)")

    # Cache
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_EVENTS, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"  Cached to {CACHE_EVENTS.name}")

    return parsed


def _find_moneyline_market(event):
    """Find the moneyline (team vs team) market in an event."""
    for market in event.get("markets", []):
        question = market.get("question", "")
        # Skip non-ML markets
        if "O/U" in question or "Spread" in question or "1H" in question:
            continue
        if "Over" in question:
            continue
        if " vs" in question.lower():
            return market
    return None


# ---------------------------------------------------------------------------
# Phase 2: Fetch Price History
# ---------------------------------------------------------------------------
def _load_raw_cache():
    """Load the raw price history cache (token -> tick data) from build_price_history.py."""
    if CACHE_RAW_HISTORIES.exists():
        with open(CACHE_RAW_HISTORIES) as f:
            return json.load(f)
    return {}


def fetch_price_histories(events, use_cache=True):
    """Fetch CLOB price history for events after the cutoff date.

    Uses the raw tick-level cache from build_price_history.py as primary source,
    then falls back to the CLOB API for any missing tokens.
    """
    if use_cache and CACHE_PRICES.exists():
        print(f"Loading cached price histories from {CACHE_PRICES.name}")
        with open(CACHE_PRICES) as f:
            histories = json.load(f)
        print(f"  Loaded {len(histories)} price histories from cache")
        return histories

    # Load raw tick cache (built by build_price_history.py)
    raw_cache = _load_raw_cache()
    print(f"  Loaded {len(raw_cache)} entries from raw price history cache")

    # Filter to events with available price history
    eligible = [
        e for e in events
        if e.get("home_token") and e.get("start_ts", 0) >= PRICE_HISTORY_CUTOFF.timestamp()
    ]
    print(f"Building price history for {len(eligible)} events (post-Jan 27)...")

    histories = {}
    api_fetches = 0
    raw_hits = 0

    for i, event in enumerate(eligible):
        token = event["home_token"]
        game_key = f"{event['game_date_et']}_{event['home_team']}_{event['away_team']}"

        # Try raw cache first
        history = raw_cache.get(token, [])
        if history:
            raw_hits += 1
        else:
            # Fall back to CLOB API
            try:
                resp = requests.get(
                    f"{CLOB_URL}/prices-history",
                    params={"market": token, "interval": "max"},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and "history" in data:
                    history = data["history"]
                elif isinstance(data, list):
                    history = data
                else:
                    history = []

                api_fetches += 1
                if history:
                    raw_cache[token] = history  # Update raw cache
            except Exception as e:
                print(f"  Error fetching history for {game_key}: {e}")
                api_fetches += 1

            if api_fetches > 0 and api_fetches % 50 == 0:
                print(f"  API fetches: {api_fetches}...")
            time.sleep(0.3)  # Rate limiting

        if history:
            snapshots = _extract_snapshots(history, event["start_ts"])
            histories[game_key] = snapshots

    print(f"  Got price history for {len(histories)} games ({raw_hits} from cache, {api_fetches} from API)")

    # Save updated raw cache
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_RAW_HISTORIES, "w") as f:
        json.dump(raw_cache, f)

    with open(CACHE_PRICES, "w") as f:
        json.dump(histories, f, indent=2)
    print(f"  Cached to {CACHE_PRICES.name}")

    return histories


def _extract_snapshots(history, game_start_ts):
    """Extract key price snapshots from a timeseries.

    Returns dict with opening, pregame, max, min prices for home team.
    """
    if not history:
        return {}

    times = [h["t"] for h in history]
    prices = [float(h["p"]) for h in history]

    # Pre-game prices only (before game start)
    pre_game = [(t, p) for t, p in zip(times, prices) if t <= game_start_ts]
    if not pre_game:
        # All data is during/after game, use first point as opening
        pre_game = [(times[0], prices[0])]

    pre_times, pre_prices = zip(*pre_game)

    # Opening: first available price
    open_price = prices[0]

    # Price closest to game start (pre-game)
    pregame_price = pre_prices[-1]  # Last pre-game price

    # Pre-game range
    home_max = max(pre_prices)
    home_min = min(pre_prices)

    return {
        "pm_open_home": round(open_price, 4),
        "pm_open_away": round(1 - open_price, 4),
        "pm_pregame_home": round(pregame_price, 4),
        "pm_pregame_away": round(1 - pregame_price, 4),
        "pm_home_max": round(home_max, 4),
        "pm_home_min": round(home_min, 4),
        "history_points": len(history),
    }


# ---------------------------------------------------------------------------
# Phase 3: Load Local Data
# ---------------------------------------------------------------------------
def load_schedule():
    """Load nba-2025-UTC.csv for days-rest calculation."""
    csv_path = DATA_DIR / "nba-2025-UTC.csv"
    df = pd.read_csv(str(csv_path))
    # Parse dates: format is DD/MM/YYYY HH:MM
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    print(f"Loaded {len(df)} games from schedule CSV")
    return df


def prob_to_american(prob):
    """Convert probability to American odds."""
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))


# ---------------------------------------------------------------------------
# Phase 4: Build Games from Polymarket Events
# ---------------------------------------------------------------------------
def build_games_from_polymarket(pm_events, price_histories, schedule_df):
    """Build game rows using Polymarket events as the primary source."""
    print("Building game list from Polymarket events...")

    matched_prices = 0
    rows = []

    for event in pm_events:
        game_date = event["game_date_et"]
        home = event["home_team"]
        away = event["away_team"]

        # Result from Polymarket resolution
        pm_winner = event.get("pm_winner")
        home_win = None
        if pm_winner == "home":
            home_win = 1
        elif pm_winner == "away":
            home_win = 0

        # Look up price history
        price_key = f"{game_date}_{home}_{away}"
        price_data = price_histories.get(price_key, {})
        has_prices = bool(price_data)
        if has_prices:
            matched_prices += 1

        # Polymarket pregame prices as market probability
        pm_home_prob = price_data.get("pm_pregame_home")
        pm_away_prob = price_data.get("pm_pregame_away")

        # Convert to American odds for Kelly calculations
        pm_ml_home = prob_to_american(pm_home_prob) if pm_home_prob else None
        pm_ml_away = prob_to_american(pm_away_prob) if pm_away_prob else None

        # Days rest from schedule
        days_rest_home = calculate_days_rest(schedule_df, home, game_date)
        days_rest_away = calculate_days_rest(schedule_df, away, game_date)

        row = {
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "pm_slug": event.get("slug"),
            "has_price_history": has_prices,
            # Results (from Polymarket resolution)
            "home_win": home_win,
            # Polymarket prices (market odds)
            "pm_open_home": price_data.get("pm_open_home"),
            "pm_open_away": price_data.get("pm_open_away"),
            "pm_pregame_home": pm_home_prob,
            "pm_pregame_away": pm_away_prob,
            "pm_home_max": price_data.get("pm_home_max"),
            "pm_home_min": price_data.get("pm_home_min"),
            "pm_ml_home": pm_ml_home,
            "pm_ml_away": pm_ml_away,
            # Polymarket token IDs
            "pm_home_token": event.get("home_token"),
            "pm_away_token": event.get("away_token"),
            # Rest days
            "days_rest_home": days_rest_home,
            "days_rest_away": days_rest_away,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Built {len(df)} games from Polymarket events")
    print(f"  Games with price history: {matched_prices}")
    print(f"  Games with results: {df['home_win'].notna().sum()}")
    return df


# ---------------------------------------------------------------------------
# Phase 5: Model Predictions
# ---------------------------------------------------------------------------
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")


def _select_model_path(kind):
    """Select the best model based on accuracy."""
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def load_model():
    """Load XGBoost ML model and optional calibrator."""
    ml_path = _select_model_path("ML")
    model = xgb.Booster()
    model.load_model(str(ml_path))
    print(f"Loaded model: {ml_path.name}")

    calibrator = None
    cal_path = ml_path.with_name(f"{ml_path.stem}_calibration.pkl")
    if cal_path.exists():
        try:
            import joblib
            calibrator = joblib.load(cal_path)
            print(f"Loaded calibrator: {cal_path.name}")
        except Exception:
            pass

    return model, calibrator


def calculate_days_rest(schedule_df, team_name, game_date):
    """Calculate days of rest for a team before a given game date."""
    game_dt = pd.Timestamp(game_date)

    team_games = schedule_df[
        (schedule_df["Home Team"] == team_name) | (schedule_df["Away Team"] == team_name)
    ]
    prev_games = team_games[team_games["Date"] < game_dt].sort_values("Date", ascending=False)

    if len(prev_games) > 0:
        last_game = prev_games.iloc[0]["Date"]
        days_off = (game_dt - last_game).days
        return max(1, days_off)
    return 7  # Default for season start


def generate_predictions(df, schedule_df):
    """Generate XGBoost predictions for all games using historical TeamData."""
    print("Generating model predictions...")

    model, calibrator = load_model()
    teams_con = sqlite3.connect(str(DATA_DIR / "TeamData.sqlite"))
    teams_cur = teams_con.cursor()

    # Get list of available TeamData tables
    available_dates = set()
    tables = teams_cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for (t,) in tables:
        available_dates.add(t)

    predictions = []
    success_count = 0
    skip_count = 0

    for idx, row in df.iterrows():
        game_date = row["game_date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Check if TeamData exists for this date
        if game_date not in available_dates:
            # Try previous days (up to 30 days back to handle gaps)
            dt = datetime.strptime(game_date, "%Y-%m-%d")
            found = False
            for offset in range(1, 31):
                alt_date = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
                if alt_date in available_dates:
                    game_date = alt_date
                    found = True
                    break
            if not found:
                predictions.append({"model_home_prob": None, "model_away_prob": None})
                skip_count += 1
                continue

        # Check team indices exist
        if home_team not in TEAM_INDEX or away_team not in TEAM_INDEX:
            predictions.append({"model_home_prob": None, "model_away_prob": None})
            skip_count += 1
            continue

        try:
            # Load team stats for that date
            team_df = pd.read_sql_query(f'SELECT * FROM "{game_date}"', teams_con)

            home_idx = TEAM_INDEX[home_team]
            away_idx = TEAM_INDEX[away_team]

            if home_idx >= len(team_df) or away_idx >= len(team_df):
                predictions.append({"model_home_prob": None, "model_away_prob": None})
                skip_count += 1
                continue

            home_series = team_df.iloc[home_idx]
            away_series = team_df.iloc[away_idx]

            # Build feature vector
            stats = pd.concat([home_series, away_series])
            stats = stats.drop(labels=["TEAM_ID", "TEAM_NAME", "Date"], errors="ignore")

            # Calculate days rest
            days_rest_home = calculate_days_rest(schedule_df, home_team, row["game_date"])
            days_rest_away = calculate_days_rest(schedule_df, away_team, row["game_date"])
            stats["Days-Rest-Home"] = days_rest_home
            stats["Days-Rest-Away"] = days_rest_away

            feature_data = stats.values.astype(float).reshape(1, -1)

            # Predict
            if calibrator is not None:
                probs = calibrator.predict_proba(feature_data)[0]
            else:
                probs = model.predict(xgb.DMatrix(feature_data))[0]

            if isinstance(probs, np.ndarray) and len(probs) >= 2:
                home_prob = float(probs[1])
                away_prob = float(probs[0])
            else:
                home_prob = float(probs) if not isinstance(probs, np.ndarray) else float(probs[0])
                away_prob = 1 - home_prob

            predictions.append({
                "model_home_prob": round(home_prob, 4),
                "model_away_prob": round(away_prob, 4),
            })
            success_count += 1

        except Exception as e:
            predictions.append({"model_home_prob": None, "model_away_prob": None})
            skip_count += 1
            if skip_count <= 5:
                print(f"  Error predicting {home_team} vs {away_team} ({game_date}): {e}")

        if (idx + 1) % 100 == 0:
            print(f"  Predicted {idx + 1}/{len(df)} games...")

    teams_con.close()
    print(f"  Generated {success_count} predictions, skipped {skip_count}")

    pred_df = pd.DataFrame(predictions)
    return pred_df


# ---------------------------------------------------------------------------
# Phase 6: Edge & Kelly Calculations
# ---------------------------------------------------------------------------
def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds >= 100:
        return american_odds / 100
    else:
        return 100 / abs(american_odds)


def calc_kelly(american_odds, model_prob):
    """Calculate raw Kelly criterion percentage."""
    decimal_odds = american_to_decimal(american_odds)
    fraction = (100 * (decimal_odds * model_prob - (1 - model_prob))) / decimal_odds
    return round(fraction, 2) if fraction > 0 else 0


def calc_tiered_kelly(american_odds, model_prob, market_prob):
    """Calculate tiered Kelly sizing based on edge magnitude."""
    edge = model_prob - market_prob
    if edge < 0.05:
        return 0

    base_kelly = calc_kelly(american_odds, model_prob)
    if base_kelly <= 0:
        return 0

    if edge < 0.07:
        tier = 0.25
    elif edge < 0.10:
        tier = 0.35
    else:
        tier = 0.50

    bet_size = base_kelly * 0.25 * tier  # quarter-Kelly * tier
    return round(min(bet_size, 10.0), 2)


def compute_edges_and_kelly(df):
    """Compute edge and Kelly sizing for each game (vs Polymarket prices)."""
    print("Computing edges and Kelly sizing...")

    # Edge vs Polymarket pregame price
    df["edge_home_pm"] = df.apply(
        lambda r: round(r["model_home_prob"] - r["pm_pregame_home"], 4)
        if pd.notna(r.get("model_home_prob")) and pd.notna(r.get("pm_pregame_home"))
        else None,
        axis=1,
    )
    df["edge_away_pm"] = df.apply(
        lambda r: round(r["model_away_prob"] - r["pm_pregame_away"], 4)
        if pd.notna(r.get("model_away_prob")) and pd.notna(r.get("pm_pregame_away"))
        else None,
        axis=1,
    )

    # Kelly vs Polymarket
    kelly_home_pm = []
    kelly_away_pm = []
    tiered_home_pm = []
    tiered_away_pm = []

    for _, r in df.iterrows():
        if pd.notna(r.get("model_home_prob")) and pd.notna(r.get("pm_ml_home")):
            kh = calc_kelly(r["pm_ml_home"], r["model_home_prob"])
            ka = calc_kelly(r["pm_ml_away"], r["model_away_prob"])
            th = calc_tiered_kelly(r["pm_ml_home"], r["model_home_prob"], r["pm_pregame_home"])
            ta = calc_tiered_kelly(r["pm_ml_away"], r["model_away_prob"], r["pm_pregame_away"])
        else:
            kh = ka = th = ta = None

        kelly_home_pm.append(kh)
        kelly_away_pm.append(ka)
        tiered_home_pm.append(th)
        tiered_away_pm.append(ta)

    df["kelly_home_pm"] = kelly_home_pm
    df["kelly_away_pm"] = kelly_away_pm
    df["tiered_kelly_home_pm"] = tiered_home_pm
    df["tiered_kelly_away_pm"] = tiered_away_pm

    # Determine recommended bet side (using Polymarket odds)
    bet_sides = []
    bet_kellys = []

    for _, r in df.iterrows():
        kh = r.get("kelly_home_pm") or 0
        ka = r.get("kelly_away_pm") or 0

        if kh > 0 and kh >= ka:
            bet_sides.append("home")
            bet_kellys.append(kh)
        elif ka > 0:
            bet_sides.append("away")
            bet_kellys.append(ka)
        else:
            bet_sides.append("none")
            bet_kellys.append(0)

    df["bet_side"] = bet_sides
    df["bet_kelly"] = bet_kellys

    # Model correctness
    df["model_predicted_winner"] = df.apply(
        lambda r: "home" if pd.notna(r.get("model_home_prob")) and r["model_home_prob"] > 0.5 else "away",
        axis=1,
    )
    df["model_correct"] = df.apply(
        lambda r: (r["model_predicted_winner"] == "home" and r.get("home_win") == 1)
        or (r["model_predicted_winner"] == "away" and r.get("home_win") == 0)
        if pd.notna(r.get("home_win")) and pd.notna(r.get("model_home_prob"))
        else None,
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Phase 7: Assemble & Export
# ---------------------------------------------------------------------------
def print_summary(df):
    """Print summary statistics of the dataset."""
    print("\n" + "=" * 60)
    print("BACKTESTING DATASET SUMMARY")
    print("=" * 60)
    print(f"Total games: {len(df)}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    has_result = df["home_win"].notna().sum()
    print(f"Games with results: {has_result}")

    has_prices = df["has_price_history"].sum()
    print(f"Games with price history: {has_prices}")

    has_model = df["model_home_prob"].notna().sum()
    print(f"Games with model predictions: {has_model}")

    # Model accuracy
    correct = df["model_correct"]
    if correct.notna().sum() > 0:
        accuracy = correct.sum() / correct.notna().sum()
        print(f"\nModel accuracy: {accuracy:.1%} ({int(correct.sum())}/{correct.notna().sum()})")

    # Betting analysis
    bets = df[df["bet_side"] != "none"]
    print(f"\nGames where Kelly recommends a bet: {len(bets)} ({len(bets)/len(df):.0%})")

    if len(bets) > 0:
        # How many bets are underdogs vs favorites
        underdog_bets = bets[
            ((bets["bet_side"] == "home") & (bets["pm_pregame_home"] < 0.5))
            | ((bets["bet_side"] == "away") & (bets["pm_pregame_away"] < 0.5))
        ]
        valid_underdogs = underdog_bets.dropna(subset=["pm_pregame_home"])
        print(f"  Underdog bets: {len(valid_underdogs)} ({len(valid_underdogs)/len(bets):.0%})")

        # Win rate on bets
        bet_wins = bets.apply(
            lambda r: (r["bet_side"] == "home" and r.get("home_win") == 1)
            or (r["bet_side"] == "away" and r.get("home_win") == 0),
            axis=1,
        )
        if bet_wins.sum() > 0:
            print(f"  Bet win rate: {bet_wins.sum()/len(bets):.1%} ({int(bet_wins.sum())}/{len(bets)})")

        # Average Kelly %
        print(f"  Average Kelly %: {bets['bet_kelly'].mean():.2f}%")
        print(f"  Median Kelly %: {bets['bet_kelly'].median():.2f}%")

    # Edge distribution
    if df["edge_home_pm"].notna().sum() > 0:
        all_edges = pd.concat([
            df["edge_home_pm"].dropna(),
            df["edge_away_pm"].dropna(),
        ])
        print(f"\nEdge distribution (vs Polymarket):")
        print(f"  Mean: {all_edges.mean():.4f}")
        print(f"  Positive edges: {(all_edges > 0).sum()}")
        print(f"  Edges > 5%: {(all_edges > 0.05).sum()}")
        print(f"  Edges > 10%: {(all_edges > 0.10).sum()}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build NBA backtesting dataset")
    parser.add_argument("--no-cache", action="store_true", help="Re-fetch all API data")
    parser.add_argument("--no-prices", action="store_true", help="Skip CLOB price history")
    args = parser.parse_args()

    use_cache = not args.no_cache

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Fetch Polymarket events (primary game source)
    pm_events = fetch_polymarket_events(use_cache=use_cache)

    # Phase 2: Fetch price histories
    if args.no_prices:
        price_histories = {}
        print("Skipping price history (--no-prices)")
    else:
        price_histories = fetch_price_histories(pm_events, use_cache=use_cache)

    # Phase 3: Load schedule for days-rest
    schedule_df = load_schedule()

    # Phase 4: Build game list from Polymarket events
    df = build_games_from_polymarket(pm_events, price_histories, schedule_df)

    # Phase 5: Model predictions
    pred_df = generate_predictions(df, schedule_df)
    df["model_home_prob"] = pred_df["model_home_prob"]
    df["model_away_prob"] = pred_df["model_away_prob"]

    # Phase 6: Edges & Kelly
    df = compute_edges_and_kelly(df)

    # Phase 7: Export
    df.to_csv(str(OUTPUT_CSV), index=False)
    print(f"\nDataset saved to {OUTPUT_CSV}")

    print_summary(df)


if __name__ == "__main__":
    main()
