"""
Build expanded historical backtest dataset for flip strategy validation.

Combines:
- OddsData.sqlite: game results + sportsbook ML odds (2012-2026)
- TeamData.sqlite: team stats for model predictions
- XGBoost model: pre-game probability predictions
- Q1 scores: from fetch_q1_scores.py (nba_api)

Then runs the flip simulation on the full dataset.

Output: Data/backtest/historical_backtest.csv
        Console: flip strategy results across thousands of games

Usage:
    python build_historical_backtest.py
"""

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
BACKTEST_DIR = DATA_DIR / "backtest"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
OUTPUT_CSV = BACKTEST_DIR / "historical_backtest.csv"

STARTING_BANKROLL = 1000.0
MIN_EDGE = 0.07
FAV_MIN_CONF = 0.60
DOG_MIN_ENTRY = 0.30
MIN_ENTRY_PRICE = 0.05

# Q1 price model coefficients (calibrated on 313 Polymarket games, R²=0.93)
Q1_INTERCEPT = 0.0209
Q1_PREGAME_COEF = 0.9455
Q1_DIFF_COEF = 0.0096

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

ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")


# ── Load data ──────────────────────────────────────────────────────────────

def load_odds_data():
    """Load all games from OddsData.sqlite with ML odds and results."""
    conn = sqlite3.connect(str(DATA_DIR / "OddsData.sqlite"))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    # Use '_new' tables for historical seasons (have Win_Margin), clean tables for recent
    target_tables = sorted([t for t in tables if '_new' in t])
    clean_tables = sorted([t for t in tables if not t.startswith('odds_') and '_new' not in t])
    # Avoid duplicates: clean tables override _new for same season
    clean_seasons = {t for t in clean_tables}
    new_seasons = {t.replace('odds_', '').replace('_new', '') for t in target_tables}
    # Use clean if available, else _new
    use_tables = []
    for t in clean_tables:
        use_tables.append((t, t))
    for t in target_tables:
        season = t.replace('odds_', '').replace('_new', '')
        if season not in clean_seasons:
            use_tables.append((t, season))

    all_games = []
    for table_name, season in use_tables:
        try:
            df = pd.read_sql(f'SELECT * FROM [{table_name}]', conn)
            # Standardize columns
            for col in ['index', 'Unnamed: 0']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            df['season'] = season
            all_games.append(df)
        except Exception as e:
            print(f"  Error loading {table_name}: {e}")

    conn.close()
    combined = pd.concat(all_games, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])

    # Filter to 2012+ (need TeamData for predictions)
    combined = combined[combined['Date'] >= '2012-10-01'].copy()
    combined = combined.dropna(subset=['ML_Home', 'ML_Away', 'Win_Margin'])

    # Normalize historical team names to current names (matching TeamData)
    name_remap = {
        "Charlotte Bobcats": "Charlotte Hornets",
        "New Jersey Nets": "Brooklyn Nets",
        "Seattle SuperSonics": "Oklahoma City Thunder",
        "New Orleans Hornets": "New Orleans Pelicans",
        "Los Angeles Clippers": "LA Clippers",
    }
    combined['Home'] = combined['Home'].replace(name_remap)
    combined['Away'] = combined['Away'].replace(name_remap)

    combined = combined.sort_values('Date').reset_index(drop=True)

    print(f"Loaded {len(combined)} games from OddsData (2012-2026)")
    return combined


def load_q1_scores():
    """Load Q1 scores from fetch_q1_scores.py output."""
    q1_path = BACKTEST_DIR / "q1_scores.csv"
    if not q1_path.exists():
        print("WARNING: q1_scores.csv not found. Run fetch_q1_scores.py first.")
        return pd.DataFrame()

    df = pd.read_csv(str(q1_path))
    df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"Loaded {len(df)} Q1 scores")
    return df


def load_model():
    """Load XGBoost ML model + calibrator."""
    candidates = list(MODEL_DIR.glob("*ML*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost ML model in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        return (path.stat().st_mtime, float(match.group(1)) if match else 0.0)

    ml_path = max(candidates, key=score)
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


def american_to_prob(ml):
    """Convert American moneyline to implied probability."""
    try:
        ml = float(ml)
    except (ValueError, TypeError):
        return np.nan
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    elif ml > 0:
        return 100 / (ml + 100)
    else:
        return 0.5


def estimate_q1_price(pregame_price, q1_score_diff):
    """Estimate Q1 market price using calibrated linear model."""
    price = Q1_INTERCEPT + Q1_PREGAME_COEF * pregame_price + Q1_DIFF_COEF * q1_score_diff
    return np.clip(price, 0.01, 0.99)


# ── Generate predictions ───────────────────────────────────────────────────

def generate_predictions(odds_df):
    """Run XGBoost on all historical games using TeamData snapshots."""
    print("\nGenerating model predictions...")
    model, calibrator = load_model()

    teams_con = sqlite3.connect(str(DATA_DIR / "TeamData.sqlite"))
    available_dates = set(
        r[0] for r in teams_con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )

    results = []
    success = skip = 0
    td_cache = {}  # Cache TeamData DataFrames to avoid re-reading

    for idx, row in odds_df.iterrows():
        game_date = row['Date'].strftime('%Y-%m-%d')
        home = row['Home']
        away = row['Away']

        # Normalize historical team names for TeamData lookup
        name_remap = {
            "Charlotte Bobcats": "Charlotte Hornets",
            "New Jersey Nets": "Brooklyn Nets",
            "Seattle SuperSonics": "Oklahoma City Thunder",
            "New Orleans Hornets": "New Orleans Pelicans",
            "Los Angeles Clippers": "LA Clippers",
        }
        home_td = name_remap.get(home, home)
        away_td = name_remap.get(away, away)

        # Find a TeamData date that has both teams (search backwards up to 60 days)
        dt = row['Date']
        home_series = away_series = None
        td_date = None
        for offset in range(0, 60):
            candidate = (dt - timedelta(days=offset)).strftime('%Y-%m-%d')
            if candidate not in available_dates:
                continue
            # Check if cached
            if candidate not in td_cache:
                td_cache[candidate] = pd.read_sql_query(f'SELECT * FROM "{candidate}"', teams_con)
            tdf = td_cache[candidate]
            h = tdf[tdf['TEAM_NAME'] == home_td]
            a = tdf[tdf['TEAM_NAME'] == away_td]
            if not h.empty and not a.empty:
                home_series = h.iloc[0]
                away_series = a.iloc[0]
                td_date = candidate
                break

        if home_series is None or away_series is None:
            results.append({'model_home_prob': None, 'model_away_prob': None})
            skip += 1
            continue

        try:

            stats = pd.concat([home_series, away_series])
            stats = stats.drop(labels=["index", "TEAM_ID", "TEAM_NAME", "Date"], errors="ignore")

            # Days rest from OddsData
            days_home = row.get('Days_Rest_Home', 3)
            days_away = row.get('Days_Rest_Away', 3)
            stats["Days-Rest-Home"] = days_home if pd.notna(days_home) else 3
            stats["Days-Rest-Away"] = days_away if pd.notna(days_away) else 3

            feature_data = stats.values.astype(float).reshape(1, -1)

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

            results.append({
                'model_home_prob': round(home_prob, 4),
                'model_away_prob': round(away_prob, 4),
            })
            success += 1

        except Exception as e:
            results.append({'model_home_prob': None, 'model_away_prob': None})
            skip += 1
            if skip <= 5:
                print(f"  Error: {home} vs {away} ({game_date}): {e}")

        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(odds_df)} games predicted...")

    teams_con.close()
    print(f"  Predictions: {success} success, {skip} skipped")
    return pd.DataFrame(results)


# ── Build dataset ──────────────────────────────────────────────────────────

def build_dataset():
    """Build the full historical backtest dataset."""
    odds = load_odds_data()
    q1 = load_q1_scores()

    # Generate model predictions
    preds = generate_predictions(odds)
    odds['model_home_prob'] = preds['model_home_prob']
    odds['model_away_prob'] = preds['model_away_prob']

    # Market implied probabilities
    odds['market_home'] = odds['ML_Home'].apply(american_to_prob)
    odds['market_away'] = odds['ML_Away'].apply(american_to_prob)

    # Edge
    odds['edge_home'] = odds['model_home_prob'] - odds['market_home']
    odds['edge_away'] = odds['model_away_prob'] - odds['market_away']

    # Game outcome
    odds['home_won'] = (odds['Win_Margin'] > 0).astype(int)

    # Merge Q1 scores
    odds['game_date_str'] = odds['Date'].dt.strftime('%Y-%m-%d')
    if not q1.empty:
        q1['game_date_str'] = q1['game_date'].dt.strftime('%Y-%m-%d')
        odds = odds.merge(
            q1[['game_date_str', 'home_team', 'away_team',
                'home_q1_pts', 'away_q1_pts']],
            left_on=['game_date_str', 'Home', 'Away'],
            right_on=['game_date_str', 'home_team', 'away_team'],
            how='left',
        )
        odds = odds.drop(columns=['home_team', 'away_team'], errors='ignore')
    else:
        odds['home_q1_pts'] = None
        odds['away_q1_pts'] = None

    odds['q1_diff_home'] = odds['home_q1_pts'] - odds['away_q1_pts']

    # Determine bet side (highest positive edge that passes filter)
    def get_bet_info(row):
        mhp = row.get('model_home_prob')
        map_ = row.get('model_away_prob')
        if pd.isna(mhp) or pd.isna(map_):
            return 'none', 0, 0, 0

        eh = row['edge_home']
        ea = row['edge_away']

        best_side = 'none'
        best_edge = 0
        model_prob = 0
        entry_price = 0

        if eh >= MIN_EDGE:
            best_side = 'home'
            best_edge = eh
            model_prob = mhp
            entry_price = row['market_home']
        if ea >= MIN_EDGE and ea > eh:
            best_side = 'away'
            best_edge = ea
            model_prob = map_
            entry_price = row['market_away']

        # Apply leg filters
        if best_side != 'none':
            is_fav = model_prob >= FAV_MIN_CONF
            if not is_fav and entry_price < DOG_MIN_ENTRY:
                return 'none', 0, 0, 0
            if entry_price < MIN_ENTRY_PRICE:
                return 'none', 0, 0, 0

        return best_side, best_edge, model_prob, entry_price

    bet_info = odds.apply(get_bet_info, axis=1, result_type='expand')
    odds['bet_side'] = bet_info[0]
    odds['bet_edge'] = bet_info[1]
    odds['model_prob'] = bet_info[2]
    odds['entry_price'] = bet_info[3]

    # Bet outcome
    odds['bet_won'] = odds.apply(
        lambda r: (r['bet_side'] == 'home' and r['home_won'] == 1) or
                  (r['bet_side'] == 'away' and r['home_won'] == 0)
        if r['bet_side'] != 'none' else None, axis=1
    )

    # Opposite side won (for flip strategy)
    odds['opp_won'] = odds.apply(
        lambda r: not r['bet_won'] if pd.notna(r['bet_won']) else None, axis=1
    )

    # Q1 score diff from bet side's perspective
    odds['q1_diff_bet'] = odds.apply(
        lambda r: r['q1_diff_home'] * (1 if r['bet_side'] == 'home' else -1)
        if r['bet_side'] != 'none' and pd.notna(r.get('q1_diff_home')) else None, axis=1
    )

    # Estimated Q1 prices
    odds['est_q1_bet_price'] = odds.apply(
        lambda r: estimate_q1_price(r['entry_price'], r['q1_diff_bet'])
        if pd.notna(r.get('q1_diff_bet')) and r['bet_side'] != 'none' else None, axis=1
    )
    odds['est_q1_opp_price'] = odds.apply(
        lambda r: 1.0 - estimate_q1_price(r['entry_price'], r['q1_diff_bet'])
        if pd.notna(r.get('q1_diff_bet')) and r['bet_side'] != 'none' else None, axis=1
    )

    # Is favorite / underdog
    odds['is_favorite'] = odds['model_prob'] >= FAV_MIN_CONF

    # Save
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    save_cols = ['Date', 'Home', 'Away', 'season', 'ML_Home', 'ML_Away', 'Win_Margin',
                 'market_home', 'market_away', 'model_home_prob', 'model_away_prob',
                 'edge_home', 'edge_away', 'home_won',
                 'bet_side', 'bet_edge', 'model_prob', 'entry_price',
                 'bet_won', 'opp_won', 'is_favorite',
                 'home_q1_pts', 'away_q1_pts', 'q1_diff_home', 'q1_diff_bet',
                 'est_q1_bet_price', 'est_q1_opp_price']
    out = odds[[c for c in save_cols if c in odds.columns]].copy()
    out.to_csv(str(OUTPUT_CSV), index=False)
    print(f"\nSaved {len(out)} games to {OUTPUT_CSV}")

    return odds


# ── Flip simulation ────────────────────────────────────────────────────────

def simulate(df, name, strategy_fn):
    """Run bankroll simulation with flat 2% sizing."""
    bankroll = STARTING_BANKROLL
    history = []

    bettable = df[df['bet_side'] != 'none'].copy()

    for _, row in bettable.iterrows():
        trades = strategy_fn(row, bankroll)
        if not trades:
            continue
        for pnl, desc, bet_amt in trades:
            bankroll += pnl
            history.append({
                'date': row['Date'],
                'game': f"{row['Away']} @ {row['Home']}",
                'desc': desc,
                'bet_amount': round(bet_amt, 2),
                'pnl': round(pnl, 2),
                'bankroll': round(bankroll, 2),
            })

    if not history:
        return None

    total = len(history)
    wins = sum(1 for h in history if h['pnl'] >= 0)
    losses = total - wins

    running_peak = STARTING_BANKROLL
    max_dd = 0
    for h in history:
        running_peak = max(running_peak, h['bankroll'])
        dd = (running_peak - h['bankroll']) / running_peak if running_peak > 0 else 0
        max_dd = max(max_dd, dd)

    daily = {}
    for h in history:
        d = h['date']
        if d not in daily:
            daily[d] = {'start': h['bankroll'] - h['pnl'], 'pnl': 0}
        daily[d]['pnl'] += h['pnl']
    dr = np.array([d['pnl'] / d['start'] for d in daily.values() if d['start'] > 0])
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0

    return {
        'name': name,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / total,
        'final': round(bankroll, 2),
        'ret': (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL,
        'max_dd': max_dd,
        'sharpe': round(sharpe, 2),
        'history': history,
    }


# ── Strategy functions ─────────────────────────────────────────────────────

def strategy_current(row, bankroll):
    """Current: sell dog at Q1 if leading, else hold. Favs hold."""
    bet_amount = bankroll * 0.02
    if bet_amount <= 0:
        return []

    is_fav = row.get('is_favorite', False)

    # FAV: hold to resolution
    if is_fav:
        if row['bet_won']:
            pnl = bet_amount * (1.0 / row['entry_price'] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, 'FAV hold', bet_amount)]

    # DOG: Q1 exit if leading
    if (pd.notna(row.get('q1_diff_bet')) and row['q1_diff_bet'] > 0
            and pd.notna(row.get('est_q1_bet_price'))):
        q1_price = row['est_q1_bet_price']
        pnl = bet_amount * (q1_price / row['entry_price'] - 1)
        return [(pnl, 'DOG q1_exit', bet_amount)]

    # DOG: no Q1 exit, hold
    if row['bet_won']:
        pnl = bet_amount * (1.0 / row['entry_price'] - 1)
    else:
        pnl = -bet_amount
    return [(pnl, 'DOG hold', bet_amount)]


def strategy_flip(row, bankroll):
    """Flip: sell dog at Q1 + buy fav at Q1 discounted price."""
    bet_amount = bankroll * 0.02
    if bet_amount <= 0:
        return []

    is_fav = row.get('is_favorite', False)

    if is_fav:
        if row['bet_won']:
            pnl = bet_amount * (1.0 / row['entry_price'] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, 'FAV hold', bet_amount)]

    # DOG: Q1 exit + flip
    if (pd.notna(row.get('q1_diff_bet')) and row['q1_diff_bet'] > 0
            and pd.notna(row.get('est_q1_bet_price'))
            and pd.notna(row.get('est_q1_opp_price'))
            and row['est_q1_opp_price'] > 0):
        q1_price = row['est_q1_bet_price']
        dog_pnl = bet_amount * (q1_price / row['entry_price'] - 1)
        bankroll_after = bankroll + dog_pnl

        fav_q1_price = row['est_q1_opp_price']
        fav_amount = bankroll_after * 0.02
        if fav_amount <= 0 or fav_q1_price <= 0 or fav_q1_price >= 1:
            return [(dog_pnl, 'DOG q1_exit', bet_amount)]

        if row['opp_won']:
            fav_pnl = fav_amount * (1.0 / fav_q1_price - 1)
        else:
            fav_pnl = -fav_amount

        return [
            (dog_pnl, 'DOG q1_exit', bet_amount),
            (fav_pnl, 'FLIP fav_hold', fav_amount),
        ]

    # DOG: no Q1 exit
    if row['bet_won']:
        pnl = bet_amount * (1.0 / row['entry_price'] - 1)
    else:
        pnl = -bet_amount
    return [(pnl, 'DOG hold', bet_amount)]


def strategy_fav_q1_only(row, bankroll):
    """Only buy fav at Q1 discount when dog was leading. Skip dog entirely."""
    bet_amount = bankroll * 0.02
    if bet_amount <= 0:
        return []

    is_fav = row.get('is_favorite', False)

    if is_fav:
        if row['bet_won']:
            pnl = bet_amount * (1.0 / row['entry_price'] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, 'FAV hold', bet_amount)]

    # DOG spot: only enter if Q1 exit conditions met, buy the FAVORITE
    if (pd.notna(row.get('q1_diff_bet')) and row['q1_diff_bet'] > 0
            and pd.notna(row.get('est_q1_opp_price'))
            and row['est_q1_opp_price'] > 0
            and row['est_q1_opp_price'] < 1):
        fav_q1_price = row['est_q1_opp_price']
        if row['opp_won']:
            pnl = bet_amount * (1.0 / fav_q1_price - 1)
        else:
            pnl = -bet_amount
        return [(pnl, 'FAV q1_buy', bet_amount)]

    return []


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    df = build_dataset()

    # Stats
    bettable = df[df['bet_side'] != 'none']
    has_q1 = bettable[bettable['q1_diff_bet'].notna()]
    q1_exits = has_q1[has_q1['q1_diff_bet'] > 0]

    print(f"\n{'='*90}")
    print(f"DATASET SUMMARY")
    print(f"{'='*90}")
    print(f"Total games: {len(df)}")
    print(f"Games with model predictions: {df['model_home_prob'].notna().sum()}")
    print(f"Bettable (edge >= 7%): {len(bettable)}")
    print(f"With Q1 scores: {len(has_q1)}")
    print(f"Q1 underdog leading (flip candidates): {len(q1_exits)}")

    if len(has_q1) == 0:
        print("\nNo Q1 data available. Run fetch_q1_scores.py first.")
        return

    # Run simulations
    strategies = [
        ("A) Current (sell dog Q1)", strategy_current),
        ("B) Flip (sell dog + buy fav Q1)", strategy_flip),
        ("C) Fav-only at Q1", strategy_fav_q1_only),
    ]

    results = []
    for name, fn in strategies:
        r = simulate(has_q1, name, fn)
        if r:
            results.append(r)

    # Print comparison
    print(f"\n{'='*105}")
    print(f"FLIP STRATEGY — HISTORICAL BACKTEST (Flat 2%, {len(has_q1)} games with Q1 data)")
    print(f"{'='*105}")

    hdr = f"{'Strategy':<38s} {'Trades':>6s} {'W/L':>8s} {'WR':>6s} {'Final $':>10s} {'Return':>9s} {'MaxDD':>7s} {'Sharpe':>7s}"
    print(hdr)
    print("-" * 105)
    for r in results:
        wl = f"{r['wins']}/{r['losses']}"
        print(f"{r['name']:<38s} {r['trades']:>6d} {wl:>8s} {r['win_rate']:>5.1%} "
              f"${r['final']:>9.2f} {r['ret']:>+8.1%} "
              f"{r['max_dd']:>6.1%} {r['sharpe']:>7.2f}")

    # Breakdown by trade type
    print(f"\n{'='*105}")
    print("BREAKDOWN BY TRADE TYPE")
    print(f"{'='*105}")
    for r in results:
        print(f"\n--- {r['name']} ---")
        types = {}
        for h in r['history']:
            d = h['desc']
            if d not in types:
                types[d] = {'n': 0, 'wins': 0, 'pnl': 0}
            types[d]['n'] += 1
            types[d]['wins'] += 1 if h['pnl'] >= 0 else 0
            types[d]['pnl'] += h['pnl']
        for desc, t in sorted(types.items()):
            wr = t['wins'] / t['n'] if t['n'] > 0 else 0
            print(f"  {desc:<25s} {t['n']:>5d} trades, {t['wins']}W/{t['n']-t['wins']}L "
                  f"({wr:>5.1%}), P&L: ${t['pnl']:>+10.2f}")

    # Season breakdown for flip
    if len(results) > 1:
        print(f"\n{'='*105}")
        print("FLIP STRATEGY BY SEASON")
        print(f"{'='*105}")
        r = results[1]  # Flip strategy
        seasons = {}
        for h in r['history']:
            date = h['date']
            if hasattr(date, 'year'):
                yr = date.year
                season = f"{yr-1}-{str(yr)[-2:]}" if date.month < 7 else f"{yr}-{str(yr+1)[-2:]}"
            else:
                season = str(date)[:4]
            if season not in seasons:
                seasons[season] = {'n': 0, 'wins': 0, 'pnl': 0}
            seasons[season]['n'] += 1
            seasons[season]['wins'] += 1 if h['pnl'] >= 0 else 0
            seasons[season]['pnl'] += h['pnl']

        print(f"  {'Season':<12s} {'Trades':>7s} {'WR':>7s} {'P&L':>12s}")
        for season in sorted(seasons.keys()):
            s = seasons[season]
            wr = s['wins'] / s['n'] if s['n'] > 0 else 0
            print(f"  {season:<12s} {s['n']:>7d} {wr:>6.1%} ${s['pnl']:>+11.2f}")


if __name__ == "__main__":
    main()
