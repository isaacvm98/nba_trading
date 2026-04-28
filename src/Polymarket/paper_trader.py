"""
Polymarket Paper Trading Bot

Simulates live betting on Polymarket to validate strategy before using real money.
Tracks hypothetical positions, monitors price movements, and logs exit signals.

Usage:
    # Initialize positions for today's games (run ~2 hours before games or start of day)
    python -m src.Polymarket.paper_trader --init

    # Monitor prices and check exit conditions (run periodically via cron)
    python -m src.Polymarket.paper_trader --monitor

    # View current positions and P&L
    python -m src.Polymarket.paper_trader --status

    # Close all positions (end of day or manual)
    python -m src.Polymarket.paper_trader --close-all
"""

import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.DataProviders.PolymarketOddsProvider import PolymarketOddsProvider
from src.DataProviders.InjuryProvider import InjuryProvider
from src.DataProviders.PriceHistoryProvider import PriceHistoryProvider, calculate_delta_adjustment
from src.Utils.tools import get_json_data, to_data_frame
from src.Utils.Kelly_Criterion import calculate_kelly_criterion, calculate_tiered_kelly
from src.Utils.InjuryAdjustment import calculate_injury_adjustment, format_injury_adjustment
from src.Utils.Dictionaries import team_index_current
from src.Utils.DrawdownManager import DrawdownManager

# Configuration
DATA_DIR = Path("Data/paper_trading")

# Timing settings
MINUTES_BEFORE_GAME = 10  # Only bet on games starting within this window

# Price delta adjustment settings (primary)
PRICE_DELTA_ENABLED = True
PRICE_DELTA_LOOKBACK_HOURS = 24  # Look back 24 hours for price movement
PRICE_DELTA_THRESHOLD = 0.05  # Start blending at 5% move
PRICE_DELTA_MAX_BLEND = 0.50  # Max 50% blend toward market at large moves
PRICE_DELTA_SKIP_THRESHOLD = 0.10  # Skip bet entirely if move > 10%

# Injury adjustment settings (sanity check)
INJURY_ADJUSTMENT_ENABLED = True
INJURY_MAX_ADJUSTMENT = 0.10  # Cap at 10%
INJURY_ADJUSTMENT_FACTOR = 0.025  # 2.5% per unit of impact
INJURY_INCLUDE_QUESTIONABLE = False  # Only count OUT players

# Kelly Criterion settings
KELLY_FRACTION = 0.25  # Use 25% of calculated Kelly (fractional Kelly)
KELLY_MAX_BET = 0.10  # Max 10% of bankroll per bet
KELLY_MAX_DAILY = 0.50  # Max 50% total daily exposure
USE_TIERED_KELLY = True  # Use tiered Kelly sizing based on edge magnitude
POSITIONS_FILE = DATA_DIR / "positions.json"
TRADES_LOG = DATA_DIR / "trades.json"

# Strategy parameters
STARTING_BANKROLL = 1000  # Starting bankroll for paper trading
BANKROLL_FILE = DATA_DIR / "bankroll.json"

# Risk Management Configuration
# Legacy early exits (stop-loss/take-profit on ALL bets) are DISABLED.
# Analysis shows that blanket early exits destroy edge by locking in losses on normal variance.
EARLY_EXIT_ENABLED = False  # Set to True for legacy stop-loss/take-profit on ALL bets

# Underdog-only take-profit: ENABLED by default.
# Backtest shows deep underdogs (<35% entry) have low win rates where capturing
# partial gains beats holding to resolution. Favorites hold to resolution.
UNDERDOG_TAKE_PROFIT_ENABLED = True
UNDERDOG_TAKE_PROFIT_MAX_ENTRY = 0.35  # Only apply TP to entry_prob below this

# Graduated take-profit thresholds for underdogs (derived from backtesting)
# Format: (max_entry_prob, take_profit_pct)
UNDERDOG_TAKE_PROFIT_BY_ENTRY = [
    (0.20, 0.25),   # Deep underdog (<20%): take profit at 25% price move
    (0.35, 0.08),   # Moderate underdog (20-35%): take profit at 8% price move
]

# Legacy settings - only apply if EARLY_EXIT_ENABLED = True
STOP_LOSS_PCT = 0.40       # Exit if price drops 40% from entry
TAKE_PROFIT_PCT = 0.50     # Exit if price rises 50% from entry

# Legacy dynamic take-profit (only used if EARLY_EXIT_ENABLED = True)
TAKE_PROFIT_BY_ENTRY = [
    # (max_entry_prob, take_profit_pct)
    (0.30, 0.12),   # Deep underdog (<30%): take profit at 12% move
    (0.40, 0.15),   # Underdog (30-40%): take profit at 15% move
    (0.50, 0.18),   # Slight underdog (40-50%): take profit at 18% move
    (0.60, 0.20),   # Slight favorite (50-60%): take profit at 20% move
    (0.70, 0.25),   # Favorite (60-70%): take profit at 25% move
    (1.00, 0.30),   # Heavy favorite (70%+): take profit at 30% move
]

# Position Limits
MAX_SIMULTANEOUS_POSITIONS = 6   # Max 6 open positions at once
MAX_SAME_TEAM_POSITIONS = 3      # Max 3 positions involving the same team

# Entry Criteria - Edge Thresholds
MIN_EDGE_THRESHOLD = 0.05  # Only bet if model_prob - market_prob > 5%
MAX_EDGE_THRESHOLD = 0.25  # Skip if edge > 25% (likely model error or stale data)

# Model Confidence Settings
MAX_CI_WIDTH = 0.18  # Skip bets where confidence interval width > 18%
MIN_CONFIDENCE = 0.10  # Skip bets where model confidence < 10% (too close to 50/50)

# Discord Webhook for Alerts (set to None to disable)
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1471009722772623361/2FEqyrUTnUGOEKrLB-RVZOdcL1Fg_xKzACG1eVEGevHrJ2kFi7gwvDfDn1c7qgcr68Wq"


def get_take_profit_threshold(model_prob, entry_prob=None):
    """Get take-profit threshold based on entry price.

    Underdogs get tighter take-profits since we're buying cheap.
    entry_prob: the probability at which we entered (what we paid for shares)
    model_prob: kept for backwards compatibility, used if entry_prob not provided
    """
    # Use entry_prob if provided, otherwise fall back to model_prob
    prob = entry_prob if entry_prob is not None else model_prob

    # For the side we bet, use the entry probability
    # (caller should pass the entry prob of the side we bet)
    for max_prob, take_profit in TAKE_PROFIT_BY_ENTRY:
        if prob <= max_prob:
            return take_profit

    return 0.30  # Default fallback


def is_underdog_position(position):
    """Get the entry probability of the bet side for a position.

    Returns:
        float or None: Entry probability of the bet side, or None if no bet.
        A position is an underdog if this value < 0.50.
    """
    bet_side = position.get('bet_side')
    if not bet_side:
        return None
    if bet_side == 'home':
        return position.get('entry_home_prob')
    else:
        return position.get('entry_away_prob')


def get_underdog_take_profit_threshold(entry_prob):
    """Get take-profit threshold for an underdog position.

    Only returns a threshold if entry_prob < UNDERDOG_TAKE_PROFIT_MAX_ENTRY
    and a matching bucket exists in UNDERDOG_TAKE_PROFIT_BY_ENTRY.

    Args:
        entry_prob: The entry probability on the bet side.

    Returns:
        float or None: Take-profit percentage threshold, or None if no TP applies.
    """
    if entry_prob is None or entry_prob >= UNDERDOG_TAKE_PROFIT_MAX_ENTRY:
        return None

    for max_prob, take_profit in UNDERDOG_TAKE_PROFIT_BY_ENTRY:
        if entry_prob <= max_prob:
            return take_profit

    return None


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_bankroll():
    """Load current bankroll from file."""
    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE, "r") as f:
            data = json.load(f)
            return data.get("bankroll", STARTING_BANKROLL)
    return STARTING_BANKROLL


def save_bankroll(bankroll):
    """Save bankroll to file."""
    ensure_data_dir()
    with open(BANKROLL_FILE, "w") as f:
        json.dump({"bankroll": bankroll, "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def _get_bet_amount(position):
    """Get the bet amount, preferring the stored entry-time value."""
    stored = position.get('bet_amount')
    if stored is not None:
        return stored
    # Fallback for older positions without stored bet_amount
    kelly_pct = position.get('bet_kelly', 0) / 100
    bankroll = load_bankroll()
    return bankroll * kelly_pct


def calculate_position_pnl(position, won):
    """Calculate P&L for a closed position (game resolved).

    Args:
        position: The position dict
        won: True if our bet won, False if lost

    Returns:
        pnl: Dollar amount won or lost
    """
    bet_side = position.get('bet_side')
    if not bet_side:
        return 0  # No bet placed

    bet_amount = _get_bet_amount(position)

    if bet_side == 'home':
        odds = position.get('entry_home_odds', 0)
    else:
        odds = position.get('entry_away_odds', 0)

    if won:
        # Calculate winnings based on American odds
        if odds > 0:
            pnl = bet_amount * (odds / 100)
        else:
            pnl = bet_amount * (100 / abs(odds))
    else:
        pnl = -bet_amount

    return round(pnl, 2)


def calculate_exit_pnl(position, exit_price_change):
    """Calculate P&L for an early exit (stop-loss or take-profit).

    In Polymarket, you buy shares at entry probability and sell at exit probability.
    P&L = position_size * price_change

    Args:
        position: The position dict
        exit_price_change: Percentage change in price (e.g., -0.15 for -15%)

    Returns:
        pnl: Dollar amount won or lost
    """
    bet_side = position.get('bet_side')
    if not bet_side:
        return 0

    bet_amount = _get_bet_amount(position)

    # P&L is simply the position size times the price change
    # If we bought at 0.50 and price went to 0.55, that's +10% on our position
    pnl = bet_amount * exit_price_change

    return round(pnl, 2)


# XGBoost Model Loading
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
DATA_URL = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2025-26&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
SCHEDULE_PATH = BASE_DIR / "Data" / "nba-2025-UTC.csv"

import re
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

_xgb_ml = None
_xgb_ml_calibrator = None


def _select_model_path(kind):
    """Select the best model based on accuracy."""
    import joblib
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    """Load calibrator if available."""
    import joblib
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    try:
        return joblib.load(calibration_path)
    except Exception:
        return None


def load_xgb_model():
    """Load XGBoost moneyline model."""
    global _xgb_ml, _xgb_ml_calibrator
    if _xgb_ml is None:
        ml_path = _select_model_path("ML")
        _xgb_ml = xgb.Booster()
        _xgb_ml.load_model(str(ml_path))
        _xgb_ml_calibrator = _load_calibrator(ml_path)
        print(f"Loaded model: {ml_path.name}")
    return _xgb_ml, _xgb_ml_calibrator


def get_model_predictions(games, odds):
    """
    Get XGBoost model predictions for the given games.

    Returns dict: {game_key: {'home_prob': float, 'away_prob': float}}
    """
    print("Loading team stats from NBA.com...")
    try:
        stats_json = get_json_data(DATA_URL)
        df = to_data_frame(stats_json)
    except Exception as e:
        print(f"Error fetching NBA stats: {e}")
        return {}

    print("Loading game schedule...")
    try:
        schedule_df = pd.read_csv(SCHEDULE_PATH, parse_dates=['Date'])
    except Exception as e:
        print(f"Error loading schedule: {e}")
        return {}

    print("Loading XGBoost model...")
    model, calibrator = load_xgb_model()

    today = datetime.now()
    predictions = {}

    # Build feature data for each game
    for game_key in odds.keys():
        home_team, away_team = game_key.split(":")

        # Check if teams exist in index
        if home_team not in team_index_current or away_team not in team_index_current:
            print(f"  Skipping {game_key}: team not found in index")
            continue

        try:
            # Get team stats
            home_team_series = df.iloc[team_index_current.get(home_team)]
            away_team_series = df.iloc[team_index_current.get(away_team)]

            # Calculate days rest
            home_games = schedule_df[
                (schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)
            ]
            away_games = schedule_df[
                (schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)
            ]

            previous_home_games = home_games.loc[
                home_games['Date'] <= today
            ].sort_values('Date', ascending=False).head(1)['Date']
            previous_away_games = away_games.loc[
                away_games['Date'] <= today
            ].sort_values('Date', ascending=False).head(1)['Date']

            if len(previous_home_games) > 0:
                last_home_date = previous_home_games.iloc[0]
                home_days_off = timedelta(days=1) + today - last_home_date
            else:
                home_days_off = timedelta(days=7)

            if len(previous_away_games) > 0:
                last_away_date = previous_away_games.iloc[0]
                away_days_off = timedelta(days=1) + today - last_away_date
            else:
                away_days_off = timedelta(days=7)

            # Build feature vector
            stats = pd.concat([home_team_series, away_team_series])
            stats['Days-Rest-Home'] = home_days_off.days
            stats['Days-Rest-Away'] = away_days_off.days

            # Remove non-numeric columns
            feature_data = stats.drop(labels=['TEAM_ID', 'TEAM_NAME'], errors='ignore')
            feature_data = feature_data.values.astype(float).reshape(1, -1)

            # Get prediction
            if calibrator is not None:
                probs = calibrator.predict_proba(feature_data)[0]
            else:
                probs = model.predict(xgb.DMatrix(feature_data))[0]

            # probs[0] = away win prob, probs[1] = home win prob
            if isinstance(probs, np.ndarray) and len(probs) >= 2:
                home_prob = float(probs[1])
                away_prob = float(probs[0])
            else:
                # Single probability output (home win prob)
                home_prob = float(probs) if not isinstance(probs, np.ndarray) else float(probs[0])
                away_prob = 1 - home_prob

            # Estimate model confidence based on probability extremity
            # Predictions closer to 50% are less confident
            # Confidence = how far from 50%, scaled to 0-1
            prob_extremity = abs(home_prob - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%

            # Estimate confidence interval width (wider = less confident)
            # Use inverse of extremity: low extremity = wide CI
            # Base CI width is ~20% for 50/50, narrows as prediction becomes extreme
            base_ci_width = 0.20
            ci_width = base_ci_width * (1 - prob_extremity * 0.5)  # Range: 10-20%

            predictions[game_key] = {
                'home_prob': home_prob,
                'away_prob': away_prob,
                'confidence': round(prob_extremity, 3),
                'ci_width': round(ci_width, 3),
                'days_rest_home': home_days_off.days,
                'days_rest_away': away_days_off.days,
            }

        except Exception as e:
            print(f"  Error processing {game_key}: {e}")
            continue

    return predictions


def load_positions():
    """Load current positions from file."""
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_positions(positions):
    """Save positions to file."""
    ensure_data_dir()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def log_trade(trade):
    """Append trade to trades log."""
    ensure_data_dir()
    trades = []
    if TRADES_LOG.exists():
        with open(TRADES_LOG, "r") as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADES_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def probability_to_price(prob):
    """Convert probability to Polymarket price (same thing, just clarity)."""
    return prob


def american_odds_to_probability(odds):
    """Convert American odds back to probability."""
    if odds is None:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def init_positions(force=False):
    """Initialize positions for today's games based on model predictions.

    Args:
        force: If True, fetch all active games regardless of start time.
               If False, only fetch games starting within MINUTES_BEFORE_GAME.
    """
    print("=" * 60)
    print("INITIALIZING PAPER TRADING POSITIONS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Get current Polymarket odds
    if force:
        print(f"\nFetching Polymarket odds (FORCE MODE - all active games)...")
        provider = PolymarketOddsProvider()
    else:
        print(f"\nFetching Polymarket odds (games starting within {MINUTES_BEFORE_GAME} minutes)...")
        provider = PolymarketOddsProvider(minutes_before_game=MINUTES_BEFORE_GAME)
    odds = provider.get_odds()

    if not odds:
        print("No games with valid odds found.")
        return

    print(f"Found {len(odds)} games with valid odds\n")

    # Get XGBoost model predictions
    print("-" * 40)
    model_predictions = get_model_predictions(list(odds.keys()), odds)
    print("-" * 40)
    print(f"\nGot predictions for {len(model_predictions)} games\n")

    # Initialize price history provider if enabled
    price_history_provider = None
    if PRICE_DELTA_ENABLED:
        print("Initializing price history provider...")
        price_history_provider = PriceHistoryProvider()
        print(f"Price delta lookback: {PRICE_DELTA_LOOKBACK_HOURS}h, threshold: {PRICE_DELTA_THRESHOLD:.0%}, skip at: {PRICE_DELTA_SKIP_THRESHOLD:.0%}\n")

    # Initialize injury provider if enabled (for sanity checks)
    injury_provider = None
    if INJURY_ADJUSTMENT_ENABLED:
        print("Fetching injury data...")
        all_teams = []
        for game_key in odds.keys():
            home_team, away_team = game_key.split(":")
            all_teams.extend([home_team, away_team])
        try:
            injury_provider = InjuryProvider(all_teams, include_questionable=INJURY_INCLUDE_QUESTIONABLE)
            print(f"Loaded injury data for {len(all_teams)} teams\n")
        except Exception as e:
            print(f"Warning: Failed to fetch injury data: {e}")
            print("Continuing without injury sanity checks\n")

    # Load existing positions and count limits
    positions = load_positions()

    # Count existing open positions with bets
    open_positions = [p for p in positions.values()
                      if p['status'] == 'open' and p.get('bet_side')]
    current_position_count = len(open_positions)

    # Count positions per team
    team_position_counts = {}
    for p in open_positions:
        home = p.get('home_team')
        away = p.get('away_team')
        if home:
            team_position_counts[home] = team_position_counts.get(home, 0) + 1
        if away:
            team_position_counts[away] = team_position_counts.get(away, 0) + 1

    print(f"Open positions: {current_position_count}/{MAX_SIMULTANEOUS_POSITIONS}")

    # Collect all potential bets for prioritization
    potential_bets = []

    # For each game, evaluate for potential bet
    for game_key, game_odds in odds.items():
        home_team, away_team = game_key.split(":")

        home_ml = game_odds[home_team]['money_line_odds']
        away_ml = game_odds[away_team]['money_line_odds']
        ou_line = game_odds['under_over_odds']

        # Get token IDs for WebSocket price monitoring
        home_token_id = game_odds.get('home_token_id')
        away_token_id = game_odds.get('away_token_id')

        if home_ml is None or away_ml is None:
            continue

        # Convert Polymarket odds to probabilities
        market_home_prob = american_odds_to_probability(home_ml)
        market_away_prob = american_odds_to_probability(away_ml)

        # Get model prediction (or fall back to market if not available)
        if game_key in model_predictions:
            pred = model_predictions[game_key]
            model_home_prob = pred['home_prob']
            model_away_prob = pred['away_prob']
            model_confidence = pred.get('confidence', 0.5)
            model_ci_width = pred.get('ci_width', 0.15)
        else:
            print(f"  No model prediction for {game_key}, skipping...")
            continue

        # Initialize tracking variables
        adjusted_home_prob = model_home_prob
        adjusted_away_prob = model_away_prob
        home_delta = 0.0
        away_delta = 0.0
        delta_blend = 0.0
        delta_reason = "no_data"
        skip_due_to_delta = False

        # PRIMARY: Get price delta and adjust model probability
        if price_history_provider:
            home_token = game_odds.get('home_token_id')
            away_token = game_odds.get('away_token_id')

            if home_token:
                home_delta_data = price_history_provider.get_price_delta(home_token, PRICE_DELTA_LOOKBACK_HOURS)
                home_delta = home_delta_data['delta']

                # Check if we should skip this bet entirely (large move)
                if abs(home_delta) >= PRICE_DELTA_SKIP_THRESHOLD:
                    skip_due_to_delta = True
                    delta_reason = f"skip_large_move_{home_delta:+.1%}"
                else:
                    # Apply delta adjustment to model probability
                    adjusted_home_prob, delta_blend, delta_reason = calculate_delta_adjustment(
                        model_home_prob,
                        market_home_prob,
                        home_delta,
                        delta_threshold=PRICE_DELTA_THRESHOLD,
                        max_blend=PRICE_DELTA_MAX_BLEND
                    )
                    adjusted_away_prob = 1 - adjusted_home_prob

            if away_token:
                away_delta_data = price_history_provider.get_price_delta(away_token, PRICE_DELTA_LOOKBACK_HOURS)
                away_delta = away_delta_data['delta']

        # SANITY CHECK: Get injury data to explain/validate price movement
        home_injuries = {'injured_players': [], 'total_impact': 0.0}
        away_injuries = {'injured_players': [], 'total_impact': 0.0}
        home_out_players = []
        away_out_players = []

        if injury_provider:
            home_injuries = injury_provider.get_team_injuries(home_team)
            away_injuries = injury_provider.get_team_injuries(away_team)

            home_out_players = [p['name'] for p in home_injuries['injured_players']
                               if p['designation'].lower() in ['out', 'o']]
            away_out_players = [p['name'] for p in away_injuries['injured_players']
                               if p['designation'].lower() in ['out', 'o']]

        # Calculate edge using adjusted probabilities
        home_edge = adjusted_home_prob - market_home_prob
        away_edge = adjusted_away_prob - market_away_prob

        # Calculate Kelly sizing based on adjusted probabilities
        try:
            if USE_TIERED_KELLY:
                # Tiered Kelly adjusts sizing based on edge magnitude
                kelly_home = calculate_tiered_kelly(
                    home_ml, adjusted_home_prob, market_home_prob,
                    kelly_fraction=KELLY_FRACTION, max_bet_pct=KELLY_MAX_BET * 100
                )
                kelly_away = calculate_tiered_kelly(
                    away_ml, adjusted_away_prob, market_away_prob,
                    kelly_fraction=KELLY_FRACTION, max_bet_pct=KELLY_MAX_BET * 100
                )
            else:
                # Standard Kelly
                kelly_home = calculate_kelly_criterion(home_ml, adjusted_home_prob)
                kelly_away = calculate_kelly_criterion(away_ml, adjusted_away_prob)
        except:
            kelly_home = kelly_away = 0

        # Determine which side to bet (positive Kelly = positive edge)
        position_id = f"{game_key}_{datetime.now().strftime('%Y%m%d')}"

        if position_id in positions:
            print(f"Position already exists: {away_team} @ {home_team}")
            continue

        # Determine bet side based on Kelly (unless skipped due to large delta)
        bet_side = None
        bet_kelly = 0
        raw_kelly = 0
        skip_reason = None

        if skip_due_to_delta:
            skip_reason = "large_price_move"
        elif kelly_home > 0 and kelly_home >= kelly_away:
            bet_side = "home"
            raw_kelly = kelly_home
            bet_edge = home_edge
        elif kelly_away > 0:
            bet_side = "away"
            raw_kelly = kelly_away
            bet_edge = away_edge

        # Apply edge thresholds
        if bet_side and not skip_reason:
            if bet_edge < MIN_EDGE_THRESHOLD:
                skip_reason = f"edge_too_small_{bet_edge:.1%}"
                bet_side = None
            elif bet_edge > MAX_EDGE_THRESHOLD:
                skip_reason = f"edge_too_large_{bet_edge:.1%}"
                bet_side = None

        # Check model confidence
        if bet_side and not skip_reason:
            if model_ci_width > MAX_CI_WIDTH:
                skip_reason = f"low_confidence_ci_{model_ci_width:.1%}"
                bet_side = None
            elif model_confidence < MIN_CONFIDENCE:
                skip_reason = f"low_confidence_{model_confidence:.1%}"
                bet_side = None

        # Check position limits
        if bet_side and not skip_reason:
            # Check total position limit
            if current_position_count >= MAX_SIMULTANEOUS_POSITIONS:
                skip_reason = "max_positions_reached"
                bet_side = None
            # Check per-team limits
            elif team_position_counts.get(home_team, 0) >= MAX_SAME_TEAM_POSITIONS:
                skip_reason = f"max_team_positions_{home_team}"
                bet_side = None
            elif team_position_counts.get(away_team, 0) >= MAX_SAME_TEAM_POSITIONS:
                skip_reason = f"max_team_positions_{away_team}"
                bet_side = None

        # Apply fractional Kelly and caps
        if raw_kelly > 0 and bet_side:
            if USE_TIERED_KELLY:
                # Tiered Kelly already applies fraction and caps
                bet_kelly = raw_kelly
            else:
                # Standard Kelly needs fraction and caps applied
                bet_kelly = raw_kelly * KELLY_FRACTION  # Fractional Kelly
                bet_kelly = min(bet_kelly, KELLY_MAX_BET * 100)  # Per-bet cap (convert to %)

            # Update position counts for limit tracking
            current_position_count += 1
            team_position_counts[home_team] = team_position_counts.get(home_team, 0) + 1
            team_position_counts[away_team] = team_position_counts.get(away_team, 0) + 1

        # Snapshot bet_amount at entry time for accurate P&L calculation
        entry_bankroll = load_bankroll()
        entry_bet_amount = (bet_kelly / 100) * entry_bankroll if bet_kelly and bet_side else 0

        position = {
            "game_key": game_key,
            "home_team": home_team,
            "away_team": away_team,
            "home_token_id": home_token_id,
            "away_token_id": away_token_id,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_home_prob": market_home_prob,
            "entry_away_prob": market_away_prob,
            "entry_home_odds": home_ml,
            "entry_away_odds": away_ml,
            "ou_line": ou_line,
            "model_home_prob": model_home_prob,
            "model_away_prob": model_away_prob,
            "model_confidence": model_confidence,
            "model_ci_width": model_ci_width,
            "adjusted_home_prob": adjusted_home_prob,
            "adjusted_away_prob": adjusted_away_prob,
            "home_price_delta_24h": home_delta,
            "away_price_delta_24h": away_delta,
            "delta_blend_factor": delta_blend,
            "delta_reason": delta_reason,
            "home_injury_impact": home_injuries['total_impact'],
            "away_injury_impact": away_injuries['total_impact'],
            "home_out_players": home_out_players,
            "away_out_players": away_out_players,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "kelly_home": kelly_home,
            "kelly_away": kelly_away,
            "raw_kelly": raw_kelly,
            "bet_side": bet_side,
            "bet_kelly": bet_kelly,
            "bet_amount": round(entry_bet_amount, 2),
            "entry_bankroll": round(entry_bankroll, 2),
            "skip_reason": skip_reason,
            "status": "open",
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl": None
        }

        positions[position_id] = position

        # Display with color coding for edge
        home_edge_str = f"{home_edge:+.1%}"
        away_edge_str = f"{away_edge:+.1%}"

        print(f"{away_team} @ {home_team}")
        print(f"  Market:  Home {market_home_prob:.1%} ({home_ml:+d}) | Away {market_away_prob:.1%} ({away_ml:+d})")
        print(f"  Model:   Home {model_home_prob:.1%} | Away {model_away_prob:.1%}")

        # Display price delta information (primary adjustment)
        if price_history_provider:
            delta_str = f"Home {home_delta:+.1%} | Away {away_delta:+.1%}"
            if skip_due_to_delta:
                print(f"  24h Delta: {delta_str} >>> SKIP (move > {PRICE_DELTA_SKIP_THRESHOLD:.0%})")
            elif delta_blend > 0:
                print(f"  24h Delta: {delta_str} (blend {delta_blend:.0%} toward market)")
            else:
                print(f"  24h Delta: {delta_str} (stable)")

        # Display injury information (sanity check)
        if injury_provider:
            home_out_str = ", ".join(home_out_players) if home_out_players else "none"
            away_out_str = ", ".join(away_out_players) if away_out_players else "none"
            print(f"  Injuries: Home [{home_out_str}] | Away [{away_out_str}]")

        # Show adjusted probability
        if adjusted_home_prob != model_home_prob:
            print(f"  Adjusted: Home {adjusted_home_prob:.1%} | Away {adjusted_away_prob:.1%}")

        print(f"  Edge:    Home {home_edge_str} | Away {away_edge_str}")
        print(f"  Conf:    {model_confidence:.0%} (CI width: {model_ci_width:.1%})")
        print(f"  Kelly:   Home {kelly_home:.1f}% | Away {kelly_away:.1f}%")
        if bet_side:
            if USE_TIERED_KELLY:
                print(f"  >>> BET: {bet_side.upper()} {bet_kelly:.1f}% (tiered Kelly)")
            else:
                print(f"  >>> BET: {bet_side.upper()} {bet_kelly:.1f}% (raw {raw_kelly:.1f}% x {KELLY_FRACTION:.0%})")
        elif skip_reason:
            # Show specific skip reason
            if "edge_too_small" in skip_reason:
                print(f"  >>> NO BET (edge below {MIN_EDGE_THRESHOLD:.0%} threshold)")
            elif "edge_too_large" in skip_reason:
                print(f"  >>> NO BET (edge above {MAX_EDGE_THRESHOLD:.0%} - possible model error)")
            elif "max_positions" in skip_reason:
                print(f"  >>> NO BET (at max {MAX_SIMULTANEOUS_POSITIONS} positions)")
            elif "max_team" in skip_reason:
                print(f"  >>> NO BET (team at max {MAX_SAME_TEAM_POSITIONS} positions)")
            elif "large_price_move" in skip_reason:
                print(f"  >>> NO BET (large price movement)")
            elif "low_confidence" in skip_reason:
                print(f"  >>> NO BET (low model confidence)")
            else:
                print(f"  >>> NO BET ({skip_reason})")
        else:
            print(f"  >>> NO BET (no positive edge)")
        print(f"  O/U: {ou_line}")
        print()

        # Log the entry
        log_trade({
            "type": "ENTRY",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": position_id,
            "game": f"{away_team} @ {home_team}",
            "market_home_prob": market_home_prob,
            "market_away_prob": market_away_prob,
            "model_home_prob": model_home_prob,
            "model_away_prob": model_away_prob,
            "adjusted_home_prob": adjusted_home_prob,
            "adjusted_away_prob": adjusted_away_prob,
            "home_price_delta_24h": home_delta,
            "away_price_delta_24h": away_delta,
            "delta_blend_factor": delta_blend,
            "delta_reason": delta_reason,
            "home_injury_impact": home_injuries['total_impact'],
            "away_injury_impact": away_injuries['total_impact'],
            "home_out_players": home_out_players,
            "away_out_players": away_out_players,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "kelly_home": kelly_home,
            "kelly_away": kelly_away,
            "bet_side": bet_side,
            "home_odds": home_ml,
            "away_odds": away_ml
        })

    save_positions(positions)

    # Summary
    open_positions = [p for p in positions.values() if p['status'] == 'open']
    bets = [p for p in open_positions if p.get('bet_side')]
    skipped = [p for p in open_positions if 'skip' in p.get('delta_reason', '')]
    print("=" * 60)
    print(f"Total positions: {len(open_positions)}")
    print(f"Positions with bets: {len(bets)}")
    print(f"Skipped (large price move): {len(skipped)}")
    if bets:
        total_kelly = sum(p['bet_kelly'] for p in bets)
        total_raw = sum(p.get('raw_kelly', 0) for p in bets)
        print(f"Total allocation: {total_kelly:.1f}% (raw: {total_raw:.1f}%)")
        if total_kelly > KELLY_MAX_DAILY * 100:
            print(f"  WARNING: Exceeds daily cap of {KELLY_MAX_DAILY:.0%} - consider reducing bets")


def monitor_positions():
    """Monitor open positions and log current prices.

    Exit logic:
    - EARLY_EXIT_ENABLED: Legacy stop-loss + take-profit on all bets
    - UNDERDOG_TAKE_PROFIT_ENABLED: Take-profit only on underdogs (<35% entry)
    - Neither: All positions run to resolution
    """
    print("=" * 60)
    print("MONITORING POSITIONS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    if EARLY_EXIT_ENABLED:
        print("Legacy early exits ENABLED (all bets)")
    elif UNDERDOG_TAKE_PROFIT_ENABLED:
        print(f"Underdog take-profit ENABLED (entry < {UNDERDOG_TAKE_PROFIT_MAX_ENTRY:.0%})")
    else:
        print("All exits DISABLED - positions will run to resolution")
    print("=" * 60)

    positions = load_positions()
    open_positions = {k: v for k, v in positions.items() if v['status'] == 'open'}

    if not open_positions:
        print("No open positions to monitor.")
        return

    # Get current Polymarket odds
    provider = PolymarketOddsProvider()
    current_odds = provider.get_odds()

    for position_id, position in open_positions.items():
        game_key = position['game_key']
        home_team = position['home_team']
        away_team = position['away_team']

        print(f"\n{away_team} @ {home_team}")

        if game_key not in current_odds:
            print("  Game not found in current odds (may have started/ended)")
            continue

        game_odds = current_odds[game_key]
        current_home_ml = game_odds[home_team]['money_line_odds']
        current_away_ml = game_odds[away_team]['money_line_odds']

        if current_home_ml is None:
            print("  Market resolved or invalid odds")
            # Don't close positions when market disappears - wait for resolution
            continue

        current_home_prob = american_odds_to_probability(current_home_ml)
        current_away_prob = american_odds_to_probability(current_away_ml)

        entry_home_prob = position['entry_home_prob']
        entry_away_prob = position['entry_away_prob']

        # Calculate price change
        home_change = (current_home_prob - entry_home_prob) / entry_home_prob
        away_change = (current_away_prob - entry_away_prob) / entry_away_prob

        print(f"  Entry: Home {entry_home_prob:.1%} | Away {entry_away_prob:.1%}")
        print(f"  Current: Home {current_home_prob:.1%} | Away {current_away_prob:.1%}")
        print(f"  Change: Home {home_change:+.1%} | Away {away_change:+.1%}")

        # Track max profit/drawdown during position lifetime for analytics
        bet_side = position.get('bet_side')
        if bet_side:
            current_change = home_change if bet_side == 'home' else away_change
            position['current_price_change'] = current_change
            # Track max/min for analytics
            if 'max_profit_pct' not in position:
                position['max_profit_pct'] = current_change
                position['max_drawdown_pct'] = current_change
            else:
                position['max_profit_pct'] = max(position['max_profit_pct'], current_change)
                position['max_drawdown_pct'] = min(position['max_drawdown_pct'], current_change)

        # === EXIT LOGIC ===
        exit_triggered = False
        exit_reason = None

        if EARLY_EXIT_ENABLED:
            # Legacy: stop-loss and take-profit on ALL positions
            model_home_prob = position.get('model_home_prob', 0.5)
            take_profit_pct = get_take_profit_threshold(model_home_prob)

            if position['kelly_home'] > 0:
                if home_change <= -STOP_LOSS_PCT:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS (Home dropped {home_change:.1%})"
                elif take_profit_pct is not None and home_change >= take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (Home up {home_change:.1%}, threshold {take_profit_pct:.0%})"

            if position['kelly_away'] > 0:
                if away_change <= -STOP_LOSS_PCT:
                    exit_triggered = True
                    exit_reason = f"STOP_LOSS (Away dropped {away_change:.1%})"
                elif take_profit_pct is not None and away_change >= take_profit_pct:
                    exit_triggered = True
                    exit_reason = f"TAKE_PROFIT (Away up {away_change:.1%}, threshold {take_profit_pct:.0%})"

            tp_str = f"{take_profit_pct:.0%}" if take_profit_pct else "disabled (high confidence)"
            print(f"  Take-profit threshold: {tp_str}")

        elif UNDERDOG_TAKE_PROFIT_ENABLED and bet_side:
            # Underdog-only take-profit (no stop-loss)
            entry_prob = entry_home_prob if bet_side == 'home' else entry_away_prob

            if entry_prob < UNDERDOG_TAKE_PROFIT_MAX_ENTRY:
                tp_threshold = get_underdog_take_profit_threshold(entry_prob)

                if tp_threshold is not None:
                    price_change_for_side = home_change if bet_side == 'home' else away_change

                    if price_change_for_side >= tp_threshold:
                        exit_triggered = True
                        exit_reason = (
                            f"UNDERDOG_TAKE_PROFIT ({bet_side.title()} up "
                            f"{price_change_for_side:.1%}, threshold {tp_threshold:.0%}, "
                            f"entry {entry_prob:.1%})"
                        )

                    print(f"  Underdog TP: {price_change_for_side:+.1%} / {tp_threshold:.0%} threshold")
                else:
                    print(f"  Status: Holding (no TP bucket for entry {entry_prob:.1%})")
            else:
                print(f"  Status: Holding (entry {entry_prob:.1%} >= {UNDERDOG_TAKE_PROFIT_MAX_ENTRY:.0%}, not TP eligible)")
        else:
            print(f"  Status: Holding (waiting for resolution)")

        if exit_triggered:
            print(f"  *** EXIT SIGNAL: {exit_reason} ***")
            position['status'] = 'exit_signal'
            position['exit_reason'] = exit_reason
            position['exit_time'] = datetime.now(timezone.utc).isoformat()
            position['exit_home_prob'] = current_home_prob
            position['exit_away_prob'] = current_away_prob

            log_trade({
                "type": "EXIT_SIGNAL",
                "time": datetime.now(timezone.utc).isoformat(),
                "position_id": position_id,
                "game": f"{away_team} @ {home_team}",
                "reason": exit_reason,
                "entry_home_prob": entry_home_prob,
                "exit_home_prob": current_home_prob,
                "change": home_change
            })

    save_positions(positions)


def show_status():
    """Show current positions and P&L summary."""
    print("=" * 60)
    print("PAPER TRADING STATUS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    positions = load_positions()

    if not positions:
        print("No positions found.")
        return

    open_positions = [p for p in positions.values() if p['status'] == 'open']
    exit_signals = [p for p in positions.values() if p['status'] == 'exit_signal']
    closed = [p for p in positions.values() if p['status'] == 'closed']

    print(f"\nOpen positions: {len(open_positions)}")
    print(f"Exit signals: {len(exit_signals)}")
    print(f"Closed: {len(closed)}")

    if open_positions:
        print("\n--- OPEN POSITIONS ---")
        for p in open_positions:
            print(f"{p['away_team']} @ {p['home_team']}")
            print(f"  Market: Home {p['entry_home_prob']:.1%} ({p['entry_home_odds']:+d})")
            print(f"  Model:  Home {p.get('model_home_prob', 0):.1%} | Away {p.get('model_away_prob', 0):.1%}")
            # Show price delta if present
            if p.get('home_price_delta_24h') is not None:
                home_delta = p.get('home_price_delta_24h', 0)
                delta_reason = p.get('delta_reason', 'unknown')
                print(f"  24h Delta: Home {home_delta:+.1%} ({delta_reason})")
            # Show injury info if present
            home_out = p.get('home_out_players', [])
            away_out = p.get('away_out_players', [])
            if home_out or away_out:
                home_out_str = ", ".join(home_out) if home_out else "none"
                away_out_str = ", ".join(away_out) if away_out else "none"
                print(f"  Injuries: Home [{home_out_str}] | Away [{away_out_str}]")
            # Show adjusted probability if different from model
            if p.get('adjusted_home_prob') and p.get('adjusted_home_prob') != p.get('model_home_prob'):
                print(f"  Adjusted: Home {p.get('adjusted_home_prob', 0):.1%} | Away {p.get('adjusted_away_prob', 0):.1%}")
            print(f"  Edge:   Home {p.get('home_edge', 0):+.1%} | Away {p.get('away_edge', 0):+.1%}")
            bet_side = p.get('bet_side')
            if bet_side:
                print(f"  BET:    {bet_side.upper()} ({p.get('bet_kelly', 0):.1f}%)")
            else:
                reason = p.get('delta_reason', '')
                if 'skip' in reason:
                    print(f"  BET:    None (large price movement)")
                else:
                    print(f"  BET:    None (no edge)")

    if exit_signals:
        print("\n--- EXIT SIGNALS ---")
        for p in exit_signals:
            print(f"{p['away_team']} @ {p['home_team']}")
            print(f"  Reason: {p['exit_reason']}")
            print(f"  Entry: {p['entry_home_prob']:.1%} -> Exit: {p.get('exit_home_prob', 'N/A')}")


def close_all():
    """Close all open positions (end of day)."""
    print("=" * 60)
    print("CLOSING ALL POSITIONS")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    positions = load_positions()

    # Get final prices
    provider = PolymarketOddsProvider()
    final_odds = provider.get_odds()

    for position_id, position in positions.items():
        if position['status'] != 'open':
            continue

        game_key = position['game_key']
        home_team = position['home_team']
        away_team = position['away_team']

        position['status'] = 'closed'
        position['exit_time'] = datetime.now(timezone.utc).isoformat()
        position['exit_reason'] = 'manual_close'

        if game_key in final_odds:
            game_odds = final_odds[game_key]
            if game_odds[home_team]['money_line_odds']:
                final_home_prob = american_odds_to_probability(game_odds[home_team]['money_line_odds'])
                position['exit_home_prob'] = final_home_prob

        print(f"Closed: {away_team} @ {home_team}")

        log_trade({
            "type": "CLOSE",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": position_id,
            "game": f"{away_team} @ {home_team}",
            "reason": "manual_close"
        })

    save_positions(positions)
    print("\nAll positions closed.")


def show_drawdown_status():
    """Show current drawdown tracking status."""
    drawdown_manager = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    drawdown_manager.sync_bankroll(load_bankroll())
    status = drawdown_manager.get_status()

    print("=" * 60)
    print("DRAWDOWN TRACKING STATUS")
    print("=" * 60)

    print(f"\nBankroll: ${status['current_bankroll']:.2f}")
    print(f"Peak:     ${status['peak_bankroll']:.2f}")

    print(f"\nDaily P&L:    ${status['daily_pnl']:+.2f}")
    print(f"Weekly P&L:   ${status['weekly_pnl']:+.2f}")
    print(f"Total DD:     {status['total_drawdown']:.1%}")


def show_dashboard():
    """Show comprehensive dashboard with all trading status."""
    from src.Utils.PerformanceAnalytics import PerformanceAnalytics

    print("=" * 70)
    print("                     NBA BETTING DASHBOARD")
    print(f"                    {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Bankroll & Risk Status
    print("\n" + "-" * 30 + " BANKROLL " + "-" * 30)
    bankroll = load_bankroll()
    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
    dm.sync_bankroll(bankroll)
    status = dm.get_status()

    print(f"Current: ${bankroll:.2f}  |  Peak: ${status['peak_bankroll']:.2f}")
    print(f"\nDaily P&L:  ${status['daily_pnl']:+.2f}")
    print(f"Weekly P&L: ${status['weekly_pnl']:+.2f}")
    print(f"Drawdown:   {status['total_drawdown']*100:+.1f}%")

    # Open Positions
    print("\n" + "-" * 28 + " POSITIONS " + "-" * 28)
    positions = load_positions()
    open_positions = [p for p in positions.values() if p['status'] == 'open' and p.get('bet_side')]

    if open_positions:
        print(f"Open: {len(open_positions)}/{MAX_SIMULTANEOUS_POSITIONS}")
        print()
        for p in open_positions:
            bet_side = p.get('bet_side', '?')
            edge = p.get(f'{bet_side}_edge', 0)
            kelly = p.get('bet_kelly', 0)
            current_change = p.get('current_price_change', 0)

            # Color code based on current P&L
            if current_change > 0:
                change_str = f"\033[92m{current_change:+.1%}\033[0m"
            elif current_change < 0:
                change_str = f"\033[91m{current_change:+.1%}\033[0m"
            else:
                change_str = f"{current_change:+.1%}"

            print(f"  {p['away_team']:3s} @ {p['home_team']:3s} | {bet_side.upper():4s} {kelly:.1f}% | Edge: {edge:+.1%} | Now: {change_str}")
    else:
        print("No open positions with bets")

    # Recent Performance
    print("\n" + "-" * 26 + " PERFORMANCE " + "-" * 26)
    try:
        analytics = PerformanceAnalytics(data_dir=DATA_DIR)
        resolved = analytics.get_resolved_positions(days=7)
        closed = analytics.get_closed_positions(days=7)

        if resolved:
            wins = sum(1 for p in resolved if p.get('won'))
            resolved_pnl = sum(p.get('pnl', 0) for p in resolved)
            print(f"Resolved (7d): {wins}/{len(resolved)} wins ({wins/len(resolved)*100:.0f}%) | ${resolved_pnl:+.2f}")

        if closed:
            closed_pnl = sum(p.get('pnl', 0) for p in closed)
            print(f"Early Exits:   {len(closed)} positions | ${closed_pnl:+.2f}")

        # Edge bucket summary
        if resolved:
            bucket_stats = analytics.analyze_by_edge_bucket(resolved)
            if bucket_stats:
                print("\nBy Edge:")
                for bucket_name in ["5-7%", "7-10%", "10%+"]:
                    if bucket_name in bucket_stats:
                        s = bucket_stats[bucket_name]
                        print(f"  {bucket_name}: {s['wins']}/{s['count']} ({s['win_rate']*100:.0f}%) ${s['total_pnl']:+.2f}")
    except Exception as e:
        print(f"Performance data unavailable: {e}")

    # Configuration Summary
    print("\n" + "-" * 27 + " SETTINGS " + "-" * 27)
    early_exit_str = "ENABLED" if EARLY_EXIT_ENABLED else "DISABLED"
    underdog_tp_str = f"ENABLED (<{UNDERDOG_TAKE_PROFIT_MAX_ENTRY:.0%} entry)" if UNDERDOG_TAKE_PROFIT_ENABLED else "DISABLED"
    tiered_str = "ENABLED" if USE_TIERED_KELLY else "DISABLED"
    print(f"Legacy Early Exits: {early_exit_str}")
    print(f"Underdog Take-Profit: {underdog_tp_str}")
    print(f"Tiered Kelly: {tiered_str}")
    print(f"Edge Range: {MIN_EDGE_THRESHOLD:.0%} - {MAX_EDGE_THRESHOLD:.0%}")
    print(f"Position Limits: {MAX_SIMULTANEOUS_POSITIONS} total, {MAX_SAME_TEAM_POSITIONS} per team")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Polymarket Paper Trading Bot')
    parser.add_argument('--init', action='store_true', help='Initialize positions for today\'s games')
    parser.add_argument('--force', action='store_true', help='Force init for all active games (ignore time window)')
    parser.add_argument('--monitor', action='store_true', help='Monitor open positions')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--close-all', action='store_true', help='Close all positions')
    parser.add_argument('--drawdown-status', action='store_true', help='Show drawdown tracking status')
    parser.add_argument('--dashboard', action='store_true', help='Show comprehensive dashboard')

    args = parser.parse_args()

    if args.dashboard:
        show_dashboard()
    elif args.init:
        init_positions(force=args.force)
    elif args.monitor:
        monitor_positions()
    elif args.status:
        show_status()
    elif args.close_all:
        close_all()
    elif args.drawdown_status:
        show_drawdown_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
