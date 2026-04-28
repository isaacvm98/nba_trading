"""
Polymarket Paper Trading Scheduler — V2 Dual-Leg Strategy

Smart scheduling - places bets 10 minutes before each game starts.
Dual-leg strategy from Stage 4 backtest:
  Leg 1 (Favorites): edge >= 7%, model conf >= 60% -> hold to binary resolution
  Leg 2 (Underdogs): edge >= 7%, conf < 60%, entry >= $0.30 -> ESPN WP exits

Usage:
    python -m src.Polymarket.scheduler

    # Or run in background (Windows):
    start /B python -m src.Polymarket.scheduler > logs/scheduler.log 2>&1
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from src.DataProviders.PolymarketOddsProvider import PolymarketOddsProvider
from src.Polymarket.websocket_monitor import WebSocketPriceMonitor

# V2 strategy: imports from paper_trader_v2 for positions/bankroll/trades
from src.Polymarket.paper_trader_v2 import (
    load_positions,
    save_positions,
    log_trade,
    load_bankroll,
    save_bankroll,
    DATA_DIR,
    STARTING_BANKROLL,
    # Strategy parameters
    MIN_EDGE,
    MIN_ENTRY_PRICE,
    MAX_BET_PCT,
    FAV_MIN_CONF,
    DOG_MIN_ENTRY,
    ESPN_SL_THRESH,
    ESPN_TP_THRESH,
    Q1_UNDERDOG_EXIT,
    half_kelly_size,
)

# Model prediction pipeline (shared between V1 and V2)
from src.Polymarket.paper_trader import (
    get_model_predictions,
    american_odds_to_probability,
    DISCORD_WEBHOOK_URL,
)

from src.Utils.DrawdownManager import DrawdownManager
from src.Utils.AlertManager import AlertManager
from src.Polymarket.price_logger import get_price_logger
from src.DataProviders.espn_wp_logger import get_espn_wp_logger

try:
    from src.DataProviders.ESPNProvider import ESPNProvider
    _ESPN_AVAILABLE = True
except ImportError:
    _ESPN_AVAILABLE = False

# Configuration
MINUTES_BEFORE_GAME = 10   # Place bets X minutes before game
MONITOR_INTERVAL = 5    # Monitor every X minutes
LOCAL_TIMEZONE_OFFSET = -5  # EST = UTC-5 (adjust for your timezone)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "scheduler.log"

# Polymarket API
GAMMA_API_URL = "https://gamma-api.polymarket.com"
NBA_SERIES_ID = "10345"
GAMES_TAG_ID = "100639"


def setup_logging():
    """Set up logging to file and console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_todays_games_with_times():
    """Fetch today's games with their start times from Polymarket."""
    try:
        params = {
            "series_id": NBA_SERIES_ID,
            "tag_id": GAMES_TAG_ID,
            "active": "true",
            "closed": "false",
            "order": "startTime",
            "ascending": "true",
            "limit": 50
        }

        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        now = datetime.now(timezone.utc)
        today = now.date()
        tomorrow = today + timedelta(days=1)

        games = []
        for event in events:
            end_date_str = event.get("endDate", "")
            if not end_date_str:
                continue

            try:
                # endDate is the game time
                game_time = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                game_date = game_time.date()

                # Only include games starting within the next 18 hours
                # This catches today's games without pulling in tomorrow's full slate
                hours_until_game = (game_time - now).total_seconds() / 3600
                if hours_until_game > 18 or hours_until_game < 0:
                    continue

                # Skip games that already started
                if game_time <= now:
                    continue

                title = event.get("title", "")
                games.append({
                    "title": title,
                    "event": event,
                    "game_time": game_time,
                    "bet_time": game_time - timedelta(minutes=MINUTES_BEFORE_GAME)
                })

            except (ValueError, TypeError):
                continue

        # Sort by game time
        games.sort(key=lambda x: x["game_time"])
        return games

    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return []


def init_single_game(event, game_time, logger, alert_mgr=None, ws_monitor=None):
    """Initialize position for a single game using V2 dual-leg strategy.

    Leg 1 (Favorites): edge >= 7%, model conf >= 60% -> hold to resolution
    Leg 2 (Underdogs): edge >= 7%, conf < 60%, entry >= $0.30 -> ESPN WP exits

    Args:
        event: Polymarket event data
        game_time: Game start time (datetime)
        logger: Logger instance
        alert_mgr: AlertManager for Discord notifications
        ws_monitor: WebSocketPriceMonitor for real-time prices
    """
    title = event.get("title", "")
    logger.info(f"Initializing position for: {title}")

    # Get current odds for this game
    provider = PolymarketOddsProvider()
    all_odds = provider.get_odds()

    # Find matching game
    matching_key = None
    for game_key in all_odds.keys():
        home, away = game_key.split(":")
        if home.split()[-1] in title or away.split()[-1] in title:
            matching_key = game_key
            break

    if not matching_key:
        logger.warning(f"Could not find odds for: {title}")
        return None

    game_odds = all_odds[matching_key]
    home_team, away_team = matching_key.split(":")

    home_ml = game_odds[home_team]['money_line_odds']
    away_ml = game_odds[away_team]['money_line_odds']

    # Get token IDs for WebSocket price monitoring
    home_token_id = game_odds.get('home_token_id')
    away_token_id = game_odds.get('away_token_id')

    if home_ml is None or away_ml is None:
        logger.warning(f"Invalid odds for: {title}")
        return None

    # Get model prediction
    predictions = get_model_predictions([matching_key], {matching_key: game_odds})

    if matching_key not in predictions:
        logger.warning(f"No model prediction for: {title}")
        return None

    pred = predictions[matching_key]
    model_home_prob = pred['home_prob']
    model_away_prob = pred['away_prob']

    # Market probabilities (from Polymarket token prices)
    market_home_prob = american_odds_to_probability(home_ml)
    market_away_prob = american_odds_to_probability(away_ml)

    # Calculate edges
    home_edge = model_home_prob - market_home_prob
    away_edge = model_away_prob - market_away_prob

    # Determine best bet side (highest edge that passes minimum)
    bet_side = None
    bet_edge = 0
    model_prob = 0
    entry_price = 0

    if home_edge >= MIN_EDGE:
        bet_side = "home"
        bet_edge = home_edge
        model_prob = model_home_prob
        entry_price = market_home_prob
    if away_edge >= MIN_EDGE and away_edge > home_edge:
        bet_side = "away"
        bet_edge = away_edge
        model_prob = model_away_prob
        entry_price = market_away_prob

    # Log game info
    local_game_time = game_time + timedelta(hours=LOCAL_TIMEZONE_OFFSET)
    logger.info(f"  Game time: {local_game_time.strftime('%I:%M %p')} local")
    logger.info(f"  Market: Home {market_home_prob:.1%} | Away {market_away_prob:.1%}")
    logger.info(f"  Model:  Home {model_home_prob:.1%} | Away {model_away_prob:.1%}")
    logger.info(f"  Edge:   Home {home_edge:+.1%} | Away {away_edge:+.1%}")

    if not bet_side:
        logger.info(f"  >>> NO BET (no edge >= {MIN_EDGE:.0%})")
        return None

    if entry_price < MIN_ENTRY_PRICE:
        logger.info(f"  >>> NO BET (entry price ${entry_price:.3f} < ${MIN_ENTRY_PRICE})")
        return None

    # Determine which leg this bet belongs to
    is_favorite = model_prob >= FAV_MIN_CONF
    is_underdog = not is_favorite

    # LEG 2 filter: underdogs must have entry >= DOG_MIN_ENTRY
    if is_underdog and entry_price < DOG_MIN_ENTRY:
        logger.info(f"  >>> NO BET (underdog entry ${entry_price:.3f} < ${DOG_MIN_ENTRY:.2f} floor)")
        return None

    leg = "LEG1_FAV" if is_favorite else "LEG2_DOG"
    exit_strategy = "hold to resolution" if is_favorite else "ESPN WP exits"

    # Size the bet (Half-Kelly, capped at MAX_BET_PCT)
    entry_bankroll = load_bankroll()
    bet_amount = half_kelly_size(model_prob, entry_price, entry_bankroll)

    if bet_amount <= 0:
        logger.info(f"  >>> NO BET (bankroll depleted)")
        return None

    # Create position
    positions = load_positions()
    position_id = f"{matching_key}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    position = {
        "game_key": matching_key,
        "home_team": home_team,
        "away_team": away_team,
        "home_token_id": home_token_id,
        "away_token_id": away_token_id,
        "game_time": game_time.isoformat(),
        "bet_side": bet_side,
        "entry_price": round(entry_price, 4),
        "bet_amount": round(bet_amount, 2),
        "model_prob": round(model_prob, 4),
        "bet_edge": round(bet_edge, 4),
        "leg": leg,
        "is_favorite": is_favorite,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "entry_bankroll": round(entry_bankroll, 2),
        "status": "open",
        # Exit tracking
        "exit_time": None,
        "exit_type": None,
        "exit_price": None,
        "pnl": None,
        # ESPN WP snapshots (filled during monitoring for underdogs)
        "espn_event_id": None,
        "espn_q1_wp": None,
        "espn_q1_score": None,
        "espn_q2_wp": None,
        "espn_q2_score": None,
    }

    positions[position_id] = position
    save_positions(positions)

    logger.info(f"  >>> {leg} BET {bet_side.upper()} ${bet_amount:.2f} "
                f"(edge {bet_edge:+.1%}, conf {model_prob:.1%}, {exit_strategy})")

    log_trade({
        "type": "ENTRY",
        "time": datetime.now(timezone.utc).isoformat(),
        "position_id": position_id,
        "game": f"{away_team} @ {home_team}",
        "game_time": game_time.isoformat(),
        "bet_side": bet_side,
        "leg": leg,
        "entry_price": entry_price,
        "bet_amount": bet_amount,
        "model_prob": model_prob,
        "bet_edge": bet_edge,
    })

    # Send bet placement alert to Discord
    if alert_mgr:
        game = f"{away_team} @ {home_team}"
        alert_mgr.info(f"NEW BET: {game} - {leg} {bet_side.upper()}", {
            "size": f"${bet_amount:.2f} (Half-Kelly, cap {MAX_BET_PCT:.0%})",
            "edge": f"{bet_edge:+.1%}",
            "conf": f"{model_prob:.1%}",
            "exit": exit_strategy,
            "system": "NBA",
        })

    # Subscribe to WebSocket for real-time price monitoring
    if ws_monitor:
        token_id = home_token_id if bet_side == "home" else away_token_id
        if token_id:
            ws_monitor.subscribe([token_id])
            logger.info(f"  Subscribed to WebSocket for {bet_side} token")

    return position_id


def get_live_prices(home_team, away_team, logger, game_time=None, market_id=None):
    """Fetch live in-game prices from Polymarket for a specific game.

    Args:
        home_team: Full home team name
        away_team: Full away team name
        logger: Logger instance
        game_time: ISO format game time string (used to match correct event)
        market_id: Optional market ID for direct lookup

    Returns:
        (home_prob, away_prob, is_closed, market_id) or (None, None, None, None) if not found
    """
    try:
        # Query active markets - NOTE: Don't use tag_id as it excludes in-progress games
        # Use closed=false to get games that haven't resolved yet
        params = {
            "series_id": NBA_SERIES_ID,
            "active": "true",
            "closed": "false",
            "limit": 100
        }
        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        home_short = home_team.split()[-1]
        away_short = away_team.split()[-1]

        # Parse game_time for date matching
        game_date = None
        if game_time:
            try:
                from datetime import datetime
                if isinstance(game_time, str):
                    game_date = datetime.fromisoformat(game_time.replace('Z', '+00:00')).date()
            except:
                pass

        # Find matching events (may be multiple for same teams)
        matching_events = []
        for event in events:
            title = event.get("title", "")
            if home_short in title and away_short in title:
                matching_events.append(event)

        # If we have game_time, filter to event with matching end date
        if game_date and len(matching_events) > 1:
            for event in matching_events:
                end_date_str = event.get("endDate", "")
                if end_date_str:
                    try:
                        from datetime import datetime
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                        if end_date == game_date:
                            matching_events = [event]
                            break
                    except:
                        continue

        # Process the first (or only) matching event
        for event in matching_events[:1]:
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # Match full game moneyline only (exclude 1H, O/U, Spread)
                if " vs" in question.lower() and "O/U" not in question and "Spread" not in question and "1H" not in question:
                    # If we have a specific market_id, verify it matches
                    found_market_id = market.get("id")
                    if market_id and str(found_market_id) != str(market_id):
                        continue

                    is_closed = market.get("closed", False)
                    outcome_prices = market.get("outcomePrices", "")
                    outcomes = market.get("outcomes", "")

                    if isinstance(outcome_prices, str):
                        import json as json_module
                        try:
                            outcome_prices = json_module.loads(outcome_prices)
                        except:
                            continue

                    if isinstance(outcomes, str):
                        import json as json_module
                        try:
                            outcomes = json_module.loads(outcomes)
                        except:
                            outcomes = []

                    if len(outcome_prices) >= 2:
                        # Map prices to teams using outcomes array
                        # outcomes = ['Team1', 'Team2'], prices = [price1, price2]
                        home_prob = None
                        away_prob = None

                        # Match using both city and team name for robustness
                        # e.g., "Brooklyn Nets" → check "Nets" and "Brooklyn"
                        home_words = [w.lower() for w in home_team.split() if len(w) > 2]
                        away_words = [w.lower() for w in away_team.split() if len(w) > 2]

                        for i, outcome in enumerate(outcomes):
                            outcome_lower = outcome.lower()
                            if any(w in outcome_lower for w in home_words):
                                home_prob = float(outcome_prices[i])
                            elif any(w in outcome_lower for w in away_words):
                                away_prob = float(outcome_prices[i])

                        # Only use prices if BOTH teams were matched
                        if home_prob is None or away_prob is None:
                            logger.warning(f"Could not match outcomes {outcomes} to teams {home_team}/{away_team} - skipping")
                            continue

                        logger.debug(f"Live prices for {away_short}@{home_short}: away={away_prob:.3f}, home={home_prob:.3f}, closed={is_closed}, market_id={found_market_id}")
                        return home_prob, away_prob, is_closed, found_market_id
            break
    except Exception as e:
        logger.error(f"Error fetching live prices: {e}")

    return None, None, None, None


def check_market_resolved(game_key, home_team, away_team, logger, game_time=None):
    """Check if a market has resolved by querying the API for closed markets.

    Args:
        game_key: Game identifier string
        home_team: Full home team name
        away_team: Full away team name
        logger: Logger instance
        game_time: Optional ISO format game time for date matching

    Returns:
        (resolved, winner) - resolved is True if market is done, winner is 'home' or 'away' or None
    """
    try:
        # Query closed markets - resolved games are no longer "active"
        params = {
            "series_id": NBA_SERIES_ID,
            "closed": "true",
            "limit": 100,
            "order": "endDate",
            "ascending": "false"  # Most recent first
        }
        response = requests.get(f"{GAMMA_API_URL}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        home_short = home_team.split()[-1]
        away_short = away_team.split()[-1]
        logger.debug(f"Checking resolution for {away_short} @ {home_short} (API fallback)...")

        # Parse game_time for date matching
        game_date = None
        if game_time:
            try:
                if isinstance(game_time, str):
                    game_date = datetime.fromisoformat(game_time.replace('Z', '+00:00')).date()
            except:
                pass

        # Find matching events (filter by date if provided)
        matching_events = []
        for event in events:
            title = event.get("title", "")
            if home_short in title and away_short in title:
                matching_events.append(event)

        # If we have game_date, filter to event with matching end date
        if game_date and len(matching_events) > 1:
            for event in matching_events:
                end_date_str = event.get("endDate", "")
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                        if end_date == game_date:
                            matching_events = [event]
                            break
                    except:
                        continue

        for event in matching_events[:1]:
            title = event.get("title", "")
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # Match full game moneyline only (exclude 1H, O/U, Spread)
                if " vs" in question.lower() and "O/U" not in question and "Spread" not in question and "1H" not in question:
                    is_closed = market.get("closed", False)
                    if is_closed:
                        logger.info(f"Found closed market: {title}")
                        outcome_prices = market.get("outcomePrices", "")
                        outcomes = market.get("outcomes", "")

                        if isinstance(outcome_prices, str):
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except:
                                logger.warning(f"Failed to parse outcome prices: {outcome_prices}")
                                continue

                        if isinstance(outcomes, str):
                            try:
                                outcomes = json.loads(outcomes)
                            except:
                                outcomes = []

                        if len(outcome_prices) >= 2:
                            # Map prices to teams using outcomes array
                            home_prob = None
                            away_prob = None

                            for i, outcome in enumerate(outcomes):
                                if home_short in outcome:
                                    home_prob = float(outcome_prices[i])
                                elif away_short in outcome:
                                    away_prob = float(outcome_prices[i])

                            # Fallback if outcomes didn't match
                            if home_prob is None or away_prob is None:
                                away_prob = float(outcome_prices[0])
                                home_prob = float(outcome_prices[1])

                            logger.debug(f"Outcome prices - Away: {away_prob}, Home: {home_prob}")

                            # Resolved markets have ~1.0 for winner, ~0.0 for loser
                            if home_prob > 0.95:
                                logger.info(f"Market resolved: HOME won")
                                return True, 'home'
                            elif away_prob > 0.95:
                                logger.info(f"Market resolved: AWAY won")
                                return True, 'away'
                            else:
                                logger.warning(f"Market closed but no clear winner: home={home_prob}, away={away_prob}")
                    break
            break

    except Exception as e:
        logger.error(f"Error checking resolved markets: {e}")

    return False, None


def _calculate_v2_pnl(position, won):
    """Calculate P&L for V2 positions using Polymarket share model.

    Buy shares at entry_price, get $1 if win, $0 if loss.
    Win P&L = bet_amount * (1/entry_price - 1)
    Loss P&L = -bet_amount
    """
    bet_amount = position.get('bet_amount', 0)
    entry_price = position.get('entry_price', 0)

    if not bet_amount or not entry_price:
        return 0

    if won:
        return round(bet_amount * (1.0 / entry_price - 1), 2)
    else:
        return round(-bet_amount, 2)


def _calculate_v2_exit_pnl(position, exit_price):
    """Calculate P&L for early exit at a given Polymarket price.

    P&L = bet_amount * (exit_price / entry_price - 1)
    """
    bet_amount = position.get('bet_amount', 0)
    entry_price = position.get('entry_price', 0)

    if not bet_amount or not entry_price:
        return 0

    return round(bet_amount * (exit_price / entry_price - 1), 2)


def _record_resolution(position, position_id, winner, pnl, logger):
    """Record a position resolution and update drawdown manager.

    Args:
        position: The position dict (will be modified)
        position_id: Position identifier
        winner: 'home' or 'away'
        pnl: Dollar P&L amount
        logger: Logger instance
    """
    bet_side = position.get('bet_side')
    game = f"{position.get('away_team')} @ {position.get('home_team')}"
    won = (bet_side == winner) if bet_side else False

    # Update bankroll
    bankroll = load_bankroll()
    new_bankroll = bankroll + pnl
    save_bankroll(new_bankroll)

    # Send resolution alert (file + Discord)
    try:
        am = AlertManager(
            data_dir=DATA_DIR,
            enable_console=False,
            webhook_url=DISCORD_WEBHOOK_URL,
            webhook_platform="discord"
        )
        leg = position.get('leg', '?')
        am.resolution(game, won, pnl, {
            "bet_side": bet_side,
            "leg": leg,
            "winner": winner,
            "bankroll": new_bankroll,
        })
    except Exception as e:
        logger.warning(f"Alert error: {e}")

    # Record in drawdown manager (tracking only, no halt)
    try:
        dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
        dm.record_pnl(pnl, position_id, new_bankroll)
    except Exception as e:
        logger.error(f"Error recording to drawdown manager: {e}")


def monitor_positions(logger, ws_monitor=None):
    """Monitor open positions — V2 dual-leg strategy.

    Leg 1 (Favorites): Hold to binary resolution. No early exits.
    Leg 2 (Underdogs): ESPN WP exits at Q1 (leading) and halftime (SL/TP).
    Both legs: Detect market resolution via WebSocket/REST/API.

    Args:
        logger: Logger instance
        ws_monitor: Optional WebSocketPriceMonitor for real-time prices
    """
    positions = load_positions()
    open_positions = {k: v for k, v in positions.items() if v['status'] == 'open'}

    if not open_positions:
        return

    # Get current odds for active markets
    provider = PolymarketOddsProvider()
    current_odds = provider.get_odds()

    # Fetch ESPN win probabilities for underdog exit logic + logging
    espn_games = {}
    if _ESPN_AVAILABLE:
        try:
            espn = ESPNProvider('nba')
            espn.CACHE_TTL = 0  # Fresh data
            espn_games = espn.get_all_live_win_probabilities()
        except Exception as e:
            logger.debug(f"ESPN fetch skipped: {e}")

    bankroll = load_bankroll()

    for position_id, position in open_positions.items():
        game_key = position['game_key']
        home_team = position['home_team']
        away_team = position['away_team']
        bet_side = position.get('bet_side')
        entry_price = position.get('entry_price', 0)
        bet_amount = position.get('bet_amount', 0)
        is_favorite = position.get('is_favorite', position.get('model_prob', 0) >= FAV_MIN_CONF)
        leg = position.get('leg', 'LEG1_FAV' if is_favorite else 'LEG2_DOG')

        # ===== GET CURRENT POLYMARKET PRICE =====
        current_pm_price = None  # Price for our bet side

        if game_key not in current_odds:
            # Game has started - try WebSocket first, then REST API
            current_home_prob = None
            current_away_prob = None
            is_closed = False
            found_market_id = None
            stored_market_id = position.get('market_id')
            game_time = position.get('game_time')

            # Try WebSocket prices first (faster, real-time)
            ws_resolved = False
            ws_winner = None
            if ws_monitor and bet_side:
                token_id = position.get('home_token_id') if bet_side == 'home' else position.get('away_token_id')
                if token_id:
                    ws_price = ws_monitor.get_price(token_id)
                    if ws_price and ws_price.get('mid') is not None:
                        bet_price = ws_price.get('mid')

                        if bet_price >= 0.95 or bet_price <= 0.05:
                            logger.info(f"WebSocket prices at resolution level for {away_team}@{home_team} ({bet_side}={bet_price:.3f}) - verifying with API...")
                            api_resolved, api_winner = check_market_resolved(game_key, home_team, away_team, logger, game_time=game_time)

                            if api_resolved and api_winner:
                                ws_resolved = True
                                ws_winner = api_winner
                                logger.info(f"API confirmed resolution: {away_team}@{home_team} -> {api_winner.upper()} won")
                            else:
                                logger.debug(f"Market not officially closed yet, tracking prices")
                                current_pm_price = bet_price
                        else:
                            current_pm_price = bet_price
                            logger.debug(f"WebSocket prices: {away_team}@{home_team} -> {bet_side}={bet_price:.3f}")

            # Log price observation (WebSocket path)
            if current_pm_price is not None:
                if bet_side == 'home':
                    current_home_prob = current_pm_price
                    current_away_prob = 1 - current_pm_price
                else:
                    current_away_prob = current_pm_price
                    current_home_prob = 1 - current_pm_price
                game_date = position.get('game_time', '')[:10]
                get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="ws")

            # Log ESPN win probability
            if espn_games:
                espn_key = f"{home_team}:{away_team}"
                espn_game = espn_games.get(espn_key)
                if espn_game and espn_game.get('home_win_prob') is not None:
                    game_date = position.get('game_time', '')[:10]
                    get_espn_wp_logger().log(
                        game_date, home_team, away_team,
                        espn_game['home_win_prob'], espn_game['away_win_prob'],
                        home_score=espn_game.get('home_score'),
                        away_score=espn_game.get('away_score'),
                        period=espn_game.get('period'),
                        clock=espn_game.get('clock'),
                        source="scheduler",
                    )

            # If WebSocket + API confirmed resolution, process it
            if ws_resolved and ws_winner:
                won = (bet_side == ws_winner) if bet_side else False
                pnl = _calculate_v2_pnl(position, won)

                position['status'] = 'resolved'
                position['exit_time'] = datetime.now(timezone.utc).isoformat()
                position['exit_type'] = 'resolution'
                position['exit_price'] = 1.0 if won else 0.0
                position['winner'] = ws_winner
                position['won'] = won
                position['pnl'] = pnl

                if pnl != 0:
                    _record_resolution(position, position_id, ws_winner, pnl, logger)

                result_str = "WON" if won else "LOST"
                logger.info(f"RESOLVED (WS): {away_team} @ {home_team} - {leg} {ws_winner.upper()} won | {result_str} ${abs(pnl):.2f}")

                log_trade({
                    "type": "RESOLVED",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "position_id": position_id,
                    "game": f"{away_team} @ {home_team}",
                    "leg": leg,
                    "winner": ws_winner,
                    "bet_side": bet_side,
                    "won": won,
                    "pnl": pnl,
                    "source": "websocket",
                })
                continue

            # Fall back to REST API if WebSocket didn't have prices
            if current_pm_price is None:
                last_ws = position.get('last_ws_update')
                if last_ws:
                    try:
                        ws_age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_ws.replace('Z', '+00:00'))).total_seconds()
                        if ws_age < 300:
                            logger.debug(f"Skipping REST fallback for {away_team} @ {home_team} - WebSocket data is {ws_age:.0f}s old")
                            continue
                    except (ValueError, TypeError):
                        pass

                logger.info(f"Game in progress: {away_team} @ {home_team} - fetching live prices via REST...")
                current_home_prob, current_away_prob, is_closed, found_market_id = get_live_prices(
                    home_team, away_team, logger, game_time=game_time, market_id=stored_market_id
                )

                if found_market_id and not stored_market_id:
                    position['market_id'] = found_market_id

                if current_home_prob is not None:
                    current_pm_price = current_home_prob if bet_side == 'home' else current_away_prob
                    game_date = position.get('game_time', '')[:10]
                    get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="rest")
            else:
                is_closed = False

            if current_pm_price is None:
                # Couldn't get prices - check if resolved via API
                logger.info(f"Could not get live prices - checking if resolved via API...")
                resolved, winner = check_market_resolved(game_key, home_team, away_team, logger, game_time=game_time)

                if resolved and winner:
                    won = (bet_side == winner) if bet_side else False
                    pnl = _calculate_v2_pnl(position, won)

                    position['status'] = 'resolved'
                    position['exit_time'] = datetime.now(timezone.utc).isoformat()
                    position['exit_type'] = 'resolution'
                    position['exit_price'] = 1.0 if won else 0.0
                    position['winner'] = winner
                    position['won'] = won
                    position['pnl'] = pnl

                    if pnl != 0:
                        _record_resolution(position, position_id, winner, pnl, logger)

                    result_str = "WON" if won else "LOST"
                    logger.info(f"RESOLVED: {away_team} @ {home_team} - {leg} {winner.upper()} won | {result_str} ${abs(pnl):.2f}")

                    log_trade({
                        "type": "RESOLVED",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": position_id,
                        "game": f"{away_team} @ {home_team}",
                        "leg": leg,
                        "winner": winner,
                        "bet_side": bet_side,
                        "won": won,
                        "pnl": pnl,
                    })
                else:
                    logger.info(f"Market not yet resolved: {away_team} @ {home_team}")
                continue

            # Check if market is closed (game ended) with a clear winner
            if is_closed:
                if current_home_prob and current_home_prob > 0.95:
                    winner = 'home'
                elif current_away_prob and current_away_prob > 0.95:
                    winner = 'away'
                else:
                    winner = None

                if winner:
                    won = (bet_side == winner) if bet_side else False
                    pnl = _calculate_v2_pnl(position, won)

                    position['status'] = 'resolved'
                    position['exit_time'] = datetime.now(timezone.utc).isoformat()
                    position['exit_type'] = 'resolution'
                    position['exit_price'] = 1.0 if won else 0.0
                    position['winner'] = winner
                    position['won'] = won
                    position['pnl'] = pnl

                    if pnl != 0:
                        _record_resolution(position, position_id, winner, pnl, logger)

                    result_str = "WON" if won else "LOST"
                    logger.info(f"RESOLVED: {away_team} @ {home_team} - {leg} {winner.upper()} won | {result_str} ${abs(pnl):.2f}")

                    log_trade({
                        "type": "RESOLVED",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": position_id,
                        "game": f"{away_team} @ {home_team}",
                        "leg": leg,
                        "winner": winner,
                        "bet_side": bet_side,
                        "won": won,
                        "pnl": pnl,
                    })
                    continue

            # ===== LIVE POSITION — CHECK EXIT LOGIC =====
            logger.info(f"Live: {away_team} @ {home_team} | {leg} {bet_side.upper()} @ ${entry_price:.3f} -> ${current_pm_price:.3f}")

            # Track unrealized P&L
            unrealized_pnl = bet_amount * (current_pm_price / entry_price - 1) if entry_price > 0 else 0
            position['current_pm_price'] = current_pm_price
            position['unrealized_pnl'] = round(unrealized_pnl, 2)

            # FAVORITES: just hold, no exit logic
            if is_favorite:
                logger.info(f"  {leg} HOLD (favorite — holds to resolution) | unrealized ${unrealized_pnl:+.2f}")
                continue

            # ===== UNDERDOG ESPN WP EXIT LOGIC =====
            espn_key = f"{home_team}:{away_team}"
            espn_data = espn_games.get(espn_key) or espn_games.get(f"{away_team}:{home_team}")
            if not espn_data:
                for key, data in espn_games.items():
                    if home_team in key or away_team in key:
                        espn_data = data
                        break

            if not espn_data:
                logger.info(f"  {leg} HOLD (ESPN: no data, game may not have started)")
                continue

            period = espn_data.get('period', 0)
            home_wp = espn_data.get('home_win_prob', 0.5)
            home_score = espn_data.get('home_score', 0)
            away_score = espn_data.get('away_score', 0)

            bet_wp = home_wp if bet_side == 'home' else (1 - home_wp)
            score_diff = (home_score - away_score) * (1 if bet_side == 'home' else -1)

            logger.info(f"  ESPN: P{period} {away_score}-{home_score} | bet WP: {bet_wp:.1%} | score diff: {score_diff:+d}")

            # Store ESPN snapshots
            if period >= 1 and position.get('espn_q1_wp') is None:
                position['espn_q1_wp'] = round(bet_wp, 4)
                position['espn_q1_score'] = f"{home_score}-{away_score}"

            if period >= 2 and position.get('espn_q2_wp') is None:
                position['espn_q2_wp'] = round(bet_wp, 4)
                position['espn_q2_score'] = f"{home_score}-{away_score}"

            if espn_data.get('event_id') and not position.get('espn_event_id'):
                position['espn_event_id'] = espn_data['event_id']

            # EXIT LOGIC: ESPN WP as signal, Polymarket price as execution
            exit_type = None

            # Q1 underdog exit: underdog is leading after Q1
            if (Q1_UNDERDOG_EXIT and period >= 1
                    and score_diff > 0 and position.get('espn_q1_wp') is not None
                    and current_pm_price and current_pm_price > 0):
                exit_type = "q1_exit"
                logger.info(f"  >>> Q1 UNDERDOG EXIT: leading by {score_diff}, selling at PM ${current_pm_price:.3f}")

            # Halftime stop-loss
            elif period >= 2 and bet_wp < ESPN_SL_THRESH and current_pm_price and current_pm_price > 0:
                exit_type = "stop"
                logger.info(f"  >>> HALFTIME STOP-LOSS: WP {bet_wp:.1%} < {ESPN_SL_THRESH:.0%}, selling at PM ${current_pm_price:.3f}")

            # Halftime take-profit
            elif period >= 2 and bet_wp > ESPN_TP_THRESH and current_pm_price and current_pm_price > 0:
                exit_type = "take_profit"
                logger.info(f"  >>> HALFTIME TAKE-PROFIT: WP {bet_wp:.1%} > {ESPN_TP_THRESH:.0%}, selling at PM ${current_pm_price:.3f}")

            # Execute exit
            if exit_type and current_pm_price:
                pnl = _calculate_v2_exit_pnl(position, current_pm_price)
                bankroll += pnl

                position['status'] = 'closed'
                position['exit_time'] = datetime.now(timezone.utc).isoformat()
                position['exit_type'] = exit_type
                position['exit_price'] = round(current_pm_price, 4)
                position['pnl'] = pnl

                save_bankroll(bankroll)

                # Record in drawdown manager
                try:
                    dm = DrawdownManager(data_dir=DATA_DIR, starting_bankroll=STARTING_BANKROLL)
                    dm.record_pnl(pnl, position_id, bankroll)
                except Exception as e:
                    logger.error(f"Error recording to drawdown manager: {e}")

                # Discord alert
                try:
                    am = AlertManager(
                        data_dir=DATA_DIR, enable_console=False,
                        webhook_url=DISCORD_WEBHOOK_URL, webhook_platform="discord"
                    )
                    game = f"{away_team} @ {home_team}"
                    am.info(f"ESPN EXIT: {game} - {exit_type}", {
                        "leg": leg, "pnl": f"${pnl:+.2f}",
                        "exit_price": f"${current_pm_price:.3f}",
                        "bankroll": f"${bankroll:.2f}",
                    })
                except Exception as e:
                    logger.warning(f"Alert error: {e}")

                result_str = "PROFIT" if pnl >= 0 else "LOSS"
                logger.info(f"  EXIT: {result_str} ${pnl:+.2f} | Bankroll: ${bankroll:.2f}")

                log_trade({
                    "type": "EXIT",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "position_id": position_id,
                    "game": f"{away_team} @ {home_team}",
                    "leg": leg,
                    "exit_type": exit_type,
                    "exit_price": current_pm_price,
                    "espn_wp": bet_wp,
                    "pnl": pnl,
                    "bankroll": bankroll,
                })
            else:
                logger.info(f"  {leg} HOLD (no exit signal yet) | unrealized ${unrealized_pnl:+.2f}")

            continue

        # ===== PRE-GAME: Market still open, just track prices =====
        game_odds = current_odds[game_key]
        current_home_ml = game_odds[home_team]['money_line_odds']

        if current_home_ml is None:
            continue

        current_home_prob = american_odds_to_probability(current_home_ml)
        current_away_prob = 1 - current_home_prob
        current_pm_price = current_home_prob if bet_side == 'home' else current_away_prob

        # Log price observation (pre-game path)
        game_date = position.get('game_time', '')[:10]
        get_price_logger().log(game_date, home_team, away_team, current_home_prob, current_away_prob, source="pregame")

        unrealized_pnl = bet_amount * (current_pm_price / entry_price - 1) if entry_price > 0 else 0
        logger.debug(f"Pre-game: {away_team} @ {home_team} | {leg} @ ${current_pm_price:.3f} | unrealized ${unrealized_pnl:+.2f}")

    save_positions(positions)


def generate_daily_report(logger, report_date=None):
    """Generate end-of-day summary with P&L — V2 dual-leg format.

    Args:
        logger: Logger instance
        report_date: Date string (YYYY-MM-DD) for the report. Defaults to today.
    """
    positions = load_positions()
    if report_date is None:
        report_date = datetime.now().strftime('%Y-%m-%d')

    report_file = DATA_DIR / f"report_{report_date}.txt"

    # Filter positions for this date (by entry_time or position_id)
    date_positions = {
        k: v for k, v in positions.items()
        if report_date in v.get('entry_time', '') or report_date.replace('-', '') in k
    }

    bets = [p for p in date_positions.values() if p.get('bet_side')]
    resolved = [p for p in date_positions.values() if p.get('status') == 'resolved']
    closed = [p for p in date_positions.values() if p.get('status') == 'closed']
    all_finished = resolved + closed

    # Calculate P&L from all finished positions
    total_pnl = sum(p.get('pnl', 0) for p in all_finished)
    wins = [p for p in all_finished if (p.get('pnl') or 0) > 0]
    losses = [p for p in all_finished if (p.get('pnl') or 0) <= 0]

    current_bankroll = load_bankroll()

    # Leg breakdown
    fav_bets = [p for p in bets if p.get('is_favorite', False)]
    dog_bets = [p for p in bets if not p.get('is_favorite', False)]
    fav_finished = [p for p in all_finished if p.get('is_favorite', False)]
    dog_finished = [p for p in all_finished if not p.get('is_favorite', False)]

    report = []
    report.append("=" * 60)
    report.append(f"DAILY REPORT (V2 DUAL-LEG) - {report_date}")
    report.append("=" * 60)
    report.append(f"\nBets placed: {len(bets)} (favorites: {len(fav_bets)}, underdogs: {len(dog_bets)})")
    report.append(f"ESPN exits (underdogs): {len(closed)}")
    report.append(f"Game resolutions: {len(resolved)}")

    report.append(f"\n--- P&L SUMMARY ---")
    report.append(f"Wins: {len(wins)} | Losses: {len(losses)}")
    if all_finished:
        win_rate = len(wins) / len(all_finished) * 100
        report.append(f"Win Rate: {win_rate:.1f}%")
    report.append(f"Daily P&L: ${total_pnl:+.2f}")
    report.append(f"Current Bankroll: ${current_bankroll:.2f}")

    # Leg breakdown P&L
    if fav_finished:
        fav_pnl = sum(p.get('pnl', 0) for p in fav_finished)
        fav_wins = sum(1 for p in fav_finished if (p.get('pnl') or 0) > 0)
        report.append(f"  Leg 1 (Favorites): {len(fav_finished)} trades, "
                       f"{fav_wins}W/{len(fav_finished)-fav_wins}L, P&L ${fav_pnl:+.2f}")
    if dog_finished:
        dog_pnl = sum(p.get('pnl', 0) for p in dog_finished)
        dog_wins = sum(1 for p in dog_finished if (p.get('pnl') or 0) > 0)
        report.append(f"  Leg 2 (Underdogs): {len(dog_finished)} trades, "
                       f"{dog_wins}W/{len(dog_finished)-dog_wins}L, P&L ${dog_pnl:+.2f}")

    if closed:
        report.append("\n--- ESPN EXITS (UNDERDOGS) ---")
        for p in closed:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            exit_type = p.get('exit_type', '?')
            pnl = p.get('pnl', 0)
            exit_price = p.get('exit_price', 0)
            report.append(f"  {game}")
            report.append(f"    {side} | {exit_type} @ ${exit_price:.3f} | P&L: ${pnl:+.2f}")

    if resolved:
        report.append("\n--- GAME RESOLUTIONS ---")
        for p in resolved:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            leg = p.get('leg', '?')
            winner = p.get('winner', '').upper()
            result = "WIN" if p.get('won') else "LOSS"
            pnl = p.get('pnl', 0)
            report.append(f"  {game}")
            report.append(f"    {leg} {side} | Winner: {winner} | {result}: ${pnl:+.2f}")

    if bets:
        report.append("\n--- ALL BETS ---")
        for p in bets:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', '').upper()
            leg = p.get('leg', '?')
            edge = p.get('bet_edge', 0)
            conf = p.get('model_prob', 0)
            status = p.get('status', 'unknown')
            report.append(f"  {game}: {leg} {side} (edge: {edge:+.1%}, conf: {conf:.1%}) [{status}]")

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)

    with open(report_file, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved: {report_file}")
    logger.info(f"Daily P&L: ${total_pnl:+.2f} | Bankroll: ${current_bankroll:.2f}")
    print(report_text)


def run_scheduler():
    """Main scheduler loop - smart per-game timing."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("POLYMARKET V2 DUAL-LEG SCHEDULER")
    logger.info("=" * 60)
    logger.info(f"Strategy: Leg 1 (FAV >= {FAV_MIN_CONF:.0%}) hold | Leg 2 (DOG >= ${DOG_MIN_ENTRY:.2f}) ESPN exits")
    logger.info(f"Edge threshold: {MIN_EDGE:.0%} | Sizing: Half-Kelly (cap {MAX_BET_PCT:.0%})")
    logger.info(f"Bets placed: {MINUTES_BEFORE_GAME} minutes before each game")
    logger.info(f"Monitor interval: {MONITOR_INTERVAL} minutes")
    logger.info(f"Timezone offset: UTC{LOCAL_TIMEZONE_OFFSET:+d}")
    logger.info("")

    # Initialize AlertManager for Discord notifications
    alert_mgr = AlertManager(
        data_dir=DATA_DIR,
        enable_console=False,
        enable_file=True,
        webhook_url=DISCORD_WEBHOOK_URL,
        webhook_platform="discord",
    )

    # Initialize WebSocket monitor for real-time prices
    ws_monitor = WebSocketPriceMonitor(logger=logger)
    ws_started = ws_monitor.start()
    if ws_started:
        logger.info("WebSocket price monitor started")
    else:
        logger.warning("WebSocket monitor failed to start - falling back to REST API polling")
        ws_monitor = None

    # Subscribe to open positions' token IDs
    if ws_monitor:
        positions = load_positions()
        token_ids = []
        for pos in positions.values():
            if pos.get('status') == 'open' and pos.get('bet_side'):
                bet_side = pos.get('bet_side')
                token_id = pos.get('home_token_id') if bet_side == 'home' else pos.get('away_token_id')
                if token_id:
                    token_ids.append(token_id)
        if token_ids:
            ws_monitor.subscribe(token_ids)
            logger.info(f"Subscribed to {len(token_ids)} position tokens via WebSocket")

    # Send startup alert
    alert_mgr.info("NBA Paper Trader V2 (Dual-Leg) started", {"system": "NBA", "status": "online"})

    initialized_games = set()  # Track games we've already bet on
    last_monitor = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            today = now.date()

            # Get today's games with times
            games = get_todays_games_with_times()

            # Check if we should generate a report for any date with all positions resolved
            positions = load_positions()

            # Group positions by entry date
            positions_by_date = {}
            for pos_id, pos in positions.items():
                entry_time = pos.get('entry_time', '')
                if entry_time:
                    entry_date = entry_time[:10]  # YYYY-MM-DD
                    if entry_date not in positions_by_date:
                        positions_by_date[entry_date] = []
                    positions_by_date[entry_date].append(pos)

            # Check each date - generate report if all positions are closed/resolved and we haven't reported yet
            for date, date_positions in positions_by_date.items():
                report_file = DATA_DIR / f"report_{date}.txt"
                # Positions are done if they're resolved (game ended) or closed (early exit)
                all_done = all(p.get('status') in ['resolved', 'closed'] for p in date_positions)
                has_pnl = any(p.get('pnl') is not None for p in date_positions)

                if all_done and has_pnl and not report_file.exists():
                    logger.info(f"All positions closed for {date}. Generating report...")
                    generate_daily_report(logger, report_date=date)

            # Track open positions for logging
            open_count = len([p for p in positions.values() if p['status'] == 'open'])

            if games:
                logger.debug(f"Found {len(games)} upcoming games")

                for game in games:
                    title = game['title']
                    game_time = game['game_time']
                    bet_time = game['bet_time']

                    # Skip if already initialized
                    game_id = f"{title}_{game_time.date()}"
                    if game_id in initialized_games:
                        continue

                    # Check if it's time to bet (within 5 min of bet_time)
                    time_until_bet = (bet_time - now).total_seconds()

                    if -300 <= time_until_bet <= 300:  # Within 5 min window
                        logger.info("-" * 40)
                        logger.info(f"TIME TO BET: {title}")
                        logger.info(f"Game starts in {MINUTES_BEFORE_GAME} minutes")
                        logger.info("-" * 40)

                        result = init_single_game(game['event'], game_time, logger, alert_mgr, ws_monitor)
                        initialized_games.add(game_id)

                    elif time_until_bet > 0:
                        hours = int(time_until_bet // 3600)
                        mins = int((time_until_bet % 3600) // 60)
                        logger.debug(f"  {title}: bet in {hours}h {mins}m")

            # Monitor existing positions
            if last_monitor is None or (now - last_monitor).total_seconds() >= MONITOR_INTERVAL * 60:
                positions = load_positions()
                open_count = len([p for p in positions.values() if p['status'] == 'open'])

                if open_count > 0:
                    logger.info(f"Monitoring {open_count} open positions...")
                    monitor_positions(logger, ws_monitor=ws_monitor)
                    last_monitor = now

            # Sleep before next check
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("\nScheduler stopping...")
            if ws_monitor:
                ws_monitor.stop()
                logger.info("WebSocket monitor stopped")
            logger.info("Scheduler stopped")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(60)


def show_schedule():
    """Show today's game schedule and bet times."""
    logger = setup_logging()
    games = get_todays_games_with_times()

    print("=" * 60)
    print("TODAY'S GAME SCHEDULE")
    print("=" * 60)
    print(f"\nBets will be placed {MINUTES_BEFORE_GAME} minutes before each game\n")

    if not games:
        print("No upcoming games found.")
        return

    for game in games:
        title = game['title']
        game_time = game['game_time'] + timedelta(hours=LOCAL_TIMEZONE_OFFSET)
        bet_time = game['bet_time'] + timedelta(hours=LOCAL_TIMEZONE_OFFSET)

        print(f"{title}")
        print(f"  Game: {game_time.strftime('%I:%M %p')} local")
        print(f"  Bet:  {bet_time.strftime('%I:%M %p')} local")
        print()


def show_status():
    """Show current positions and bankroll status — V2 dual-leg format."""
    positions = load_positions()
    bankroll = load_bankroll()

    print("=" * 60)
    print("PAPER TRADING V2 — DUAL-LEG STATUS")
    print("=" * 60)

    print(f"\nStarting Bankroll: ${STARTING_BANKROLL:.2f}")
    print(f"Current Bankroll:  ${bankroll:.2f}")
    print(f"Return:            {(bankroll - STARTING_BANKROLL) / STARTING_BANKROLL:+.1%}")
    print(f"Total P&L:         ${bankroll - STARTING_BANKROLL:+.2f}")

    open_pos = [p for p in positions.values() if p['status'] == 'open']
    resolved_pos = [p for p in positions.values() if p['status'] == 'resolved']
    closed_pos = [p for p in positions.values() if p['status'] == 'closed']
    all_finished = resolved_pos + closed_pos

    fav_open = [p for p in open_pos if p.get('is_favorite', False)]
    dog_open = [p for p in open_pos if not p.get('is_favorite', False)]

    print(f"\nOpen positions: {len(open_pos)} (favorites: {len(fav_open)}, underdogs: {len(dog_open)})")
    print(f"ESPN exits (underdogs): {len(closed_pos)}")
    print(f"Resolved (game ended): {len(resolved_pos)}")

    if all_finished:
        wins = len([p for p in all_finished if (p.get('pnl') or 0) > 0])
        losses = len([p for p in all_finished if (p.get('pnl') or 0) <= 0])
        win_rate = wins / len(all_finished) * 100
        print(f"\nWin/Loss: {wins}W - {losses}L ({win_rate:.1f}%)")

        # Leg breakdown
        fav_finished = [p for p in all_finished if p.get('is_favorite', False)]
        dog_finished = [p for p in all_finished if not p.get('is_favorite', False)]
        if fav_finished:
            fav_pnl = sum(p.get('pnl', 0) for p in fav_finished)
            fav_wins = sum(1 for p in fav_finished if (p.get('pnl') or 0) > 0)
            print(f"  Leg 1 (Favorites): {len(fav_finished)} trades, "
                  f"{fav_wins}W/{len(fav_finished)-fav_wins}L, P&L ${fav_pnl:+.2f}")
        if dog_finished:
            dog_pnl = sum(p.get('pnl', 0) for p in dog_finished)
            dog_wins = sum(1 for p in dog_finished if (p.get('pnl') or 0) > 0)
            print(f"  Leg 2 (Underdogs): {len(dog_finished)} trades, "
                  f"{dog_wins}W/{len(dog_finished)-dog_wins}L, P&L ${dog_pnl:+.2f}")

    if open_pos:
        print("\n--- OPEN POSITIONS ---")
        for p in open_pos:
            game = f"{p['away_team']} @ {p['home_team']}"
            side = p.get('bet_side', 'none').upper()
            leg = p.get('leg', '?')
            edge = p.get('bet_edge', 0)
            conf = p.get('model_prob', 0)
            print(f"  {game}: {leg} {side} @ ${p.get('entry_price', 0):.3f} | "
                  f"${p.get('bet_amount', 0):.2f} | edge {edge:+.1%} conf {conf:.1%}")


def fix_exit_signals():
    """Legacy: no longer needed in V2. Kept for backwards compatibility."""
    print("V2 dual-leg strategy does not use exit_signal status. No action needed.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Polymarket Smart Scheduler')
    parser.add_argument('--schedule', action='store_true', help='Show today\'s schedule')
    parser.add_argument('--status', action='store_true', help='Show current status and bankroll')
    parser.add_argument('--test', action='store_true', help='Run one init cycle')
    parser.add_argument('--fix-exits', action='store_true', help='Fix exit_signal positions with realized P/L')
    parser.add_argument('--report', action='store_true', help='Generate daily report')
    args = parser.parse_args()

    if args.schedule:
        show_schedule()
    elif args.status:
        show_status()
    elif args.fix_exits:
        fix_exit_signals()
    elif args.report:
        logger = setup_logging()
        generate_daily_report(logger)
    elif args.test:
        logger = setup_logging()
        games = get_todays_games_with_times()
        if games:
            init_single_game(games[0]['event'], games[0]['game_time'], logger)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
