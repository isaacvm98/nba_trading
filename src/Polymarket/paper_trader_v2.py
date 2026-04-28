"""
Paper Trader V2 — Dual-Leg Backtest-Proven Strategy

Two-leg strategy from Stage 4 of empirical_test_report.md:
  Leg 1 (Favorites): edge >= 7%, model conf >= 60% -> hold to binary resolution
  Leg 2 (Underdogs): edge >= 7%, conf < 60%, entry >= $0.30 -> ESPN WP exits
    * Q1 underdog exit: if underdog is leading after Q1, sell at Polymarket price
    * Halftime stop-loss: ESPN WP < 25% -> sell at Polymarket price
    * Halftime take-profit: ESPN WP > 65% -> sell at Polymarket price

Sizing: Half-Kelly criterion — K = (p*b - q) / b / 2, capped at 10% per bet.
Backtest results (no look-ahead bias):
  Half-Kelly, 76 bets, 57.9% WR, +333.4% return, 35.7% MaxDD, Sharpe 5.78

Usage:
    python -m src.Polymarket.paper_trader_v2 --init        # Place bets pre-game
    python -m src.Polymarket.paper_trader_v2 --monitor     # Check live ESPN exits (underdogs only)
    python -m src.Polymarket.paper_trader_v2 --resolve     # Close finished games
    python -m src.Polymarket.paper_trader_v2 --status      # View positions/P&L
    python -m src.Polymarket.paper_trader_v2 --reset       # Fresh start
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.DataProviders.PolymarketOddsProvider import PolymarketOddsProvider
from src.DataProviders.ESPNProvider import ESPNProvider
from src.DataProviders.PriceHistoryProvider import PriceHistoryProvider

# Reuse model prediction pipeline from existing paper trader
from src.Polymarket.paper_trader import get_model_predictions, american_odds_to_probability


def fetch_market_context(token_id):
    """Fetch pre-game market-context features for the mispricing filter.

    Returns dict with open_price, price_max, price_min, price_range.
    All values are in probability units [0,1]. Returns Nones on failure.
    """
    empty = {"open_price": None, "price_max": None, "price_min": None, "price_range": None}
    if not token_id:
        return empty
    try:
        history = PriceHistoryProvider().get_price_history(token_id)
        if not history:
            return empty
        prices = [h["p"] for h in history if "p" in h]
        if not prices:
            return empty
        return {
            "open_price": round(prices[0], 4),
            "price_max": round(max(prices), 4),
            "price_min": round(min(prices), 4),
            "price_range": round(max(prices) - min(prices), 4),
        }
    except Exception as e:
        print(f"    Market context fetch failed: {e}")
        return empty

# =========================================================================
# STRATEGY PARAMETERS (from backtest Stage 4 — no look-ahead bias)
# =========================================================================
MIN_EDGE = 0.07          # Minimum model edge over market (7%)
MIN_ENTRY_PRICE = 0.05   # Skip penny tokens (no liquidity)
MAX_BET_PCT = 0.10       # Hard cap 10% of bankroll per bet (Half-Kelly can swing)

# POLYMARKET FEES (Sports category, taker only)
# fee = shares × FEE_RATE × price × (1 - price)
PM_FEE_RATE = 0.03       # 3% for Sports category


def pm_fee(bet_amount, price):
    """Polymarket taker fee: shares × 0.03 × p × (1-p). Collected on buy and sell."""
    if price <= 0 or price >= 1:
        return 0
    shares = bet_amount / price
    return shares * PM_FEE_RATE * price * (1 - price)

# LEG 1: Favorite bets — hold to binary resolution, no ESPN exits
FAV_MIN_CONF = 0.60      # Model probability >= 60% on bet side

# LEG 2: Underdog bets — ESPN WP exits for risk management
DOG_MIN_ENTRY = 0.30     # Min entry price for underdogs (pre-game volatility proxy)
ESPN_SL_THRESH = 0.25    # Stop-loss: ESPN WP < 25% at halftime
ESPN_TP_THRESH = 0.65    # Take-profit: ESPN WP > 65% at halftime
Q1_UNDERDOG_EXIT = True  # Sell underdog if leading after Q1

STARTING_BANKROLL = 1000.0

# =========================================================================
# FILE PATHS
# =========================================================================
DATA_DIR = Path("Data/paper_trading_v2")
POSITIONS_FILE = DATA_DIR / "positions.json"
BANKROLL_FILE = DATA_DIR / "bankroll.json"
TRADES_LOG = DATA_DIR / "trades.json"
DEPTH_LOG = DATA_DIR / "depth_log.jsonl"

CLOB_URL = "https://clob.polymarket.com"


def fetch_book_depth(token_id):
    """Fetch order book depth from Polymarket CLOB for a token.

    Returns dict with bids/asks arrays and summary stats, or None on error.
    """
    try:
        resp = requests.get(f"{CLOB_URL}/book", params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        book = resp.json()

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        # Compute depth at various price levels
        bid_sizes = [float(b.get("size", 0)) for b in bids]
        bid_prices = [float(b.get("price", 0)) for b in bids]
        ask_sizes = [float(a.get("size", 0)) for a in asks]
        ask_prices = [float(a.get("price", 0)) for a in asks]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        spread = best_ask - best_bid if best_bid and best_ask else None

        # Depth within 5% of best bid (what we'd hit selling)
        if best_bid > 0:
            depth_5pct = sum(s for p, s in zip(bid_prices, bid_sizes) if p >= best_bid * 0.95)
        else:
            depth_5pct = 0

        return {
            "token_id": token_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 4) if spread else None,
            "total_bid_depth": round(total_bid, 2),
            "total_ask_depth": round(total_ask, 2),
            "bid_depth_5pct": round(depth_5pct, 2),
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "top_5_bids": [{"price": p, "size": s} for p, s in zip(bid_prices[:5], bid_sizes[:5])],
        }
    except Exception as e:
        print(f"    Book depth error: {e}")
        return None


def log_depth(entry):
    """Append a depth snapshot to the JSONL log."""
    _ensure_dir()
    with open(DEPTH_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def half_kelly_size(model_prob, entry_price, bankroll):
    """Half-Kelly bet sizing: K = (p*b - q) / b / 2, capped at MAX_BET_PCT."""
    if entry_price <= 0 or entry_price >= 1:
        return 0
    b = (1.0 / entry_price) - 1.0  # payout odds
    q = 1.0 - model_prob
    k = (model_prob * b - q) / b
    if k <= 0:
        return 0
    half_k = k / 2.0
    return bankroll * min(half_k, MAX_BET_PCT)


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_bankroll():
    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE) as f:
            return json.load(f).get("bankroll", STARTING_BANKROLL)
    return STARTING_BANKROLL


def save_bankroll(bankroll):
    _ensure_dir()
    with open(BANKROLL_FILE, "w") as f:
        json.dump({"bankroll": round(bankroll, 2),
                    "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def load_positions():
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return {}


def save_positions(positions):
    _ensure_dir()
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def log_trade(trade):
    _ensure_dir()
    trades = []
    if TRADES_LOG.exists():
        with open(TRADES_LOG) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADES_LOG, "w") as f:
        json.dump(trades, f, indent=2, default=str)


# =========================================================================
# INIT: Place bets on upcoming games
# =========================================================================
def init_positions(force=False):
    """Scan for games and place bets meeting dual-leg strategy criteria."""
    print("=" * 70)
    print("PAPER TRADER V2 — DUAL-LEG INIT")
    print(f"  Leg 1 (FAV): edge >= {MIN_EDGE:.0%}, conf >= {FAV_MIN_CONF:.0%} -> hold")
    print(f"  Leg 2 (DOG): edge >= {MIN_EDGE:.0%}, conf < {FAV_MIN_CONF:.0%}, "
          f"entry >= ${DOG_MIN_ENTRY:.2f} -> ESPN exits")
    print(f"  Sizing: Half-Kelly (cap {MAX_BET_PCT:.0%})")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    bankroll = load_bankroll()
    print(f"\nBankroll: ${bankroll:.2f}")

    # Fetch today's Polymarket odds
    if force:
        print("Fetching all active Polymarket games...")
        provider = PolymarketOddsProvider()
    else:
        print("Fetching games starting within 60 minutes...")
        provider = PolymarketOddsProvider(minutes_before_game=60)

    odds = provider.get_odds()
    if not odds:
        print("No games found.")
        return

    print(f"Found {len(odds)} games\n")

    # Get model predictions
    predictions = get_model_predictions(list(odds.keys()), odds)
    print(f"Got predictions for {len(predictions)} games\n")

    positions = load_positions()
    bets_placed = 0
    fav_bets = 0
    dog_bets = 0

    for game_key, game_odds in odds.items():
        home_team, away_team = game_key.split(":")

        # Skip if already have position
        pos_id = f"{game_key}_{datetime.now().strftime('%Y%m%d')}"
        if pos_id in positions:
            continue

        if game_key not in predictions:
            continue

        pred = predictions[game_key]
        model_home = pred["home_prob"]
        model_away = pred["away_prob"]

        # Market probabilities (from Polymarket token prices)
        home_ml = game_odds[home_team]["money_line_odds"]
        away_ml = game_odds[away_team]["money_line_odds"]
        if home_ml is None or away_ml is None:
            continue

        market_home = american_odds_to_probability(home_ml)
        market_away = american_odds_to_probability(away_ml)

        # Calculate edges
        home_edge = model_home - market_home
        away_edge = model_away - market_away

        # Determine best bet side (highest edge that passes minimum)
        bet_side = None
        bet_edge = 0
        model_prob = 0
        entry_price = 0

        if home_edge >= MIN_EDGE:
            bet_side = "home"
            bet_edge = home_edge
            model_prob = model_home
            entry_price = market_home
        if away_edge >= MIN_EDGE and away_edge > home_edge:
            bet_side = "away"
            bet_edge = away_edge
            model_prob = model_away
            entry_price = market_away

        # Display game info
        print(f"{away_team} @ {home_team}")
        print(f"  Market: Home {market_home:.1%} | Away {market_away:.1%}")
        print(f"  Model:  Home {model_home:.1%} | Away {model_away:.1%}")
        print(f"  Edge:   Home {home_edge:+.1%} | Away {away_edge:+.1%}")

        if not bet_side:
            print(f"  -> SKIP (no edge >= {MIN_EDGE:.0%})")
            print()
            continue

        if entry_price < MIN_ENTRY_PRICE:
            print(f"  -> SKIP (entry price ${entry_price:.3f} < ${MIN_ENTRY_PRICE})")
            print()
            continue

        # Determine which leg this bet belongs to
        is_favorite = model_prob >= FAV_MIN_CONF
        is_underdog = not is_favorite

        # LEG 2 filter: underdogs must have entry >= DOG_MIN_ENTRY
        if is_underdog and entry_price < DOG_MIN_ENTRY:
            print(f"  -> SKIP (underdog entry ${entry_price:.3f} < ${DOG_MIN_ENTRY:.2f} floor)")
            print()
            continue

        # Size the bet (Half-Kelly)
        bet_amount = half_kelly_size(model_prob, entry_price, bankroll)
        if bet_amount <= 0:
            print(f"  -> SKIP (bankroll depleted)")
            print()
            continue

        leg = "LEG1_FAV" if is_favorite else "LEG2_DOG"
        exit_strategy = "hold to resolution" if is_favorite else "ESPN WP exits"

        # Get token IDs for price tracking
        home_token = game_odds.get("home_token_id")
        away_token = game_odds.get("away_token_id")

        position = {
            "game_key": game_key,
            "home_team": home_team,
            "away_team": away_team,
            "home_token_id": home_token,
            "away_token_id": away_token,
            "bet_side": bet_side,
            "entry_price": round(entry_price, 4),
            "bet_amount": round(bet_amount, 2),
            "model_prob": round(model_prob, 4),
            "bet_edge": round(bet_edge, 4),
            "leg": leg,
            "is_favorite": is_favorite,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_bankroll": round(bankroll, 2),
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

        positions[pos_id] = position
        bets_placed += 1
        if is_favorite:
            fav_bets += 1
        else:
            dog_bets += 1

        print(f"  -> {leg} BET {bet_side.upper()} ${bet_amount:.2f} "
              f"(edge {bet_edge:+.1%}, conf {model_prob:.1%}, {exit_strategy})")
        print()

        # Mispricing-filter features: fetch PM price history for bet-side token
        bet_token = home_token if bet_side == "home" else away_token
        mkt = fetch_market_context(bet_token)
        open_move = (entry_price - mkt["open_price"]) if mkt["open_price"] is not None else None

        # Rest differential: days rest of the bet side minus the opponent
        rest_home = pred.get("days_rest_home")
        rest_away = pred.get("days_rest_away")
        if rest_home is not None and rest_away is not None:
            rest_diff = (rest_home - rest_away) if bet_side == "home" else (rest_away - rest_home)
        else:
            rest_diff = None

        log_trade({
            "type": "ENTRY",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": pos_id,
            "game": f"{away_team} @ {home_team}",
            "bet_side": bet_side,
            "leg": leg,
            "entry_price": entry_price,
            "bet_amount": bet_amount,
            "model_prob": model_prob,
            "bet_edge": bet_edge,
            # Market-context features for the mispricing filter (Task 3)
            "open_price": mkt["open_price"],
            "price_max": mkt["price_max"],
            "price_min": mkt["price_min"],
            "price_range": mkt["price_range"],
            "open_move": round(open_move, 4) if open_move is not None else None,
            "days_rest_home": rest_home,
            "days_rest_away": rest_away,
            "rest_diff": rest_diff,
        })

    save_positions(positions)
    print(f"Bets placed: {bets_placed} (favorites: {fav_bets}, underdogs: {dog_bets})")
    print(f"Total open: {sum(1 for p in positions.values() if p['status'] == 'open')}")


# =========================================================================
# MONITOR: Check ESPN WP for live exit signals (underdogs only)
# =========================================================================
def monitor_positions():
    """Check live ESPN WP and execute exits on underdog positions only.

    Favorite positions (LEG1_FAV) are never exited early — they hold to resolution.
    Underdog positions (LEG2_DOG) get ESPN WP exit signals at Q1 and halftime.
    """
    print("=" * 70)
    print("PAPER TRADER V2 — MONITOR (ESPN exits for underdogs only)")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    positions = load_positions()
    open_pos = {k: v for k, v in positions.items() if v["status"] == "open"}

    if not open_pos:
        print("No open positions.")
        return

    fav_count = sum(1 for p in open_pos.values() if p.get("is_favorite", False))
    dog_count = len(open_pos) - fav_count
    print(f"Open positions: {len(open_pos)} (favorites: {fav_count}, underdogs: {dog_count})\n")

    # Fetch ESPN live data
    espn = ESPNProvider("nba")
    try:
        espn_games = espn.get_all_live_win_probabilities()
    except Exception as e:
        print(f"ESPN error: {e}")
        espn_games = {}

    # Fetch current Polymarket prices
    pm_provider = PolymarketOddsProvider()
    pm_odds = pm_provider.get_odds()

    bankroll = load_bankroll()
    exits = 0

    for pos_id, pos in open_pos.items():
        home = pos["home_team"]
        away = pos["away_team"]
        bet_side = pos["bet_side"]
        entry_price = pos["entry_price"]
        bet_amount = pos["bet_amount"]
        is_favorite = pos.get("is_favorite", pos.get("model_prob", 0) >= FAV_MIN_CONF)
        leg = pos.get("leg", "LEG1_FAV" if is_favorite else "LEG2_DOG")

        print(f"{away} @ {home} — {leg} {bet_side.upper()} @ ${entry_price:.3f}")

        # Favorites: just show status, no exit logic
        if is_favorite:
            print(f"  -> HOLD (favorite — holds to resolution)")
            print()
            continue

        # --- UNDERDOG EXIT LOGIC BELOW ---

        # Match to ESPN game
        espn_key = f"{home}:{away}"
        espn_data = espn_games.get(espn_key) or espn_games.get(f"{away}:{home}")

        if not espn_data:
            for key, data in espn_games.items():
                if home in key or away in key:
                    espn_data = data
                    break

        # Get current Polymarket price for the bet side
        pm_key = f"{home}:{away}"
        pm_game = pm_odds.get(pm_key)
        current_pm_price = None
        if pm_game and home in pm_game and away in pm_game:
            if bet_side == "home":
                home_ml = pm_game[home].get("money_line_odds")
                if home_ml is not None:
                    current_pm_price = american_odds_to_probability(home_ml)
            else:
                away_ml = pm_game[away].get("money_line_odds")
                if away_ml is not None:
                    current_pm_price = american_odds_to_probability(away_ml)

        if current_pm_price:
            entry_fee = pm_fee(bet_amount, entry_price)
            exit_fee = pm_fee(bet_amount, current_pm_price)
            unrealized_pnl = bet_amount * (current_pm_price / entry_price - 1) - entry_fee - exit_fee
            print(f"  Polymarket price: ${current_pm_price:.3f} (P&L: ${unrealized_pnl:+.2f}, fees: ${entry_fee + exit_fee:.2f})")

        if not espn_data:
            print(f"  ESPN: no data (game may not have started)")
            print()
            continue

        # Extract ESPN state
        period = espn_data.get("period", 0)
        home_wp = espn_data.get("home_win_prob", 0.5)
        home_score = espn_data.get("home_score", 0)
        away_score = espn_data.get("away_score", 0)

        bet_wp = home_wp if bet_side == "home" else (1 - home_wp)
        score_diff = (home_score - away_score) * (1 if bet_side == "home" else -1)

        print(f"  ESPN: P{period} {away_score}-{home_score} | "
              f"bet WP: {bet_wp:.1%} | score diff: {score_diff:+d}")

        # Store ESPN snapshots
        if period >= 1 and pos.get("espn_q1_wp") is None:
            pos["espn_q1_wp"] = round(bet_wp, 4)
            pos["espn_q1_score"] = f"{home_score}-{away_score}"

        if period >= 2 and pos.get("espn_q2_wp") is None:
            pos["espn_q2_wp"] = round(bet_wp, 4)
            pos["espn_q2_score"] = f"{home_score}-{away_score}"

        if espn_data.get("event_id") and not pos.get("espn_event_id"):
            pos["espn_event_id"] = espn_data["event_id"]

        # EXIT LOGIC: ESPN WP as signal, Polymarket price as execution
        exit_type = None
        exit_price = current_pm_price

        # Q1 underdog exit: underdog is leading after Q1
        if (Q1_UNDERDOG_EXIT and period >= 1
                and score_diff > 0 and pos.get("espn_q1_wp") is not None
                and exit_price and exit_price > 0):
            exit_type = "q1_exit"
            print(f"  >>> Q1 UNDERDOG EXIT: leading by {score_diff}, "
                  f"selling at Polymarket ${exit_price:.3f}")

        # Halftime stop-loss
        elif period >= 2 and bet_wp < ESPN_SL_THRESH and exit_price and exit_price > 0:
            exit_type = "stop"
            print(f"  >>> HALFTIME STOP-LOSS: WP {bet_wp:.1%} < {ESPN_SL_THRESH:.0%}, "
                  f"selling at Polymarket ${exit_price:.3f}")

        # Halftime take-profit
        elif period >= 2 and bet_wp > ESPN_TP_THRESH and exit_price and exit_price > 0:
            exit_type = "take_profit"
            print(f"  >>> HALFTIME TAKE-PROFIT: WP {bet_wp:.1%} > {ESPN_TP_THRESH:.0%}, "
                  f"selling at Polymarket ${exit_price:.3f}")

        # Execute exit
        if exit_type and exit_price:
            entry_fee = pm_fee(bet_amount, entry_price)
            exit_fee = pm_fee(bet_amount, exit_price)
            pnl = bet_amount * (exit_price / entry_price - 1) - entry_fee - exit_fee
            bankroll += pnl

            # Fetch and log order book depth at exit time
            depth_info = None
            bet_token_id = pos.get("home_token_id") if bet_side == "home" else pos.get("away_token_id")
            if bet_token_id:
                depth_info = fetch_book_depth(bet_token_id)
                if depth_info:
                    print(f"  Book depth: spread ${depth_info['spread']}, "
                          f"bid depth (5%) ${depth_info['bid_depth_5pct']:.0f}, "
                          f"total bids ${depth_info['total_bid_depth']:.0f} "
                          f"({depth_info['bid_levels']} levels)")
                    # Would our bet cause slippage?
                    if depth_info['bid_depth_5pct'] > 0:
                        fill_pct = bet_amount / depth_info['bid_depth_5pct'] * 100
                        print(f"  Slippage risk: bet ${bet_amount:.2f} vs "
                              f"${depth_info['bid_depth_5pct']:.0f} within 5% of best bid "
                              f"({fill_pct:.1f}% of available depth)")

                    log_depth({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "position_id": pos_id,
                        "game": f"{away} @ {home}",
                        "exit_type": exit_type,
                        "exit_price": exit_price,
                        "bet_amount": bet_amount,
                        **depth_info,
                    })

            pos["status"] = "closed"
            pos["exit_time"] = datetime.now(timezone.utc).isoformat()
            pos["exit_type"] = exit_type
            pos["exit_price"] = round(exit_price, 4)
            pos["pnl"] = round(pnl, 2)

            print(f"  P&L: ${pnl:+.2f} | Bankroll: ${bankroll:.2f}")
            exits += 1

            trade_entry = {
                "type": "EXIT",
                "time": datetime.now(timezone.utc).isoformat(),
                "position_id": pos_id,
                "game": f"{away} @ {home}",
                "leg": leg,
                "exit_type": exit_type,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            }
            if depth_info:
                trade_entry["book_spread"] = depth_info["spread"]
                trade_entry["bid_depth_5pct"] = depth_info["bid_depth_5pct"]
                trade_entry["total_bid_depth"] = depth_info["total_bid_depth"]
                trade_entry["bid_levels"] = depth_info["bid_levels"]
            log_trade(trade_entry)
        else:
            print(f"  -> HOLD (no exit signal yet)")

        print()

    save_positions(positions)
    save_bankroll(bankroll)
    print(f"Exits executed: {exits}")
    print(f"Bankroll: ${bankroll:.2f}")


# =========================================================================
# RESOLVE: Close positions where game is final
# =========================================================================
def resolve_positions():
    """Close positions where the game has ended (binary resolution)."""
    print("=" * 70)
    print("PAPER TRADER V2 — RESOLVE")
    print("=" * 70)

    positions = load_positions()
    open_pos = {k: v for k, v in positions.items() if v["status"] == "open"}

    if not open_pos:
        print("No open positions to resolve.")
        return

    # Fetch ESPN for final scores
    espn = ESPNProvider("nba")
    try:
        espn_games = espn.get_all_live_win_probabilities()
    except Exception as e:
        print(f"ESPN error: {e}")
        espn_games = {}

    bankroll = load_bankroll()
    resolved = 0

    for pos_id, pos in open_pos.items():
        home = pos["home_team"]
        away = pos["away_team"]
        bet_side = pos["bet_side"]
        entry_price = pos["entry_price"]
        bet_amount = pos["bet_amount"]
        leg = pos.get("leg", "?")

        # Find the game in ESPN data
        espn_data = None
        for key, data in espn_games.items():
            if home in key or away in key:
                espn_data = data
                break

        if not espn_data or not espn_data.get("is_final"):
            continue

        # Determine winner from final score
        home_score = espn_data.get("home_score", 0)
        away_score = espn_data.get("away_score", 0)

        if home_score == away_score:
            continue  # shouldn't happen in NBA

        home_won = home_score > away_score
        bet_won = (bet_side == "home" and home_won) or (bet_side == "away" and not home_won)

        entry_fee = pm_fee(bet_amount, entry_price)
        if bet_won:
            # Win: paid entry_price per share, get $1 per share, minus entry fee
            # No exit fee on resolution (price = 1.0, fee = 1*0*0.03 = 0)
            pnl = bet_amount * (1.0 / entry_price - 1) - entry_fee
        else:
            # Loss: lose full amount plus entry fee
            pnl = -bet_amount - entry_fee

        bankroll += pnl

        pos["status"] = "resolved"
        pos["exit_time"] = datetime.now(timezone.utc).isoformat()
        pos["exit_type"] = "resolution"
        pos["exit_price"] = 1.0 if bet_won else 0.0
        pos["pnl"] = round(pnl, 2)

        result = "WIN" if bet_won else "LOSS"
        print(f"{away} @ {home}: {away_score}-{home_score} | "
              f"{leg} {bet_side.upper()} | {result} | P&L: ${pnl:+.2f}")

        resolved += 1

        log_trade({
            "type": "RESOLVE",
            "time": datetime.now(timezone.utc).isoformat(),
            "position_id": pos_id,
            "game": f"{away} @ {home}",
            "leg": leg,
            "score": f"{away_score}-{home_score}",
            "bet_won": bet_won,
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    save_positions(positions)
    save_bankroll(bankroll)
    print(f"\nResolved: {resolved}")
    print(f"Bankroll: ${bankroll:.2f}")


# =========================================================================
# STATUS: Display current state
# =========================================================================
def show_status():
    """Display portfolio status with leg breakdown."""
    print("=" * 70)
    print("PAPER TRADER V2 — STATUS")
    print("=" * 70)

    bankroll = load_bankroll()
    positions = load_positions()

    open_pos = [p for p in positions.values() if p["status"] == "open"]
    closed_pos = [p for p in positions.values() if p["status"] in ("closed", "resolved")]

    total_pnl = sum(p.get("pnl", 0) for p in closed_pos)
    wins = sum(1 for p in closed_pos if (p.get("pnl") or 0) > 0)
    losses = sum(1 for p in closed_pos if (p.get("pnl") or 0) <= 0)
    win_rate = wins / len(closed_pos) if closed_pos else 0

    print(f"\nBankroll: ${bankroll:.2f} (started ${STARTING_BANKROLL:.2f})")
    print(f"Return: {(bankroll - STARTING_BANKROLL) / STARTING_BANKROLL:+.1%}")
    print(f"Total P&L: ${total_pnl:+.2f}")
    print(f"Trades: {len(closed_pos)} ({wins}W/{losses}L, {win_rate:.0%} WR)")

    # Leg breakdown
    fav_closed = [p for p in closed_pos if p.get("is_favorite", False)]
    dog_closed = [p for p in closed_pos if not p.get("is_favorite", False)]
    if fav_closed:
        fav_pnl = sum(p.get("pnl", 0) for p in fav_closed)
        fav_wins = sum(1 for p in fav_closed if (p.get("pnl") or 0) > 0)
        print(f"  Leg 1 (Favorites): {len(fav_closed)} trades, "
              f"{fav_wins}W/{len(fav_closed)-fav_wins}L, P&L ${fav_pnl:+.2f}")
    if dog_closed:
        dog_pnl = sum(p.get("pnl", 0) for p in dog_closed)
        dog_wins = sum(1 for p in dog_closed if (p.get("pnl") or 0) > 0)
        print(f"  Leg 2 (Underdogs): {len(dog_closed)} trades, "
              f"{dog_wins}W/{len(dog_closed)-dog_wins}L, P&L ${dog_pnl:+.2f}")

    # Exit type breakdown
    exit_types = {}
    for p in closed_pos:
        et = p.get("exit_type", "unknown")
        exit_types[et] = exit_types.get(et, 0) + 1
    if exit_types:
        print(f"Exit types: {', '.join(f'{k}:{v}' for k, v in sorted(exit_types.items()))}")

    if open_pos:
        print(f"\n--- Open Positions ({len(open_pos)}) ---")
        for p in open_pos:
            side = p["bet_side"].upper()
            leg = p.get("leg", "?")
            print(f"  {p['away_team']} @ {p['home_team']} | "
                  f"{leg} {side} @ ${p['entry_price']:.3f} | "
                  f"${p['bet_amount']:.2f} | "
                  f"edge {p['bet_edge']:+.1%} conf {p['model_prob']:.1%}")

    if closed_pos:
        print(f"\n--- Recent Trades (last 10) ---")
        for p in closed_pos[-10:]:
            result = "WIN" if (p.get("pnl") or 0) > 0 else "LOSS"
            leg = p.get("leg", "?")
            print(f"  {p['away_team']} @ {p['home_team']} | "
                  f"{leg} {p['bet_side'].upper()} | {p.get('exit_type', '?')} | "
                  f"{result} ${p.get('pnl', 0):+.2f}")


# =========================================================================
# RESET: Fresh start
# =========================================================================
def reset():
    """Reset all paper trading data for a fresh start."""
    _ensure_dir()
    save_bankroll(STARTING_BANKROLL)
    save_positions({})
    if TRADES_LOG.exists():
        TRADES_LOG.unlink()
    print(f"Reset complete. Bankroll: ${STARTING_BANKROLL:.2f}")
    print(f"Data dir: {DATA_DIR}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Paper Trader V2 — Dual-Leg Strategy")
    parser.add_argument("--init", action="store_true", help="Place bets on upcoming games")
    parser.add_argument("--init-force", action="store_true", help="Place bets on all active games")
    parser.add_argument("--monitor", action="store_true", help="Check ESPN WP exit signals (underdogs)")
    parser.add_argument("--resolve", action="store_true", help="Close finished games")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    parser.add_argument("--reset", action="store_true", help="Fresh start")
    args = parser.parse_args()

    if args.reset:
        reset()
    elif args.init or args.init_force:
        init_positions(force=args.init_force)
    elif args.monitor:
        monitor_positions()
    elif args.resolve:
        resolve_positions()
    elif args.status:
        show_status()
    else:
        # Default: run full cycle
        print("Running full cycle: init -> monitor -> resolve -> status\n")
        init_positions(force=True)
        print()
        monitor_positions()
        print()
        resolve_positions()
        print()
        show_status()


if __name__ == "__main__":
    main()
