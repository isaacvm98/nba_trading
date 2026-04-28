"""
Three-stage bankroll simulation comparing strategy evolution:
  Stage 1: Naive Kelly Criterion (hold to resolution)
  Stage 2: Underdog edge filter (model conf >= 60%, edge >= 7%)
  Stage 3: ESPN WP live exits (halftime SL/TP + Q1 underdog profit-taking)

Output: Data/backtest/analysis/simulation_results.csv
        Data/backtest/analysis/simulation_daily.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"
OUTPUT_DIR = BACKTEST_DIR / "analysis"

STARTING_BANKROLL = 1000.0

# POLYMARKET FEES (Sports category, taker only)
# fee = shares × FEE_RATE × price × (1 - price)
PM_FEE_RATE = 0.03


def pm_fee(bet_amount, price):
    """Polymarket taker fee: shares × 0.03 × p × (1-p)."""
    if price <= 0 or price >= 1:
        return 0
    shares = bet_amount / price
    return shares * PM_FEE_RATE * price * (1 - price)


def load_data():
    df = pd.read_csv(str(BACKTEST_DIR / "nba_backtest_dataset.csv"))
    espn = pd.read_csv(str(BACKTEST_DIR / "espn_wp_backtest.csv"))

    # Merge ESPN data
    df = df.merge(
        espn[["game_date", "home_team", "away_team",
              "espn_pregame_home", "espn_wp_min", "espn_wp_max",
              "espn_q1_end_wp", "espn_q1_end_home_score", "espn_q1_end_away_score",
              "espn_q2_end_wp", "espn_q2_end_home_score", "espn_q2_end_away_score",
              "espn_q3_end_wp"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    return df


MIN_ENTRY_PRICE = 0.05  # Skip penny tokens with no liquidity

# Real-time Polymarket price windows (minutes after tipoff)
Q1_WINDOW = (-45, -25)   # Q1 ends ~35 min real-time after tip
HT_WINDOW = (-90, -65)   # Halftime ~75-80 min after tip


def _load_live_prices():
    """Load Polymarket price history and extract Q1/halftime actual prices."""
    ph = pd.read_csv(str(BACKTEST_DIR / "nba_price_history.csv"))
    live = ph[ph["minutes_to_start"] < 0].copy()

    results = {}
    for (gd, ht, at), grp in live.groupby(["game_date", "home_team", "away_team"]):
        key = (gd, ht, at)
        row = {}

        # Q1 end: closest tick in window
        q1 = grp[(grp["minutes_to_start"] >= Q1_WINDOW[0]) & (grp["minutes_to_start"] <= Q1_WINDOW[1])]
        if len(q1) > 0:
            # Pick tick closest to -35 min (typical Q1 end)
            idx = (q1["minutes_to_start"] + 35).abs().idxmin()
            row["pm_q1_home"] = q1.loc[idx, "home_price"]
            row["pm_q1_away"] = q1.loc[idx, "away_price"]

        # Halftime: closest tick in window
        ht_ticks = grp[(grp["minutes_to_start"] >= HT_WINDOW[0]) & (grp["minutes_to_start"] <= HT_WINDOW[1])]
        if len(ht_ticks) > 0:
            idx = (ht_ticks["minutes_to_start"] + 75).abs().idxmin()
            row["pm_ht_home"] = ht_ticks.loc[idx, "home_price"]
            row["pm_ht_away"] = ht_ticks.loc[idx, "away_price"]

        results[key] = row

    return results


def prep_bets(df):
    """Prepare common bet fields for all games with Kelly > 0 and PM prices."""
    bets = df[
        (df["bet_side"] != "none")
        & df["home_win"].notna()
        & df["pm_pregame_home"].notna()
    ].copy()

    bets["bet_won"] = (
        ((bets["bet_side"] == "home") & (bets["home_win"] == 1))
        | ((bets["bet_side"] == "away") & (bets["home_win"] == 0))
    )
    bets["entry_price"] = bets.apply(
        lambda r: r["pm_pregame_home"] if r["bet_side"] == "home" else r["pm_pregame_away"], axis=1
    )
    bets["model_prob"] = bets.apply(
        lambda r: r["model_home_prob"] if r["bet_side"] == "home" else r["model_away_prob"], axis=1
    )
    bets["model_edge"] = bets.apply(
        lambda r: r["edge_home_pm"] if r["bet_side"] == "home" else r["edge_away_pm"], axis=1
    )
    # Filter out penny tokens (no liquidity, extreme ratios)
    bets = bets[bets["entry_price"] >= MIN_ENTRY_PRICE].copy()
    bets["is_underdog"] = bets["entry_price"] < 0.45

    # ESPN WP on bet side at each quarter (used for exit SIGNALS only)
    bets["espn_q1_bet_wp"] = bets.apply(
        lambda r: r["espn_q1_end_wp"] if r["bet_side"] == "home"
        else (1 - r["espn_q1_end_wp"]) if pd.notna(r.get("espn_q1_end_wp")) else None,
        axis=1,
    )
    bets["espn_q2_bet_wp"] = bets.apply(
        lambda r: r["espn_q2_end_wp"] if r["bet_side"] == "home"
        else (1 - r["espn_q2_end_wp"]) if pd.notna(r.get("espn_q2_end_wp")) else None,
        axis=1,
    )
    bets["q1_score_diff"] = bets.apply(
        lambda r: (r["espn_q1_end_home_score"] - r["espn_q1_end_away_score"])
        * (1 if r["bet_side"] == "home" else -1)
        if pd.notna(r.get("espn_q1_end_home_score")) else None,
        axis=1,
    )

    # Actual Polymarket prices at Q1 end and halftime (for exit EXECUTION)
    live_prices = _load_live_prices()
    pm_q1_prices = []
    pm_ht_prices = []
    for _, row in bets.iterrows():
        key = (row["game_date"] if isinstance(row["game_date"], str)
               else row["game_date"].strftime("%Y-%m-%d"),
               row["home_team"], row["away_team"])
        lp = live_prices.get(key, {})
        side = row["bet_side"]
        pm_q1_prices.append(lp.get(f"pm_q1_{side}", None))
        pm_ht_prices.append(lp.get(f"pm_ht_{side}", None))
    bets["pm_q1_price"] = pm_q1_prices
    bets["pm_ht_price"] = pm_ht_prices
    bets["espn_wp_range"] = bets["espn_wp_max"] - bets["espn_wp_min"]

    return bets


# =========================================================================
# SIMULATION ENGINE
# =========================================================================
def simulate(bets, strategy_name, size_fn, filter_fn=None, exit_fn=None):
    """Run a bankroll simulation.

    Args:
        bets: DataFrame with bet data
        strategy_name: Label for output
        size_fn: callable(row, bankroll) -> bet_amount
        filter_fn: callable(row) -> bool (True = take bet)
        exit_fn: callable(row) -> (exit_type, exit_price_ratio)
            exit_type: 'hold' | 'stop' | 'take_profit' | 'q1_exit'
            exit_price_ratio: fraction of entry price recovered (0-N)
    """
    bankroll = STARTING_BANKROLL
    peak = bankroll
    history = []
    daily = {}
    total_bets = 0
    wins = 0
    losses = 0
    stopped = 0
    tp_exits = 0
    q1_exits = 0

    for _, row in bets.iterrows():
        # Filter
        if filter_fn and not filter_fn(row):
            continue

        # Size
        bet_amount = size_fn(row, bankroll)
        if bet_amount <= 0 or bankroll <= 0:
            continue
        bet_amount = min(bet_amount, bankroll * 0.25)  # Hard cap 25%

        # Exit decision
        exit_type = "hold"
        if exit_fn:
            exit_type, exit_ratio = exit_fn(row)
        else:
            exit_ratio = None

        # P&L (including Polymarket taker fees)
        entry_fee = pm_fee(bet_amount, row["entry_price"])
        if exit_type in ("stop", "take_profit", "q1_exit"):
            # Sell at actual Polymarket price; exit_ratio = exit_price / entry_price
            exit_price = row["entry_price"] * exit_ratio
            exit_fee = pm_fee(bet_amount, exit_price)
            pnl = bet_amount * (exit_ratio - 1) - entry_fee - exit_fee
            if exit_type == "stop":
                stopped += 1
            elif exit_type == "take_profit":
                tp_exits += 1
            elif exit_type == "q1_exit":
                q1_exits += 1
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
        else:
            # Hold to resolution — no exit fee (price=1.0 or 0.0, fee=0)
            if row["bet_won"]:
                pnl = bet_amount * (1.0 / row["entry_price"] - 1) - entry_fee
                wins += 1
            else:
                pnl = -bet_amount - entry_fee
                losses += 1

        bankroll += pnl
        total_bets += 1
        peak = max(peak, bankroll)

        # Track daily
        date = row["game_date"]
        if date not in daily:
            daily[date] = {"start": bankroll - pnl, "end": bankroll, "bets": 0, "pnl": 0}
        daily[date]["end"] = bankroll
        daily[date]["bets"] += 1
        daily[date]["pnl"] += pnl

        history.append({
            "game_date": date,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bet_side": row["bet_side"],
            "entry_price": row["entry_price"],
            "bet_amount": bet_amount,
            "exit_type": exit_type,
            "bet_won": row["bet_won"],
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    # Compute metrics
    if total_bets == 0:
        return None

    final_bankroll = bankroll
    total_return = (final_bankroll - STARTING_BANKROLL) / STARTING_BANKROLL

    # Max drawdown from history
    running_peak = STARTING_BANKROLL
    max_dd = 0
    for h in history:
        running_peak = max(running_peak, h["bankroll"])
        dd = (running_peak - h["bankroll"]) / running_peak if running_peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Daily returns for Sharpe
    daily_returns = []
    for d in sorted(daily.keys()):
        dr = daily[d]["pnl"] / daily[d]["start"] if daily[d]["start"] > 0 else 0
        daily_returns.append(dr)

    daily_returns = np.array(daily_returns)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    return {
        "strategy": strategy_name,
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_bets,
        "stopped": stopped,
        "tp_exits": tp_exits,
        "q1_exits": q1_exits,
        "final_bankroll": round(final_bankroll, 2),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": round(sharpe, 2),
        "history": history,
        "daily": daily,
    }


def print_result(r):
    exits = ""
    if r["stopped"] or r["tp_exits"] or r["q1_exits"]:
        exits = f" | SL:{r['stopped']} TP:{r['tp_exits']} Q1:{r['q1_exits']}"
    print(
        f"  {r['strategy']:50s} | "
        f"{r['total_bets']:3d} bets W/L {r['wins']}/{r['losses']} ({r['win_rate']:.1%}) | "
        f"${STARTING_BANKROLL:.0f} -> ${r['final_bankroll']:.2f} ({r['total_return']:+.1%}) | "
        f"MaxDD {r['max_drawdown']:.1%} | Sharpe {r['sharpe_ratio']}"
        f"{exits}"
    )


# =========================================================================
# STAGE 1: Naive Kelly (hold to resolution)
# =========================================================================
def run_stage1(bets):
    print("\n" + "=" * 120)
    print("STAGE 1: NAIVE KELLY CRITERION (Hold to Resolution)")
    print("=" * 120)

    # 1a: Full Kelly, all bets
    r1a = simulate(
        bets, "Full Kelly, all bets",
        size_fn=lambda row, br: br * row["bet_kelly"] / 100,
    )
    print_result(r1a)

    # 1b: Half Kelly, all bets
    r1b = simulate(
        bets, "Half Kelly, all bets",
        size_fn=lambda row, br: br * row["bet_kelly"] / 200,
    )
    print_result(r1b)

    # 1c: Quarter Kelly, all bets
    r1c = simulate(
        bets, "Quarter Kelly, all bets",
        size_fn=lambda row, br: br * row["bet_kelly"] / 400,
    )
    print_result(r1c)

    # 1d: Flat 2%, all bets
    r1d = simulate(
        bets, "Flat 2%, all bets",
        size_fn=lambda row, br: br * 0.02,
    )
    print_result(r1d)

    # 1e: Flat 1%, all bets
    r1e = simulate(
        bets, "Flat 1%, all bets",
        size_fn=lambda row, br: br * 0.01,
    )
    print_result(r1e)

    return [r1a, r1b, r1c, r1d, r1e]


# =========================================================================
# STAGE 2: Underdog Edge Filters
# =========================================================================
def run_stage2(bets):
    print("\n" + "=" * 120)
    print("STAGE 2: EDGE & CONFIDENCE FILTERS (Hold to Resolution)")
    print("=" * 120)

    filters = [
        ("Flat 1%, edge >= 5%",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.05),
        ("Flat 1%, edge >= 7%",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07),
        ("Flat 1%, edge >= 10%",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.10),
        ("Flat 2%, edge >= 7%",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07),
        ("Flat 2%, edge >= 10%",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.10),
        ("Flat 1%, edge >= 7%, model conf >= 60%",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07 and row["model_prob"] >= 0.60),
        ("Flat 2%, edge >= 7%, model conf >= 60%",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07 and row["model_prob"] >= 0.60),
        ("Tiered Kelly, edge >= 5%",
         lambda row, br: br * (
             row.get("tiered_kelly_home_pm", 0) if row["bet_side"] == "home"
             else row.get("tiered_kelly_away_pm", 0)
         ) / 100 if pd.notna(row.get("tiered_kelly_home_pm")) else 0,
         lambda row: row["model_edge"] >= 0.05),
        ("Tiered Kelly, edge >= 7%, model conf >= 60%",
         lambda row, br: br * (
             row.get("tiered_kelly_home_pm", 0) if row["bet_side"] == "home"
             else row.get("tiered_kelly_away_pm", 0)
         ) / 100 if pd.notna(row.get("tiered_kelly_home_pm")) else 0,
         lambda row: row["model_edge"] >= 0.07 and row["model_prob"] >= 0.60),
    ]

    results = []
    for name, size_fn, filter_fn in filters:
        r = simulate(bets, name, size_fn=size_fn, filter_fn=filter_fn)
        if r:
            print_result(r)
            results.append(r)

    return results


# =========================================================================
# STAGE 3: ESPN WP Live Exits
# =========================================================================
def make_espn_exit_fn(sl_thresh=0.25, tp_thresh=0.65, q1_underdog_exit=True):
    """Create an exit function using ESPN WP as SIGNAL, actual PM prices for EXECUTION.

    ESPN WP tells us WHEN to exit (signal).
    Actual Polymarket price history tells us WHAT we'd sell at (execution).
    """
    def exit_fn(row):
        has_q1_signal = pd.notna(row.get("espn_q1_bet_wp"))
        has_q2_signal = pd.notna(row.get("espn_q2_bet_wp"))
        has_q1_price = pd.notna(row.get("pm_q1_price"))
        has_ht_price = pd.notna(row.get("pm_ht_price"))

        # Q1 underdog exit: ESPN WP says underdog is leading -> sell at actual PM Q1 price
        if q1_underdog_exit and row["is_underdog"] and has_q1_signal and has_q1_price:
            if row.get("q1_score_diff", 0) and row["q1_score_diff"] > 0:
                pm_price = row["pm_q1_price"]
                if pm_price > 0:
                    return ("q1_exit", pm_price / row["entry_price"])

        # Halftime stop-loss: ESPN WP below threshold -> sell at actual PM halftime price
        if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] < sl_thresh:
            pm_price = row["pm_ht_price"]
            if pm_price > 0:
                return ("stop", pm_price / row["entry_price"])

        # Halftime take-profit: ESPN WP above threshold -> sell at actual PM halftime price
        if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] > tp_thresh:
            pm_price = row["pm_ht_price"]
            if pm_price > 0:
                return ("take_profit", pm_price / row["entry_price"])

        return ("hold", None)

    return exit_fn


def run_stage3(bets):
    print("\n" + "=" * 120)
    print("STAGE 3: ESPN WP LIVE EXITS")
    print("=" * 120)

    configs = [
        # (name, size_fn, filter_fn, sl, tp, q1_exit)
        ("Flat 1%, edge>=7%, ESPN SL<25% TP>65%",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07,
         0.25, 0.65, False),

        ("Flat 1%, edge>=7%, ESPN SL<25% TP>65% + Q1 underdog exit",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07,
         0.25, 0.65, True),

        ("Flat 2%, edge>=7%, ESPN SL<25% TP>65% + Q1 underdog exit",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07,
         0.25, 0.65, True),

        ("Flat 1%, edge>=5%, ESPN SL<25% TP>65% + Q1 underdog exit",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.05,
         0.25, 0.65, True),

        ("Flat 2%, edge>=5%, ESPN SL<25% TP>65% + Q1 underdog exit",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.05,
         0.25, 0.65, True),

        ("Flat 1%, edge>=7%, conf>=60%, ESPN SL<25% TP>65% + Q1 exit",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07 and row["model_prob"] >= 0.60,
         0.25, 0.65, True),

        ("Flat 2%, edge>=7%, conf>=60%, ESPN SL<25% TP>65% + Q1 exit",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07 and row["model_prob"] >= 0.60,
         0.25, 0.65, True),

        ("Flat 1%, edge>=7%, ESPN SL<20% TP>70% + Q1 underdog exit",
         lambda row, br: br * 0.01,
         lambda row: row["model_edge"] >= 0.07,
         0.20, 0.70, True),

        ("Flat 2%, edge>=7%, ESPN SL<20% TP>70% + Q1 underdog exit",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07,
         0.20, 0.70, True),

        # High volatility proxy: entry price >= 0.30 (closer games = more volatile)
        ("Flat 2%, edge>=7%, entry>=0.30, ESPN SL<25% TP>65% + Q1",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07 and row["entry_price"] >= 0.30,
         0.25, 0.65, True),
    ]

    results = []
    for name, size_fn, filter_fn, sl, tp, q1 in configs:
        exit_fn = make_espn_exit_fn(sl_thresh=sl, tp_thresh=tp, q1_underdog_exit=q1)
        r = simulate(bets, name, size_fn=size_fn, filter_fn=filter_fn, exit_fn=exit_fn)
        if r:
            print_result(r)
            results.append(r)

    return results


# =========================================================================
# STAGE 4: Dual-Leg (Favorites hold, Underdogs get ESPN exits)
# =========================================================================
def make_dual_leg_exit_fn(sl_thresh=0.25, tp_thresh=0.65, q1_underdog_exit=True,
                          fav_conf=0.60):
    """Dual-leg exit: favorites hold to resolution, underdogs use ESPN WP exits."""
    def exit_fn(row):
        is_favorite = row["model_prob"] >= fav_conf

        # LEG 1: Favorites — hold to resolution (no early exit)
        if is_favorite:
            return ("hold", None)

        # LEG 2: Underdogs — ESPN WP exits
        has_q1_signal = pd.notna(row.get("espn_q1_bet_wp"))
        has_q2_signal = pd.notna(row.get("espn_q2_bet_wp"))
        has_q1_price = pd.notna(row.get("pm_q1_price"))
        has_ht_price = pd.notna(row.get("pm_ht_price"))

        # Q1 underdog exit: leading after Q1 -> sell at PM price
        if q1_underdog_exit and row["is_underdog"] and has_q1_signal and has_q1_price:
            if row.get("q1_score_diff", 0) and row["q1_score_diff"] > 0:
                pm_price = row["pm_q1_price"]
                if pm_price > 0:
                    return ("q1_exit", pm_price / row["entry_price"])

        # Halftime stop-loss
        if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] < sl_thresh:
            pm_price = row["pm_ht_price"]
            if pm_price > 0:
                return ("stop", pm_price / row["entry_price"])

        # Halftime take-profit
        if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] > tp_thresh:
            pm_price = row["pm_ht_price"]
            if pm_price > 0:
                return ("take_profit", pm_price / row["entry_price"])

        return ("hold", None)

    return exit_fn


def run_stage4(bets):
    print("\n" + "=" * 120)
    print("STAGE 4: DUAL-LEG (Favorites hold to resolution, Underdogs get ESPN exits)")
    print("  Leg 1 (favorite): model conf >= 60% + edge >= X% -> hold to resolution")
    print("  Leg 2 (underdog): conf < 60% + edge >= X% + entry >= Y -> ESPN WP exits")
    print("  Volatility proxy: entry_price (higher = closer game = more volatile)")
    print("=" * 120)

    # All filters use ONLY pre-game data:
    #   model_edge, model_prob, entry_price (PM pregame odds)
    # No espn_wp_range (that's look-ahead biased)

    configs = [
        # (name, size_fn, filter_fn, sl, tp, q1_exit, fav_conf)

        # Baseline: all edge>=7% underdogs (no vol filter)
        ("Flat 2%, edge>=7%, dual-leg (all dogs)",
         lambda row, br: br * 0.02,
         lambda row: row["model_edge"] >= 0.07,
         0.25, 0.65, True, 0.60),

        # Entry price as vol proxy: dogs entry >= $0.20
        ("Flat 2%, edge>=7%, dual-leg (dogs entry>=0.20)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.20)),
         0.25, 0.65, True, 0.60),

        # Dogs entry >= $0.25
        ("Flat 2%, edge>=7%, dual-leg (dogs entry>=0.25)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.25)),
         0.25, 0.65, True, 0.60),

        # Dogs entry >= $0.30
        ("Flat 2%, edge>=7%, dual-leg (dogs entry>=0.30)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.30)),
         0.25, 0.65, True, 0.60),

        # Dogs entry >= $0.35
        ("Flat 2%, edge>=7%, dual-leg (dogs entry>=0.35)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.35)),
         0.25, 0.65, True, 0.60),

        # Same sweep with flat 1%
        ("Flat 1%, edge>=7%, dual-leg (dogs entry>=0.25)",
         lambda row, br: br * 0.01,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.25)),
         0.25, 0.65, True, 0.60),

        ("Flat 1%, edge>=7%, dual-leg (dogs entry>=0.30)",
         lambda row, br: br * 0.01,
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.30)),
         0.25, 0.65, True, 0.60),

        # Edge >= 5% variants
        ("Flat 2%, edge>=5%, dual-leg (dogs entry>=0.25)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.05 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.25)),
         0.25, 0.65, True, 0.60),

        ("Flat 2%, edge>=5%, dual-leg (dogs entry>=0.30)",
         lambda row, br: br * 0.02,
         lambda row: (row["model_edge"] >= 0.05 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.30)),
         0.25, 0.65, True, 0.60),

        # Tiered Kelly favs + flat 2% dogs
        ("TK favs + 2% dogs, edge>=7%, dual-leg (dogs entry>=0.25)",
         lambda row, br: (
             br * (row.get("tiered_kelly_home_pm", 0) if row["bet_side"] == "home"
                   else row.get("tiered_kelly_away_pm", 0)) / 100
             if row["model_prob"] >= 0.60 and pd.notna(row.get("tiered_kelly_home_pm"))
             else br * 0.02
         ),
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.25)),
         0.25, 0.65, True, 0.60),

        ("TK favs + 2% dogs, edge>=7%, dual-leg (dogs entry>=0.30)",
         lambda row, br: (
             br * (row.get("tiered_kelly_home_pm", 0) if row["bet_side"] == "home"
                   else row.get("tiered_kelly_away_pm", 0)) / 100
             if row["model_prob"] >= 0.60 and pd.notna(row.get("tiered_kelly_home_pm"))
             else br * 0.02
         ),
         lambda row: (row["model_edge"] >= 0.07 and
                      (row["model_prob"] >= 0.60 or row["entry_price"] >= 0.30)),
         0.25, 0.65, True, 0.60),
    ]

    results = []
    for name, size_fn, filter_fn, sl, tp, q1, fav_conf in configs:
        exit_fn = make_dual_leg_exit_fn(sl_thresh=sl, tp_thresh=tp,
                                        q1_underdog_exit=q1, fav_conf=fav_conf)
        r = simulate(bets, name, size_fn=size_fn, filter_fn=filter_fn, exit_fn=exit_fn)
        if r:
            print_result(r)
            results.append(r)

    return results


# =========================================================================
# Main
# =========================================================================
def main():
    df = load_data()
    bets = prep_bets(df)
    print(f"Total bettable games: {len(bets)}")

    s1 = run_stage1(bets)
    s2 = run_stage2(bets)
    s3 = run_stage3(bets)
    s4 = run_stage4(bets)

    # Summary comparison
    print("\n" + "=" * 120)
    print("BEST FROM EACH STAGE")
    print("=" * 120)

    all_results = s1 + s2 + s3 + s4
    # Sort by total return
    all_results.sort(key=lambda r: r["total_return"], reverse=True)

    print(f"\n{'Strategy':55s} {'Bets':>5s} {'WR':>6s} {'Return':>10s} {'MaxDD':>7s} {'Sharpe':>7s}")
    print("-" * 95)
    for r in all_results[:15]:
        print(
            f"{r['strategy']:55s} {r['total_bets']:5d} {r['win_rate']:5.1%} "
            f"{r['total_return']:+9.1%} {r['max_drawdown']:6.1%} {r['sharpe_ratio']:7.2f}"
        )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "strategy": r["strategy"],
            "total_bets": r["total_bets"],
            "wins": r["wins"],
            "losses": r["losses"],
            "win_rate": round(r["win_rate"], 4),
            "stopped": r["stopped"],
            "tp_exits": r["tp_exits"],
            "q1_exits": r["q1_exits"],
            "final_bankroll": r["final_bankroll"],
            "total_return": round(r["total_return"], 4),
            "max_drawdown": round(r["max_drawdown"], 4),
            "sharpe_ratio": r["sharpe_ratio"],
        })
    pd.DataFrame(summary_rows).to_csv(str(OUTPUT_DIR / "simulation_results.csv"), index=False)

    # Save best strategy daily P&L
    best = all_results[0]
    daily_rows = []
    for date in sorted(best["daily"].keys()):
        d = best["daily"][date]
        daily_rows.append({
            "date": date,
            "bets": d["bets"],
            "pnl": round(d["pnl"], 2),
            "bankroll": round(d["end"], 2),
        })
    pd.DataFrame(daily_rows).to_csv(str(OUTPUT_DIR / "simulation_daily.csv"), index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
