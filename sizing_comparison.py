"""
Sizing strategy comparison for V2 Dual-Leg paper trader.

Compares 3 sizing approaches (+ current baseline) using the backtest dataset
with the same entry filters and exit logic as the live paper trader:
  - Leg 1 (FAV): conf >= 60%, edge >= 7%, hold to resolution
  - Leg 2 (DOG): conf < 60%, entry >= $0.30, edge >= 7%, ESPN WP exits

Sizing strategies:
  1. Flat 2% (current baseline)
  2. Half-Kelly (edge-derived Kelly / 2)
  3. Edge-scaled (1.5% base, scales with edge, capped at 4%)
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"

STARTING_BANKROLL = 1000.0


# ── Data loading (reused from backtest_simulation.py) ──────────────────────

def load_data():
    df = pd.read_csv(str(BACKTEST_DIR / "nba_backtest_dataset.csv"))
    espn = pd.read_csv(str(BACKTEST_DIR / "espn_wp_backtest.csv"))
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


MIN_ENTRY_PRICE = 0.05
Q1_WINDOW = (-45, -25)
HT_WINDOW = (-90, -65)


def _load_live_prices():
    ph = pd.read_csv(str(BACKTEST_DIR / "nba_price_history.csv"))
    live = ph[ph["minutes_to_start"] < 0].copy()
    results = {}
    for (gd, ht, at), grp in live.groupby(["game_date", "home_team", "away_team"]):
        key = (gd, ht, at)
        row = {}
        q1 = grp[(grp["minutes_to_start"] >= Q1_WINDOW[0]) & (grp["minutes_to_start"] <= Q1_WINDOW[1])]
        if len(q1) > 0:
            idx = (q1["minutes_to_start"] + 35).abs().idxmin()
            row["pm_q1_home"] = q1.loc[idx, "home_price"]
            row["pm_q1_away"] = q1.loc[idx, "away_price"]
        ht_ticks = grp[(grp["minutes_to_start"] >= HT_WINDOW[0]) & (grp["minutes_to_start"] <= HT_WINDOW[1])]
        if len(ht_ticks) > 0:
            idx = (ht_ticks["minutes_to_start"] + 75).abs().idxmin()
            row["pm_ht_home"] = ht_ticks.loc[idx, "home_price"]
            row["pm_ht_away"] = ht_ticks.loc[idx, "away_price"]
        results[key] = row
    return results


def prep_bets(df):
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
    bets = bets[bets["entry_price"] >= MIN_ENTRY_PRICE].copy()
    bets["is_underdog"] = bets["entry_price"] < 0.45

    bets["espn_q1_bet_wp"] = bets.apply(
        lambda r: r["espn_q1_end_wp"] if r["bet_side"] == "home"
        else (1 - r["espn_q1_end_wp"]) if pd.notna(r.get("espn_q1_end_wp")) else None, axis=1
    )
    bets["espn_q2_bet_wp"] = bets.apply(
        lambda r: r["espn_q2_end_wp"] if r["bet_side"] == "home"
        else (1 - r["espn_q2_end_wp"]) if pd.notna(r.get("espn_q2_end_wp")) else None, axis=1
    )
    bets["q1_score_diff"] = bets.apply(
        lambda r: (r["espn_q1_end_home_score"] - r["espn_q1_end_away_score"])
        * (1 if r["bet_side"] == "home" else -1)
        if pd.notna(r.get("espn_q1_end_home_score")) else None, axis=1
    )

    live_prices = _load_live_prices()
    pm_q1_prices, pm_ht_prices = [], []
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

    return bets


# ── Dual-leg exit (same as paper trader V2) ────────────────────────────────

def dual_leg_exit(row, fav_conf=0.60, sl=0.25, tp=0.65):
    is_favorite = row["model_prob"] >= fav_conf
    if is_favorite:
        return ("hold", None)

    has_q1_signal = pd.notna(row.get("espn_q1_bet_wp"))
    has_q2_signal = pd.notna(row.get("espn_q2_bet_wp"))
    has_q1_price = pd.notna(row.get("pm_q1_price"))
    has_ht_price = pd.notna(row.get("pm_ht_price"))

    if row["is_underdog"] and has_q1_signal and has_q1_price:
        if row.get("q1_score_diff", 0) and row["q1_score_diff"] > 0:
            pm_price = row["pm_q1_price"]
            if pm_price > 0:
                return ("q1_exit", pm_price / row["entry_price"])

    if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] < sl:
        pm_price = row["pm_ht_price"]
        if pm_price > 0:
            return ("stop", pm_price / row["entry_price"])

    if has_q2_signal and has_ht_price and row["espn_q2_bet_wp"] > tp:
        pm_price = row["pm_ht_price"]
        if pm_price > 0:
            return ("take_profit", pm_price / row["entry_price"])

    return ("hold", None)


# ── V2 entry filter (same as paper trader) ─────────────────────────────────

def v2_filter(row):
    """Leg 1: conf >= 60% & edge >= 7%.  Leg 2: conf < 60% & entry >= 0.30 & edge >= 7%."""
    if row["model_prob"] >= 0.60:
        return row["model_edge"] >= 0.07
    else:
        return row["model_edge"] >= 0.07 and row["entry_price"] >= 0.30


# ── Sizing functions ───────────────────────────────────────────────────────

def flat_2pct(row, bankroll):
    return bankroll * 0.02


def half_kelly(row, bankroll):
    """Kelly criterion / 2.  K = (p * b - q) / b  where b = (1/entry - 1)."""
    p = row["model_prob"]
    entry = row["entry_price"]
    if entry <= 0 or entry >= 1:
        return 0
    b = (1.0 / entry) - 1.0   # odds multiplier
    q = 1.0 - p
    k = (p * b - q) / b
    if k <= 0:
        return 0
    half_k = k / 2.0
    return bankroll * min(half_k, 0.10)  # hard cap 10% per bet


def edge_scaled(row, bankroll):
    """1.5% base + scales with edge, capped at 4%.
    At 7% edge  -> ~2.2%
    At 15% edge -> ~3.0%
    At 25% edge -> 4.0% (cap)
    """
    edge = row["model_edge"]
    pct = 0.015 + edge * 0.10
    pct = min(pct, 0.04)
    return bankroll * pct


# ── Simulation engine ──────────────────────────────────────────────────────

def simulate(bets, name, size_fn):
    bankroll = STARTING_BANKROLL
    history = []
    daily = {}
    wins = losses = stopped = tp_exits = q1_exits = 0

    for _, row in bets.iterrows():
        if not v2_filter(row):
            continue

        bet_amount = size_fn(row, bankroll)
        if bet_amount <= 0 or bankroll <= 0:
            continue
        bet_amount = min(bet_amount, bankroll * 0.25)

        exit_type, exit_ratio = dual_leg_exit(row)

        if exit_type in ("stop", "take_profit", "q1_exit"):
            pnl = bet_amount * (exit_ratio - 1)
            if exit_type == "stop":
                stopped += 1
            elif exit_type == "take_profit":
                tp_exits += 1
            else:
                q1_exits += 1
            (wins if pnl >= 0 else losses).__class__  # just for clarity
            if pnl >= 0:
                wins += 1
            else:
                losses += 1
        else:
            if row["bet_won"]:
                pnl = bet_amount * (1.0 / row["entry_price"] - 1)
                wins += 1
            else:
                pnl = -bet_amount
                losses += 1

        bankroll += pnl

        date = row["game_date"]
        if date not in daily:
            daily[date] = {"start": bankroll - pnl, "end": bankroll, "bets": 0,
                           "pnl": 0, "sizes": []}
        daily[date]["end"] = bankroll
        daily[date]["bets"] += 1
        daily[date]["pnl"] += pnl
        daily[date]["sizes"].append(bet_amount)

        history.append({
            "game_date": date,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "bet_side": row["bet_side"],
            "leg": "FAV" if row["model_prob"] >= 0.60 else "DOG",
            "entry_price": row["entry_price"],
            "model_edge": round(row["model_edge"], 4),
            "bet_amount": round(bet_amount, 2),
            "exit_type": exit_type,
            "bet_won": row["bet_won"],
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    if not history:
        return None

    total_bets = wins + losses
    total_return = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL

    # Max drawdown
    running_peak = STARTING_BANKROLL
    max_dd = 0
    for h in history:
        running_peak = max(running_peak, h["bankroll"])
        dd = (running_peak - h["bankroll"]) / running_peak if running_peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe
    daily_returns = []
    for d in sorted(daily.keys()):
        dr = daily[d]["pnl"] / daily[d]["start"] if daily[d]["start"] > 0 else 0
        daily_returns.append(dr)
    daily_returns = np.array(daily_returns)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Avg bet size
    all_sizes = [h["bet_amount"] for h in history]
    avg_size = np.mean(all_sizes)
    min_size = np.min(all_sizes)
    max_size = np.max(all_sizes)

    return {
        "strategy": name,
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_bets,
        "stopped": stopped,
        "tp_exits": tp_exits,
        "q1_exits": q1_exits,
        "final_bankroll": round(bankroll, 2),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": round(sharpe, 2),
        "avg_bet": round(avg_size, 2),
        "min_bet": round(min_size, 2),
        "max_bet": round(max_size, 2),
        "history": history,
        "daily": daily,
    }


# ── Output ─────────────────────────────────────────────────────────────────

def print_comparison(results):
    print("\n" + "=" * 100)
    print("SIZING STRATEGY COMPARISON — V2 Dual-Leg (same filters & exits)")
    print("=" * 100)

    hdr = (f"{'Strategy':<30s} {'Bets':>5s} {'W/L':>8s} {'WR':>6s} "
           f"{'Final $':>10s} {'Return':>8s} {'MaxDD':>7s} {'Sharpe':>7s} "
           f"{'AvgBet':>8s} {'MinBet':>8s} {'MaxBet':>8s}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        wl = f"{r['wins']}/{r['losses']}"
        exits = f"  (SL:{r['stopped']} TP:{r['tp_exits']} Q1:{r['q1_exits']})"
        print(
            f"{r['strategy']:<30s} {r['total_bets']:5d} {wl:>8s} {r['win_rate']:5.1%} "
            f"${r['final_bankroll']:>8.2f} {r['total_return']:>+7.1%} "
            f"{r['max_drawdown']:>6.1%} {r['sharpe_ratio']:>7.2f} "
            f"${r['avg_bet']:>7.2f} ${r['min_bet']:>7.2f} ${r['max_bet']:>7.2f}"
        )
        print(f"  {exits}")

    # Monthly breakdown
    print("\n" + "=" * 100)
    print("MONTHLY BREAKDOWN")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r['strategy']} ---")
        monthly = {}
        for h in r["history"]:
            month = h["game_date"].strftime("%Y-%m") if hasattr(h["game_date"], "strftime") else str(h["game_date"])[:7]
            if month not in monthly:
                monthly[month] = {"bets": 0, "wins": 0, "pnl": 0}
            monthly[month]["bets"] += 1
            monthly[month]["wins"] += 1 if h["pnl"] >= 0 else 0
            monthly[month]["pnl"] += h["pnl"]

        print(f"  {'Month':<10s} {'Bets':>5s} {'WR':>6s} {'P&L':>10s}")
        for month in sorted(monthly.keys()):
            m = monthly[month]
            wr = m["wins"] / m["bets"] if m["bets"] > 0 else 0
            print(f"  {month:<10s} {m['bets']:>5d} {wr:>5.1%} ${m['pnl']:>+9.2f}")

    # Leg breakdown
    print("\n" + "=" * 100)
    print("LEG BREAKDOWN (FAV vs DOG)")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r['strategy']} ---")
        for leg in ["FAV", "DOG"]:
            leg_trades = [h for h in r["history"] if h["leg"] == leg]
            if not leg_trades:
                continue
            leg_wins = sum(1 for h in leg_trades if h["pnl"] >= 0)
            leg_pnl = sum(h["pnl"] for h in leg_trades)
            leg_avg = np.mean([h["bet_amount"] for h in leg_trades])
            print(f"  {leg}: {len(leg_trades)} bets, {leg_wins}W/{len(leg_trades)-leg_wins}L "
                  f"({leg_wins/len(leg_trades):.1%}), P&L: ${leg_pnl:+.2f}, avg size: ${leg_avg:.2f}")


def main():
    print("Loading data...")
    df = load_data()
    bets = prep_bets(df)
    print(f"Total bettable games: {len(bets)}")

    strategies = [
        ("1. Flat 2% (current)", flat_2pct),
        ("2. Half-Kelly", half_kelly),
        ("3. Edge-Scaled (1.5-4%)", edge_scaled),
    ]

    results = []
    for name, size_fn in strategies:
        r = simulate(bets, name, size_fn)
        if r:
            results.append(r)

    print_comparison(results)

    # Save detailed trade logs
    output_dir = BACKTEST_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        trades_df = pd.DataFrame(r["history"])
        safe_name = r["strategy"].replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        trades_df.to_csv(str(output_dir / f"sizing_{safe_name}.csv"), index=False)

    print(f"\nTrade logs saved to {output_dir}")


if __name__ == "__main__":
    main()
