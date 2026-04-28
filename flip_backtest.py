"""
Flip Strategy Backtest — Professor Yang's suggestion.

Three strategies compared using V2 dual-leg filters:
  A) Current: Sell dog at Q1 exit, done (baseline)
  B) Flip: Sell dog at Q1, immediately buy the favorite at Q1 price, hold to resolution
  C) Fav-only after Q1: Skip the dog entirely, just buy the favorite at Q1 discounted price

All use Half-Kelly sizing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"

STARTING_BANKROLL = 1000.0
MIN_EDGE = 0.07
MIN_ENTRY_PRICE = 0.05
DOG_MIN_ENTRY = 0.30
FAV_MIN_CONF = 0.60
MAX_BET_PCT = 0.10

Q1_WINDOW = (-45, -25)
HT_WINDOW = (-90, -65)


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

    bets["q1_score_diff"] = bets.apply(
        lambda r: (r["espn_q1_end_home_score"] - r["espn_q1_end_away_score"])
        * (1 if r["bet_side"] == "home" else -1)
        if pd.notna(r.get("espn_q1_end_home_score")) else None, axis=1
    )

    # Load Q1 prices for BOTH sides
    live_prices = _load_live_prices()
    pm_q1_bet, pm_q1_opp, pm_ht_bet = [], [], []
    for _, row in bets.iterrows():
        key = (row["game_date"] if isinstance(row["game_date"], str)
               else row["game_date"].strftime("%Y-%m-%d"),
               row["home_team"], row["away_team"])
        lp = live_prices.get(key, {})
        side = row["bet_side"]
        opp = "away" if side == "home" else "home"
        pm_q1_bet.append(lp.get(f"pm_q1_{side}", None))
        pm_q1_opp.append(lp.get(f"pm_q1_{opp}", None))
        pm_ht_bet.append(lp.get(f"pm_ht_{side}", None))
    bets["pm_q1_price"] = pm_q1_bet
    bets["pm_q1_opp_price"] = pm_q1_opp  # Favorite's Q1 price (discounted when dog leads)
    bets["pm_ht_price"] = pm_ht_bet

    # Did the opposite side (favorite) win the game?
    bets["opp_won"] = ~bets["bet_won"]

    return bets


def size_bet(model_prob, entry_price, bankroll):
    """Flat 2% sizing (matching report baseline)."""
    return bankroll * 0.02


def v2_filter(row):
    if row["model_prob"] >= FAV_MIN_CONF:
        return row["model_edge"] >= MIN_EDGE
    else:
        return row["model_edge"] >= MIN_EDGE and row["entry_price"] >= DOG_MIN_ENTRY


def is_q1_exit(row):
    """Does this underdog qualify for Q1 exit? (leading after Q1 with price data)"""
    return (row["model_prob"] < FAV_MIN_CONF
            and row["is_underdog"]
            and pd.notna(row.get("q1_score_diff"))
            and row["q1_score_diff"] > 0
            and pd.notna(row.get("pm_q1_price"))
            and row["pm_q1_price"] > 0)


def simulate(bets, name, strategy_fn):
    """Generic simulation. strategy_fn(row, bankroll) -> list of (pnl, description) trades."""
    bankroll = STARTING_BANKROLL
    history = []

    for _, row in bets.iterrows():
        if not v2_filter(row):
            continue

        trades = strategy_fn(row, bankroll)
        if not trades:
            continue

        for pnl, desc, bet_amt in trades:
            bankroll += pnl
            history.append({
                "game_date": row["game_date"],
                "game": f"{row['away_team']} @ {row['home_team']}",
                "bet_side": row["bet_side"],
                "desc": desc,
                "bet_amount": round(bet_amt, 2),
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            })

    if not history:
        return None

    total_pnl = bankroll - STARTING_BANKROLL
    total_trades = len(history)
    wins = sum(1 for h in history if h["pnl"] >= 0)
    losses = total_trades - wins

    running_peak = STARTING_BANKROLL
    max_dd = 0
    for h in history:
        running_peak = max(running_peak, h["bankroll"])
        dd = (running_peak - h["bankroll"]) / running_peak if running_peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe (group by date)
    daily = {}
    for h in history:
        d = h["game_date"]
        if d not in daily:
            daily[d] = {"start": h["bankroll"] - h["pnl"], "pnl": 0}
        daily[d]["pnl"] += h["pnl"]
    daily_returns = [d["pnl"] / d["start"] for d in daily.values() if d["start"] > 0]
    dr = np.array(daily_returns)
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

    return {
        "name": name,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_trades,
        "final_bankroll": round(bankroll, 2),
        "total_return": total_pnl / STARTING_BANKROLL,
        "max_drawdown": max_dd,
        "sharpe": round(sharpe, 2),
        "history": history,
    }


# ── Strategy A: Current (sell dog at Q1, done) ────────────────────────────

def strategy_current(row, bankroll):
    is_fav = row["model_prob"] >= FAV_MIN_CONF
    bet_amount = size_bet(row["model_prob"], row["entry_price"], bankroll)
    if bet_amount <= 0:
        return []

    # FAV: hold to resolution
    if is_fav:
        if row["bet_won"]:
            pnl = bet_amount * (1.0 / row["entry_price"] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, "FAV hold", bet_amount)]

    # DOG: Q1 exit if leading
    if is_q1_exit(row):
        pnl = bet_amount * (row["pm_q1_price"] / row["entry_price"] - 1)
        return [(pnl, "DOG q1_exit", bet_amount)]

    # DOG: no Q1 exit, hold to resolution
    if row["bet_won"]:
        pnl = bet_amount * (1.0 / row["entry_price"] - 1)
    else:
        pnl = -bet_amount
    return [(pnl, "DOG hold", bet_amount)]


# ── Strategy B: Flip (sell dog at Q1, buy fav at Q1 price, hold) ──────────

def strategy_flip(row, bankroll):
    is_fav = row["model_prob"] >= FAV_MIN_CONF
    bet_amount = size_bet(row["model_prob"], row["entry_price"], bankroll)
    if bet_amount <= 0:
        return []

    # FAV: hold to resolution (same as current)
    if is_fav:
        if row["bet_won"]:
            pnl = bet_amount * (1.0 / row["entry_price"] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, "FAV hold", bet_amount)]

    # DOG: Q1 exit + flip to favorite
    if is_q1_exit(row) and pd.notna(row.get("pm_q1_opp_price")) and row["pm_q1_opp_price"] > 0:
        # Leg 1: Sell dog
        dog_pnl = bet_amount * (row["pm_q1_price"] / row["entry_price"] - 1)
        bankroll_after = bankroll + dog_pnl

        # Leg 2: Buy favorite at Q1 discounted price, hold to resolution
        fav_q1_price = row["pm_q1_opp_price"]
        # Use model's implied probability for the favorite side for Kelly sizing
        opp_prob = 1.0 - row["model_prob"]  # rough: complement of dog's model prob
        fav_amount = size_bet(opp_prob, fav_q1_price, bankroll_after)
        if fav_amount <= 0:
            return [(dog_pnl, "DOG q1_exit (no flip)", bet_amount)]

        # Favorite wins = dog loses (opp_won)
        if row["opp_won"]:
            fav_pnl = fav_amount * (1.0 / fav_q1_price - 1)
        else:
            fav_pnl = -fav_amount

        return [
            (dog_pnl, "DOG q1_exit", bet_amount),
            (fav_pnl, "FLIP fav_hold", fav_amount),
        ]

    # DOG: no Q1 exit, hold to resolution
    if row["bet_won"]:
        pnl = bet_amount * (1.0 / row["entry_price"] - 1)
    else:
        pnl = -bet_amount
    return [(pnl, "DOG hold", bet_amount)]


# ── Strategy C: Only buy fav at Q1 price (skip dog entirely) ──────────────

def strategy_fav_q1_only(row, bankroll):
    is_fav = row["model_prob"] >= FAV_MIN_CONF
    bet_amount = size_bet(row["model_prob"], row["entry_price"], bankroll)
    if bet_amount <= 0:
        return []

    # FAV: hold to resolution (same as current)
    if is_fav:
        if row["bet_won"]:
            pnl = bet_amount * (1.0 / row["entry_price"] - 1)
        else:
            pnl = -bet_amount
        return [(pnl, "FAV hold", bet_amount)]

    # DOG spot: only enter if Q1 exit conditions met (dog was leading)
    # but instead of buying dog, buy the FAVORITE at discounted Q1 price
    if is_q1_exit(row) and pd.notna(row.get("pm_q1_opp_price")) and row["pm_q1_opp_price"] > 0:
        fav_q1_price = row["pm_q1_opp_price"]
        opp_prob = 1.0 - row["model_prob"]
        fav_amount = size_bet(opp_prob, fav_q1_price, bankroll)
        if fav_amount <= 0:
            return []

        if row["opp_won"]:
            fav_pnl = fav_amount * (1.0 / fav_q1_price - 1)
        else:
            fav_pnl = -fav_amount
        return [(fav_pnl, "FAV q1_buy", fav_amount)]

    # No Q1 opportunity: skip entirely (don't buy dog, don't hold)
    return []


def main():
    print("Loading data...")
    df = load_data()
    bets = prep_bets(df)
    print(f"Total bettable games: {len(bets)}")

    strategies = [
        ("A) Current (sell dog Q1)", strategy_current),
        ("B) Flip (sell dog + buy fav Q1)", strategy_flip),
        ("C) Fav-only at Q1 (skip dog)", strategy_fav_q1_only),
    ]

    results = []
    for name, fn in strategies:
        r = simulate(bets, name, fn)
        if r:
            results.append(r)

    # Print comparison
    print("\n" + "=" * 105)
    print("FLIP STRATEGY COMPARISON — Flat 2% sizing")
    print("=" * 105)

    hdr = f"{'Strategy':<38s} {'Trades':>6s} {'W/L':>8s} {'WR':>6s} {'Final $':>10s} {'Return':>9s} {'MaxDD':>7s} {'Sharpe':>7s}"
    print(hdr)
    print("-" * 105)
    for r in results:
        wl = f"{r['wins']}/{r['losses']}"
        print(f"{r['name']:<38s} {r['total_trades']:>6d} {wl:>8s} {r['win_rate']:>5.1%} "
              f"${r['final_bankroll']:>9.2f} {r['total_return']:>+8.1%} "
              f"{r['max_drawdown']:>6.1%} {r['sharpe']:>7.2f}")

    # Breakdown by trade type
    print("\n" + "=" * 105)
    print("BREAKDOWN BY TRADE TYPE")
    print("=" * 105)
    for r in results:
        print(f"\n--- {r['name']} ---")
        types = {}
        for h in r["history"]:
            d = h["desc"]
            if d not in types:
                types[d] = {"n": 0, "wins": 0, "pnl": 0, "sizes": []}
            types[d]["n"] += 1
            types[d]["wins"] += 1 if h["pnl"] >= 0 else 0
            types[d]["pnl"] += h["pnl"]
            types[d]["sizes"].append(h["bet_amount"])
        for desc, t in sorted(types.items()):
            wr = t["wins"] / t["n"] if t["n"] > 0 else 0
            avg_sz = np.mean(t["sizes"])
            print(f"  {desc:<25s} {t['n']:>3d} trades, {t['wins']}W/{t['n']-t['wins']}L "
                  f"({wr:>5.1%}), P&L: ${t['pnl']:>+9.2f}, avg size: ${avg_sz:.2f}")

    # Q1 flip detail
    print("\n" + "=" * 105)
    print("Q1 FLIP DETAIL (Strategy B)")
    print("=" * 105)
    if len(results) > 1:
        flip_r = results[1]
        flip_trades = [h for h in flip_r["history"] if "FLIP" in h["desc"] or "q1_exit" in h["desc"]]
        print(f"\n{'Date':<12s} {'Game':<45s} {'Type':<18s} {'Size':>8s} {'P&L':>9s} {'Bank':>10s}")
        print("-" * 105)
        for h in flip_trades[-30:]:  # last 30
            date = h["game_date"].strftime("%Y-%m-%d") if hasattr(h["game_date"], "strftime") else str(h["game_date"])[:10]
            print(f"{date:<12s} {h['game']:<45s} {h['desc']:<18s} ${h['bet_amount']:>7.2f} ${h['pnl']:>+8.2f} ${h['bankroll']:>9.2f}")


if __name__ == "__main__":
    main()
