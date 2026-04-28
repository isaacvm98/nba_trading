"""
NBA Backtest Analysis — Three-part study:
  1. Where does the model lose? (accuracy breakdown)
  2. Entry point backtesting (using tick-level price history)
  3. Bankroll simulation (Kelly & tiered-Kelly P&L)

Usage:
    python backtest_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKTEST_DIR = BASE_DIR / "Data" / "backtest"
DATASET_CSV = BACKTEST_DIR / "nba_backtest_dataset.csv"
PRICE_HISTORY_CSV = BACKTEST_DIR / "nba_price_history.csv"
OUTPUT_DIR = BACKTEST_DIR / "analysis"


def load_data():
    df = pd.read_csv(str(DATASET_CSV))
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_price_history():
    ph = pd.read_csv(str(PRICE_HISTORY_CSV))
    ph["game_date"] = pd.to_datetime(ph["game_date"])
    return ph


# =========================================================================
# PART 1: Where does the model lose?
# =========================================================================
def analyze_model_accuracy(df):
    print("\n" + "=" * 70)
    print("PART 1: MODEL ACCURACY BREAKDOWN")
    print("=" * 70)

    has_pred = df[df["model_home_prob"].notna() & df["home_win"].notna()].copy()
    print(f"\nGames with predictions & results: {len(has_pred)}")

    overall_acc = has_pred["model_correct"].mean()
    print(f"Overall accuracy: {overall_acc:.1%} ({int(has_pred['model_correct'].sum())}/{len(has_pred)})")

    # --- By confidence bucket ---
    has_pred["confidence"] = has_pred.apply(
        lambda r: max(r["model_home_prob"], r["model_away_prob"]), axis=1
    )
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]
    has_pred["conf_bucket"] = pd.cut(has_pred["confidence"], bins=bins, labels=labels, right=False)

    print("\n--- Accuracy by Model Confidence ---")
    conf_stats = has_pred.groupby("conf_bucket", observed=True).agg(
        games=("model_correct", "size"),
        wins=("model_correct", "sum"),
        accuracy=("model_correct", "mean"),
    )
    conf_stats["accuracy"] = conf_stats["accuracy"].map("{:.1%}".format)
    print(conf_stats.to_string())

    # --- Favorites vs Underdogs (per Polymarket) ---
    has_pm = has_pred[has_pred["pm_pregame_home"].notna()].copy()
    if len(has_pm) > 0:
        has_pm["model_bet_type"] = has_pm.apply(
            lambda r: "favorite" if (
                (r["model_predicted_winner"] == "home" and r["pm_pregame_home"] >= 0.5)
                or (r["model_predicted_winner"] == "away" and r["pm_pregame_away"] >= 0.5)
            ) else "underdog",
            axis=1,
        )
        print("\n--- Accuracy: Favorite vs Underdog picks ---")
        for bt in ["favorite", "underdog"]:
            sub = has_pm[has_pm["model_bet_type"] == bt]
            if len(sub) > 0:
                acc = sub["model_correct"].mean()
                print(f"  {bt.capitalize()}: {acc:.1%} ({int(sub['model_correct'].sum())}/{len(sub)})")

    # --- By month ---
    has_pred["month"] = has_pred["game_date"].dt.to_period("M")
    print("\n--- Accuracy by Month ---")
    month_stats = has_pred.groupby("month").agg(
        games=("model_correct", "size"),
        wins=("model_correct", "sum"),
        accuracy=("model_correct", "mean"),
    )
    month_stats["accuracy"] = month_stats["accuracy"].map("{:.1%}".format)
    print(month_stats.to_string())

    # --- By edge size (model edge vs market) ---
    has_edge = has_pred[has_pred["edge_home_pm"].notna()].copy()
    if len(has_edge) > 0:
        # Use the edge on the side the model picks
        has_edge["model_edge"] = has_edge.apply(
            lambda r: r["edge_home_pm"] if r["model_predicted_winner"] == "home" else r["edge_away_pm"],
            axis=1,
        )
        edge_bins = [-1, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 1.0]
        edge_labels = ["<-10%", "-10 to -5%", "-5 to 0%", "0 to 5%", "5 to 10%", "10 to 15%", "15%+"]
        has_edge["edge_bucket"] = pd.cut(has_edge["model_edge"], bins=edge_bins, labels=edge_labels, right=False)

        print("\n--- Accuracy by Model Edge vs Polymarket ---")
        edge_stats = has_edge.groupby("edge_bucket", observed=True).agg(
            games=("model_correct", "size"),
            wins=("model_correct", "sum"),
            accuracy=("model_correct", "mean"),
        )
        edge_stats["accuracy"] = edge_stats["accuracy"].map("{:.1%}".format)
        print(edge_stats.to_string())

    # --- Home vs Away ---
    print("\n--- Accuracy by Predicted Side ---")
    for side in ["home", "away"]:
        sub = has_pred[has_pred["model_predicted_winner"] == side]
        if len(sub) > 0:
            acc = sub["model_correct"].mean()
            print(f"  Predicted {side}: {acc:.1%} ({int(sub['model_correct'].sum())}/{len(sub)})")

    # --- Where is the model WRONG and Kelly still bets? ---
    kelly_bets = has_pred[(has_pred["bet_side"] != "none") & has_pred["pm_pregame_home"].notna()].copy()
    if len(kelly_bets) > 0:
        kelly_bets["bet_won"] = kelly_bets.apply(
            lambda r: (r["bet_side"] == "home" and r["home_win"] == 1)
            or (r["bet_side"] == "away" and r["home_win"] == 0),
            axis=1,
        )
        print(f"\n--- Kelly Bet Analysis ({len(kelly_bets)} bets) ---")
        print(f"  Win rate: {kelly_bets['bet_won'].mean():.1%}")
        print(f"  Avg Kelly %: {kelly_bets['bet_kelly'].mean():.2f}%")

        # Kelly bets by edge size
        kelly_bets["model_edge"] = kelly_bets.apply(
            lambda r: r["edge_home_pm"] if r["bet_side"] == "home" else r["edge_away_pm"],
            axis=1,
        )
        edge_bins_k = [0, 0.05, 0.07, 0.10, 0.15, 1.0]
        edge_labels_k = ["0-5%", "5-7%", "7-10%", "10-15%", "15%+"]
        kelly_bets["edge_bucket"] = pd.cut(kelly_bets["model_edge"], bins=edge_bins_k, labels=edge_labels_k, right=False)

        print("\n  Kelly bets by edge size:")
        kb_stats = kelly_bets.groupby("edge_bucket", observed=True).agg(
            bets=("bet_won", "size"),
            wins=("bet_won", "sum"),
            win_rate=("bet_won", "mean"),
            avg_kelly=("bet_kelly", "mean"),
        )
        kb_stats["win_rate"] = kb_stats["win_rate"].map("{:.1%}".format)
        kb_stats["avg_kelly"] = kb_stats["avg_kelly"].map("{:.2f}%".format)
        print(kb_stats.to_string())

    return has_pred


# =========================================================================
# PART 2: Entry Point Backtesting
# =========================================================================
def backtest_entry_points(df, ph):
    print("\n" + "=" * 70)
    print("PART 2: ENTRY POINT BACKTESTING")
    print("=" * 70)

    # Only games with price history, model predictions, and results
    tradeable = df[
        (df["has_price_history"] == True)
        & df["model_home_prob"].notna()
        & df["home_win"].notna()
    ].copy()
    print(f"\nTradeable games (price history + model + result): {len(tradeable)}")

    # Determine the model's preferred side and probability
    tradeable["model_side"] = tradeable["model_predicted_winner"]
    tradeable["model_prob"] = tradeable.apply(
        lambda r: r["model_home_prob"] if r["model_predicted_winner"] == "home" else r["model_away_prob"],
        axis=1,
    )
    tradeable["model_edge"] = tradeable.apply(
        lambda r: r["edge_home_pm"] if r["model_predicted_winner"] == "home" else r["edge_away_pm"],
        axis=1,
    )
    tradeable["bet_won"] = tradeable.apply(
        lambda r: (r["model_predicted_winner"] == "home" and r["home_win"] == 1)
        or (r["model_predicted_winner"] == "away" and r["home_win"] == 0),
        axis=1,
    )

    # Merge price history with game info
    ph_merged = ph.merge(
        tradeable[["game_date", "home_team", "away_team", "model_predicted_winner",
                    "model_home_prob", "model_away_prob", "home_win"]],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    print(f"Price ticks matched to tradeable games: {len(ph_merged):,}")

    # For each tick, compute the market probability on the model's predicted side
    ph_merged["market_prob_model_side"] = ph_merged.apply(
        lambda r: r["home_price"] if r["model_predicted_winner"] == "home" else r["away_price"],
        axis=1,
    )
    ph_merged["model_prob_side"] = ph_merged.apply(
        lambda r: r["model_home_prob"] if r["model_predicted_winner"] == "home" else r["model_away_prob"],
        axis=1,
    )
    ph_merged["edge_at_tick"] = ph_merged["model_prob_side"] - ph_merged["market_prob_model_side"]
    ph_merged["bet_won"] = ph_merged.apply(
        lambda r: (r["model_predicted_winner"] == "home" and r["home_win"] == 1)
        or (r["model_predicted_winner"] == "away" and r["home_win"] == 0),
        axis=1,
    )

    # --- Strategy 1: Buy at different time windows ---
    print("\n--- Strategy: Entry by Time Before Game ---")
    time_windows = [
        ("24h+ before", 1440, 99999),
        ("12-24h before", 720, 1440),
        ("6-12h before", 360, 720),
        ("2-6h before", 120, 360),
        ("1-2h before", 60, 120),
        ("30-60min before", 30, 60),
        ("0-30min before", 0, 30),
    ]

    results = []
    for label, min_mins, max_mins in time_windows:
        window = ph_merged[
            (ph_merged["minutes_to_start"] >= min_mins)
            & (ph_merged["minutes_to_start"] < max_mins)
        ]
        if len(window) == 0:
            continue

        # Take the first tick in each window per game as entry point
        entries = window.sort_values("minutes_to_start", ascending=False).groupby(
            ["game_date", "home_team", "away_team"]
        ).first().reset_index()

        # Only take bets where model has edge > 5%
        bets = entries[entries["edge_at_tick"] > 0.05]
        if len(bets) == 0:
            results.append({"window": label, "bets": 0, "wins": 0, "win_rate": "-", "avg_edge": "-", "avg_price": "-"})
            continue

        wr = bets["bet_won"].mean()
        avg_edge = bets["edge_at_tick"].mean()
        avg_price = bets["market_prob_model_side"].mean()

        results.append({
            "window": label,
            "bets": len(bets),
            "wins": int(bets["bet_won"].sum()),
            "win_rate": f"{wr:.1%}",
            "avg_edge": f"{avg_edge:.1%}",
            "avg_price": f"{avg_price:.3f}",
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # --- Strategy 2: Edge threshold sensitivity ---
    print("\n--- Strategy: Minimum Edge Threshold (pregame entry) ---")
    # Use pregame prices from the dataset
    pregame = tradeable[tradeable["model_edge"].notna()].copy()

    thresholds = [0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
    thresh_results = []
    for thresh in thresholds:
        bets = pregame[pregame["model_edge"] >= thresh]
        if len(bets) == 0:
            continue
        wr = bets["bet_won"].mean()

        # Compute simple P&L (buy at market_prob, pay 1.0, win 1/market_prob)
        bets_copy = bets.copy()
        bets_copy["entry_price"] = bets_copy.apply(
            lambda r: r["pm_pregame_home"] if r["model_predicted_winner"] == "home" else r["pm_pregame_away"],
            axis=1,
        )
        bets_copy["pnl_per_dollar"] = bets_copy.apply(
            lambda r: (1.0 / r["entry_price"] - 1) if r["bet_won"] else -1.0,
            axis=1,
        )
        avg_pnl = bets_copy["pnl_per_dollar"].mean()
        total_roi = bets_copy["pnl_per_dollar"].sum() / len(bets_copy)

        thresh_results.append({
            "min_edge": f"{thresh:.0%}",
            "bets": len(bets),
            "wins": int(bets["bet_won"].sum()),
            "win_rate": f"{wr:.1%}",
            "avg_pnl/bet": f"{avg_pnl:+.3f}",
            "total_ROI": f"{total_roi:+.1%}",
        })

    thresh_df = pd.DataFrame(thresh_results)
    print(thresh_df.to_string(index=False))

    # --- Strategy 3: Buy the dip ---
    print("\n--- Strategy: Buy-the-Dip (enter at daily low) ---")
    # For each game, find the minimum price on the model's side in pre-game window
    pregame_ticks = ph_merged[ph_merged["minutes_to_start"] > 0]

    dip_entries = pregame_ticks.groupby(
        ["game_date", "home_team", "away_team"]
    ).agg(
        min_price=("market_prob_model_side", "min"),
        max_price=("market_prob_model_side", "max"),
        pregame_price=("market_prob_model_side", "last"),
        model_prob=("model_prob_side", "first"),
        bet_won=("bet_won", "first"),
    ).reset_index()

    dip_entries["dip_edge"] = dip_entries["model_prob"] - dip_entries["min_price"]
    dip_entries["pregame_edge"] = dip_entries["model_prob"] - dip_entries["pregame_price"]
    dip_entries["price_range"] = dip_entries["max_price"] - dip_entries["min_price"]

    # Compare: buy at dip vs buy at pregame
    for label, price_col, edge_col in [
        ("Buy at pregame", "pregame_price", "pregame_edge"),
        ("Buy at daily low (best case)", "min_price", "dip_edge"),
    ]:
        bets = dip_entries[dip_entries[edge_col] > 0.05]
        if len(bets) == 0:
            print(f"  {label}: no qualifying bets")
            continue
        wr = bets["bet_won"].mean()
        bets_copy = bets.copy()
        bets_copy["pnl"] = bets_copy.apply(
            lambda r: (1.0 / r[price_col] - 1) if r["bet_won"] else -1.0,
            axis=1,
        )
        avg_pnl = bets_copy["pnl"].mean()
        print(f"  {label}: {len(bets)} bets, {wr:.1%} win rate, avg P&L/bet: {avg_pnl:+.3f}")

    print(f"\n  Avg pre-game price swing (max-min): {dip_entries['price_range'].mean():.3f}")
    print(f"  Median pre-game price swing: {dip_entries['price_range'].median():.3f}")

    return tradeable


# =========================================================================
# PART 3: Bankroll Simulation
# =========================================================================
def simulate_bankroll(df):
    print("\n" + "=" * 70)
    print("PART 3: BANKROLL SIMULATION")
    print("=" * 70)

    # Only games with price history and Kelly sizing
    bettable = df[
        (df["bet_side"] != "none")
        & df["home_win"].notna()
        & df["pm_pregame_home"].notna()
    ].copy().sort_values("game_date")

    print(f"\nBettable games (Kelly > 0, has prices & results): {len(bettable)}")

    bettable["bet_won"] = bettable.apply(
        lambda r: (r["bet_side"] == "home" and r["home_win"] == 1)
        or (r["bet_side"] == "away" and r["home_win"] == 0),
        axis=1,
    )
    bettable["entry_price"] = bettable.apply(
        lambda r: r["pm_pregame_home"] if r["bet_side"] == "home" else r["pm_pregame_away"],
        axis=1,
    )
    bettable["model_edge"] = bettable.apply(
        lambda r: r["edge_home_pm"] if r["bet_side"] == "home" else r["edge_away_pm"],
        axis=1,
    )

    strategies = {
        "Full Kelly": lambda r: r["bet_kelly"] / 100,
        "Half Kelly": lambda r: r["bet_kelly"] / 200,
        "Quarter Kelly": lambda r: r["bet_kelly"] / 400,
        "Tiered Kelly": lambda r: (
            r["tiered_kelly_home_pm"] if r["bet_side"] == "home" else r["tiered_kelly_away_pm"]
        ) / 100 if pd.notna(r.get("tiered_kelly_home_pm")) else 0,
        "Flat 2%": lambda _: 0.02,
        "Flat 1%": lambda _: 0.01,
    }

    # Also simulate with edge filters
    edge_filters = [0.0, 0.05, 0.07, 0.10]

    for strat_name, size_fn in strategies.items():
        print(f"\n--- {strat_name} ---")
        for min_edge in edge_filters:
            filtered = bettable[bettable["model_edge"] >= min_edge] if min_edge > 0 else bettable

            bankroll = 1000.0
            peak = bankroll
            trough = bankroll
            max_drawdown = 0
            history = [bankroll]
            wins = 0
            losses = 0

            for _, row in filtered.iterrows():
                frac = size_fn(row)
                if frac <= 0 or bankroll <= 0:
                    continue

                bet_amount = bankroll * min(frac, 0.25)  # Cap at 25% per bet

                if row["bet_won"]:
                    payout = bet_amount * (1.0 / row["entry_price"] - 1)
                    bankroll += payout
                    wins += 1
                else:
                    bankroll -= bet_amount
                    losses += 1

                peak = max(peak, bankroll)
                trough = min(trough, bankroll)
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
                history.append(bankroll)

            total_bets = wins + losses
            if total_bets == 0:
                continue

            roi = (bankroll - 1000) / 1000
            edge_label = f"edge>={min_edge:.0%}" if min_edge > 0 else "all bets"
            print(
                f"  {edge_label:>10}: {total_bets:3d} bets, "
                f"W/L {wins}/{losses} ({wins/total_bets:.1%}), "
                f"${1000:.0f} -> ${bankroll:.2f} ({roi:+.1%}), "
                f"max DD {max_drawdown:.1%}"
            )

    # --- Detailed day-by-day for best strategy ---
    print("\n--- Daily P&L: Quarter Kelly, edge >= 5% ---")
    filtered = bettable[bettable["model_edge"] >= 0.05].copy()
    filtered["bet_size_frac"] = filtered["bet_kelly"] / 400

    bankroll = 1000.0
    daily_records = []

    for date, day_bets in filtered.groupby("game_date"):
        day_start = bankroll
        day_wins = 0
        day_losses = 0

        for _, row in day_bets.iterrows():
            frac = min(row["bet_size_frac"], 0.25)
            bet_amount = bankroll * frac
            if bet_amount <= 0:
                continue

            if row["bet_won"]:
                payout = bet_amount * (1.0 / row["entry_price"] - 1)
                bankroll += payout
                day_wins += 1
            else:
                bankroll -= bet_amount
                day_losses += 1

        daily_records.append({
            "date": date,
            "bets": day_wins + day_losses,
            "W/L": f"{day_wins}/{day_losses}",
            "bankroll": round(bankroll, 2),
            "day_pnl": round(bankroll - day_start, 2),
        })

    daily_df = pd.DataFrame(daily_records)
    if len(daily_df) > 0:
        print(f"  Days with bets: {len(daily_df)}")
        print(f"  Profitable days: {(daily_df['day_pnl'] > 0).sum()}/{len(daily_df)}")
        print(f"  Best day: ${daily_df['day_pnl'].max():.2f}")
        print(f"  Worst day: ${daily_df['day_pnl'].min():.2f}")
        print(f"  Final bankroll: ${bankroll:.2f}")

        # Show first/last few days
        print("\n  First 5 betting days:")
        print(daily_df.head(5).to_string(index=False))
        print("\n  Last 5 betting days:")
        print(daily_df.tail(5).to_string(index=False))

    # Save simulation detail
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(daily_df) > 0:
        daily_df.to_csv(str(OUTPUT_DIR / "daily_pnl.csv"), index=False)

    return bettable


# =========================================================================
# Main
# =========================================================================
def main():
    df = load_data()
    ph = load_price_history()

    has_pred = analyze_model_accuracy(df)
    tradeable = backtest_entry_points(df, ph)
    bettable = simulate_bankroll(df)

    # Save analysis subset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "game_date", "home_team", "away_team", "home_win",
        "pm_pregame_home", "pm_pregame_away", "pm_open_home",
        "model_home_prob", "model_away_prob",
        "edge_home_pm", "edge_away_pm",
        "kelly_home_pm", "kelly_away_pm",
        "tiered_kelly_home_pm", "tiered_kelly_away_pm",
        "bet_side", "bet_kelly", "model_predicted_winner", "model_correct",
    ]
    df[cols].to_csv(str(OUTPUT_DIR / "analysis_subset.csv"), index=False)
    print(f"\nAnalysis files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
