"""
Mispricing filter — Dr. Yang's suggestion (2026-04-17 feedback).

Dr. Yang: "You could try to create a better filter which will improve the live
performance significantly. The more accurate your model is, the less mispricing
level is required to reach a positive expected return."

This script learns a calibrated P(bet wins) from our raw features and uses a
CALIBRATED edge threshold instead of the naive model_edge >= 7%. The idea:
our model's claimed edge may over- or under-state true edge depending on
market context (pre-game drift, price range, rest). A calibration layer
corrects for that.

Pipeline:
  1. Load full 324 eligible bet candidates (bet_side != none, pm + outcome known)
  2. Build 5 features:
       - model_prob    — our model's confidence on the bet side
       - entry_price   — PM implied prob on the bet side
       - open_move     — (pm_pregame - pm_open) on bet side (+ve = market moved toward us)
       - price_range   — (pm_max - pm_min) on bet side (volatility proxy)
       - rest_diff     — days_rest of bet side minus opponent
  3. TIME-ORDERED split: first 60% = train, last 40% = test (no leakage)
  4. Fit logistic regression → calibrated P(win)
  5. Also fit isotonic on model_prob alone (baseline calibrator)
  6. Sweep thresholds:
       - baseline:  model_edge >= {0.05, 0.07, 0.09, 0.12}
       - proposed:  calibrated_edge = P_cal - entry_price >= {0.02, 0.04, 0.06, 0.08}
  7. For each gate, re-simulate on test window using Flat-2% sizing
  8. Report volume, precision (WR), Sharpe vs baseline
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from sizing_comparison import load_data, prep_bets, dual_leg_exit, STARTING_BANKROLL

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "Data" / "backtest" / "analysis"
OUT_REPORT = ANALYSIS_DIR / "mispricing_filter_report.md"

FEATURES = ["model_prob", "entry_price", "open_move", "price_range", "rest_diff"]
TRAIN_FRAC = 0.60


# ─────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────

def build_features(bets, df_raw):
    """Augment the bets DataFrame with market-behavior features.

    Features are computed ON THE BET SIDE (home or away) to avoid side bias.
    """
    # Merge open/max/min/rest from raw dataset
    keep_cols = ["game_date", "home_team", "away_team",
                 "pm_open_home", "pm_open_away",
                 "pm_home_max", "pm_home_min",
                 "days_rest_home", "days_rest_away"]
    df_raw = df_raw.copy()
    df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
    bets = bets.copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    merged = bets.merge(df_raw[keep_cols],
                        on=["game_date", "home_team", "away_team"],
                        how="left",
                        suffixes=("", "_raw"))

    # Bet-side open price
    merged["open_price"] = np.where(
        merged["bet_side"] == "home", merged["pm_open_home"], merged["pm_open_away"]
    )
    merged["open_move"] = merged["entry_price"] - merged["open_price"]

    # Price range — we only have max/min for the HOME token, so approximate:
    # for home bets it's exact; for away bets flip (max_away = 1 - min_home, etc.)
    merged["bet_side_max"] = np.where(
        merged["bet_side"] == "home",
        merged["pm_home_max"],
        1.0 - merged["pm_home_min"],
    )
    merged["bet_side_min"] = np.where(
        merged["bet_side"] == "home",
        merged["pm_home_min"],
        1.0 - merged["pm_home_max"],
    )
    merged["price_range"] = merged["bet_side_max"] - merged["bet_side_min"]

    # Rest differential (positive = bet side has more rest)
    merged["rest_diff"] = np.where(
        merged["bet_side"] == "home",
        merged["days_rest_home"] - merged["days_rest_away"],
        merged["days_rest_away"] - merged["days_rest_home"],
    )

    # Drop rows missing any feature
    before = len(merged)
    merged = merged.dropna(subset=FEATURES + ["bet_won"]).reset_index(drop=True)
    dropped = before - len(merged)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing features (of {before})")

    return merged


# ─────────────────────────────────────────────────────────────────────────
# Calibration models
# ─────────────────────────────────────────────────────────────────────────

def fit_logistic(train):
    X = train[FEATURES].to_numpy()
    y = train["bet_won"].astype(int).to_numpy()
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X), y)
    coefs = dict(zip(FEATURES, clf.coef_[0]))
    return clf, scaler, coefs


def predict_cal_prob(clf, scaler, df):
    X = df[FEATURES].to_numpy()
    return clf.predict_proba(scaler.transform(X))[:, 1]


def fit_isotonic_baseline(train):
    """Fit isotonic on model_prob alone — what does raw model prob predict?"""
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        train["model_prob"], train["bet_won"].astype(int)
    )
    return iso


# ─────────────────────────────────────────────────────────────────────────
# Simulation on a trade subset
# ─────────────────────────────────────────────────────────────────────────

def simulate_subset(trades, label):
    """Simulate Flat-2% sizing on a pre-filtered trade list (already passes gate)."""
    bankroll = STARTING_BANKROLL
    wins = losses = stopped = tp_exits = q1_exits = 0
    pnl_history = []

    for _, row in trades.iterrows():
        bet_amount = bankroll * 0.02
        if bet_amount <= 0 or bankroll <= 0:
            continue

        exit_type, exit_ratio = dual_leg_exit(row)
        if exit_type in ("stop", "take_profit", "q1_exit"):
            pnl = bet_amount * (exit_ratio - 1)
            if exit_type == "stop":
                stopped += 1
            elif exit_type == "take_profit":
                tp_exits += 1
            else:
                q1_exits += 1
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
        pnl_history.append({"date": row["game_date"], "pnl": pnl, "bankroll": bankroll})

    if not pnl_history:
        return {
            "label": label, "n": 0, "wr": 0, "final": STARTING_BANKROLL,
            "ret": 0, "maxdd": 0, "sharpe": 0,
        }

    total = wins + losses
    ret = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL

    peak = STARTING_BANKROLL
    maxdd = 0
    for h in pnl_history:
        peak = max(peak, h["bankroll"])
        maxdd = max(maxdd, (peak - h["bankroll"]) / peak if peak > 0 else 0)

    # Daily Sharpe
    daily = {}
    for h in pnl_history:
        d = h["date"]
        daily.setdefault(d, {"start": h["bankroll"] - h["pnl"], "pnl": 0})
        daily[d]["pnl"] += h["pnl"]
    drets = np.array([
        daily[d]["pnl"] / daily[d]["start"] for d in sorted(daily.keys())
        if daily[d]["start"] > 0
    ])
    sharpe = (drets.mean() / drets.std() * np.sqrt(252)) if drets.std() > 0 else 0

    return {
        "label": label,
        "n": total,
        "wins": wins,
        "wr": wins / total if total else 0,
        "final": round(bankroll, 2),
        "ret": ret,
        "maxdd": maxdd,
        "sharpe": round(sharpe, 2),
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("Loading backtest data...")
    df_raw = load_data()
    bets = prep_bets(df_raw)
    print(f"Raw eligible bets (all candidates): {len(bets)}")

    # Build features
    data = build_features(bets, df_raw)
    print(f"After feature build: {len(data)} trades")

    # Time-ordered split
    data = data.sort_values("game_date").reset_index(drop=True)
    split_idx = int(len(data) * TRAIN_FRAC)
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    print(f"Train: {len(train)} ({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"Test:  {len(test)} ({test['game_date'].min().date()} to {test['game_date'].max().date()})")

    # Fit calibrators on TRAIN only
    clf, scaler, coefs = fit_logistic(train)
    iso = fit_isotonic_baseline(train)

    print("\nLogistic regression coefficients (standardized features):")
    for f, c in coefs.items():
        print(f"  {f:<14s} {c:+.4f}")

    # Score both sets
    data["p_cal"] = np.concatenate([
        predict_cal_prob(clf, scaler, train),
        predict_cal_prob(clf, scaler, test),
    ])
    data["cal_edge"] = data["p_cal"] - data["entry_price"]
    train["p_cal"] = predict_cal_prob(clf, scaler, train)
    train["cal_edge"] = train["p_cal"] - train["entry_price"]
    test["p_cal"] = predict_cal_prob(clf, scaler, test)
    test["cal_edge"] = test["p_cal"] - test["entry_price"]

    # Baseline gate on test: model_edge >= T
    baseline_thresholds = [0.05, 0.07, 0.09, 0.12]
    cal_thresholds = [0.02, 0.04, 0.06, 0.08]

    print("\n" + "=" * 80)
    print("BASELINE FILTER (model_edge >= T) — test window")
    print("=" * 80)
    baseline_results = []
    for T in baseline_thresholds:
        gated = test[test["model_edge"] >= T]
        # apply v2-style min entry for DOGs (kept consistent with current prod)
        gated = gated[(gated["model_prob"] >= 0.60) | (gated["entry_price"] >= 0.30)]
        r = simulate_subset(gated, f"model_edge >= {T:.2f}")
        baseline_results.append(r)
        print(f"  T={T:.2f}: n={r['n']:3d} WR={r['wr']:.1%} "
              f"final=${r['final']:.2f} return={r['ret']:+.1%} "
              f"maxDD={r['maxdd']:.1%} Sharpe={r['sharpe']:.2f}")

    print("\n" + "=" * 80)
    print("MISPRICING FILTER (calibrated_edge >= T) — test window")
    print("=" * 80)
    cal_results = []
    for T in cal_thresholds:
        gated = test[test["cal_edge"] >= T]
        gated = gated[(gated["model_prob"] >= 0.60) | (gated["entry_price"] >= 0.30)]
        r = simulate_subset(gated, f"cal_edge >= {T:.2f}")
        cal_results.append(r)
        print(f"  T={T:.2f}: n={r['n']:3d} WR={r['wr']:.1%} "
              f"final=${r['final']:.2f} return={r['ret']:+.1%} "
              f"maxDD={r['maxdd']:.1%} Sharpe={r['sharpe']:.2f}")

    print("\n" + "=" * 80)
    print("COMBINED FILTER (model_edge >= 0.07 AND cal_edge >= T) — test window")
    print("=" * 80)
    combined_results = []
    for T in [0.0, 0.02, 0.04, 0.06]:
        gated = test[(test["model_edge"] >= 0.07) & (test["cal_edge"] >= T)]
        gated = gated[(gated["model_prob"] >= 0.60) | (gated["entry_price"] >= 0.30)]
        r = simulate_subset(gated, f"edge>=0.07 AND cal_edge>={T:.2f}")
        combined_results.append(r)
        print(f"  T={T:.2f}: n={r['n']:3d} WR={r['wr']:.1%} "
              f"final=${r['final']:.2f} return={r['ret']:+.1%} "
              f"maxDD={r['maxdd']:.1%} Sharpe={r['sharpe']:.2f}")

    # ─── Write report ───
    lines = [
        "# Mispricing Filter Report",
        "",
        "**Motivation:** Dr. Yang (2026-04-17): *\"Create a better filter which will "
        "improve the live performance significantly. The more accurate your model is, "
        "the less mispricing level is required to reach a positive expected return.\"*",
        "",
        "## Approach",
        "",
        "Fit a logistic regression on 5 features to learn a CALIBRATED P(bet wins). "
        "Replace the naive `model_edge >= 0.07` gate with `calibrated_edge >= T`. "
        "This corrects for market-context effects the raw edge doesn't capture.",
        "",
        "## Features",
        "",
        "| Feature | Meaning |",
        "|---------|---------|",
        "| `model_prob` | Our model's confidence on the bet side |",
        "| `entry_price` | PM implied probability on the bet side |",
        "| `open_move` | `pm_pregame - pm_open` on bet side — sharp-money direction proxy |",
        "| `price_range` | `pm_max - pm_min` on bet side — volatility / uncertainty proxy |",
        "| `rest_diff` | days_rest(bet_side) − days_rest(opponent) |",
        "",
        "## Split",
        "",
        f"- **Train:** first 60% — {len(train)} trades, "
        f"{train['game_date'].min().date()} to {train['game_date'].max().date()}",
        f"- **Test:** last 40% — {len(test)} trades, "
        f"{test['game_date'].min().date()} to {test['game_date'].max().date()}",
        "",
        "## Learned coefficients (standardized features)",
        "",
        "| Feature | Coefficient | Direction |",
        "|---------|-------------|-----------|",
    ]
    for f, c in coefs.items():
        direction = "→ wins" if c > 0 else "→ losses"
        lines.append(f"| `{f}` | {c:+.4f} | {direction} |")

    lines += [
        "",
        "## Results on test window",
        "",
        "### Baseline: `model_edge >= T`",
        "| Threshold | N | WR | Final $ | Return | MaxDD | Sharpe |",
        "|-----------|---|-----|---------|--------|-------|--------|",
    ]
    for r in baseline_results:
        lines.append(
            f"| model_edge ≥ {r['label'].split()[-1]} | {r['n']} | {r['wr']:.1%} | "
            f"${r['final']:.2f} | {r['ret']:+.1%} | {r['maxdd']:.1%} | {r['sharpe']:.2f} |"
        )

    lines += [
        "",
        "### Proposed: `cal_edge >= T`",
        "| Threshold | N | WR | Final $ | Return | MaxDD | Sharpe |",
        "|-----------|---|-----|---------|--------|-------|--------|",
    ]
    for r in cal_results:
        lines.append(
            f"| cal_edge ≥ {r['label'].split()[-1]} | {r['n']} | {r['wr']:.1%} | "
            f"${r['final']:.2f} | {r['ret']:+.1%} | {r['maxdd']:.1%} | {r['sharpe']:.2f} |"
        )

    lines += [
        "",
        "### Combined: `model_edge >= 0.07 AND cal_edge >= T`",
        "| Threshold | N | WR | Final $ | Return | MaxDD | Sharpe |",
        "|-----------|---|-----|---------|--------|-------|--------|",
    ]
    for r in combined_results:
        T = r["label"].split(">=")[-1]
        lines.append(
            f"| cal_edge ≥ {T} | {r['n']} | {r['wr']:.1%} | "
            f"${r['final']:.2f} | {r['ret']:+.1%} | {r['maxdd']:.1%} | {r['sharpe']:.2f} |"
        )

    # Pick best
    all_results = baseline_results + cal_results + combined_results
    best = max(all_results, key=lambda r: r["sharpe"] if r["n"] > 5 else -99)
    lines += [
        "",
        "## Recommendation",
        "",
        f"- Best Sharpe on test window (n>5): **{best['label']}** "
        f"— n={best['n']}, WR={best['wr']:.1%}, Sharpe={best['sharpe']:.2f}, "
        f"return={best['ret']:+.1%}.",
        "- **Caveat:** test window is small (~40% of 5 months). "
        "A rolling/walk-forward evaluation would be more robust — noted as follow-up.",
        "- **Live application:** compute `cal_edge` at bet time in `paper_trader_v2.py` "
        "and require it to clear the chosen threshold in addition to the existing edge gate.",
        "",
        "## Follow-ups for Dr. Yang meeting",
        "- Walk-forward validation on a 30-day rolling window",
        "- Expand feature set with order-book depth (liquidity-aware mispricing)",
        "- Combine with leg-weighted sizing from `leg_weight_report.md`",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {OUT_REPORT}")

    # Save calibrated data for offline inspection
    out_data = data[["game_date", "home_team", "away_team", "bet_side", "leg"
                     if "leg" in data.columns else "bet_side"] +
                    FEATURES + ["bet_won", "p_cal", "cal_edge", "model_edge"]]
    out_data.to_csv(str(ANALYSIS_DIR / "mispricing_filter_scored.csv"), index=False)
    print(f"Scored trades saved: {ANALYSIS_DIR / 'mispricing_filter_scored.csv'}")


if __name__ == "__main__":
    main()
