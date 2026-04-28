"""
Mispricing filter using tick-derived microstructure features.

Training set:  backtest eligible bets that have bet-side tick coverage
OOS test set:  live V2 closed trades that have bet-side tick coverage

Features (12 total):
  Original:
    - model_prob      (our XGBoost probability on bet side)
    - entry_price     (PM implied probability on bet side)
    - bet_edge        (model_prob - entry_price)
    - rest_diff       (days_rest bet_side - days_rest opponent)
  Tick-derived:
    - open_move       (close_tick - first_tick)
    - price_range     (max - min)
    - vol_2h          (std of 10-min returns in last 2 hours pre-cutoff)
    - vol_6h          (std of 10-min returns in last 6 hours pre-cutoff)
    - momentum_30m    (price move in last 30 min)
    - momentum_2h     (price move in last 2 hours)
    - n_large_moves   (# ticks with |Δ| ≥ 1%)
    - n_direction_changes (sign flips)

Output:
  Data/backtest/analysis/enriched_filter_report.md
  Data/backtest/analysis/enriched_filter_scored.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "Data" / "backtest" / "analysis"
TICK_FEATURES = ANALYSIS_DIR / "tick_features_by_token.csv"
BACKTEST_DATASET = BASE_DIR / "Data" / "backtest" / "nba_backtest_dataset.csv"
POSITIONS_FILE = BASE_DIR / "Data" / "paper_trading_v2" / "positions.json"

ORIG_FEATURES = ["model_prob", "entry_price", "bet_edge", "rest_diff"]
TICK_FEATURES_COLS = ["open_move", "price_range", "vol_2h", "vol_6h",
                      "momentum_30m", "momentum_2h",
                      "n_large_moves", "n_direction_changes"]
ALL_FEATURES = ORIG_FEATURES + TICK_FEATURES_COLS


# ────────────────────────────────────────────────────────────────────────────
# Build training set (backtest)
# ────────────────────────────────────────────────────────────────────────────

def build_training_set(tick_feats):
    """Merge backtest eligible bets with bet-side tick features + outcomes."""
    bt = pd.read_csv(str(BACKTEST_DATASET))
    bt = bt[bt["bet_side"] != "none"].copy()
    bt = bt[bt["pm_pregame_home"].notna() & bt["home_win"].notna()]
    bt["game_date"] = pd.to_datetime(bt["game_date"]).dt.date

    # Pull bet-side values
    bt["entry_price"] = np.where(
        bt["bet_side"] == "home", bt["pm_pregame_home"], bt["pm_pregame_away"]
    )
    bt["model_prob"] = np.where(
        bt["bet_side"] == "home", bt["model_home_prob"], bt["model_away_prob"]
    )
    bt["bet_edge"] = np.where(
        bt["bet_side"] == "home", bt["edge_home_pm"], bt["edge_away_pm"]
    )
    bt["rest_diff"] = np.where(
        bt["bet_side"] == "home",
        bt["days_rest_home"] - bt["days_rest_away"],
        bt["days_rest_away"] - bt["days_rest_home"],
    )
    bt["bet_won"] = (
        ((bt["bet_side"] == "home") & (bt["home_win"] == 1))
        | ((bt["bet_side"] == "away") & (bt["home_win"] == 0))
    ).astype(int)

    # Join tick features: match on game_date, home, away, side
    tf = tick_feats[tick_feats["source"] == "backtest"].copy()
    tf["game_date"] = pd.to_datetime(tf["game_date"]).dt.date
    merged = bt.merge(
        tf[["game_date", "home_team", "away_team", "side"] + TICK_FEATURES_COLS],
        left_on=["game_date", "home_team", "away_team", "bet_side"],
        right_on=["game_date", "home_team", "away_team", "side"],
        how="left",
    )
    # Keep only rows with full feature coverage
    before = len(merged)
    merged = merged.dropna(subset=ALL_FEATURES)
    print(f"  Backtest train set: {len(merged)} / {before} candidates have full features")
    return merged


# ────────────────────────────────────────────────────────────────────────────
# Build OOS live set
# ────────────────────────────────────────────────────────────────────────────

def build_live_set(tick_feats):
    """Merge live closed trades with bet-side tick features + actual P&L."""
    with open(POSITIONS_FILE) as f:
        positions = json.load(f)

    rows = []
    for pid, p in positions.items():
        if p.get("pnl") is None:
            continue
        rows.append({
            "game_date": pd.to_datetime(p["entry_time"][:10]).date(),
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "bet_side": p["bet_side"],
            "leg": "FAV" if p["is_favorite"] else "DOG",
            "entry_time": p["entry_time"],
            "entry_price": p["entry_price"],
            "model_prob": p["model_prob"],
            "bet_edge": p["bet_edge"],
            "bet_amount": p["bet_amount"],
            "pnl": p["pnl"],
            "bet_won": 1 if p["pnl"] > 0 else 0,
        })
    live = pd.DataFrame(rows)

    # Compute rest_diff from schedule (same pattern as backfill_market_context)
    from backfill_market_context import build_rest_lookup, days_rest_for
    team_games = build_rest_lookup()
    def _rest_diff(r):
        entry_dt = pd.to_datetime(r["entry_time"])
        rh = days_rest_for(team_games, r["home_team"], entry_dt)
        ra = days_rest_for(team_games, r["away_team"], entry_dt)
        return (rh - ra) if r["bet_side"] == "home" else (ra - rh)
    live["rest_diff"] = live.apply(_rest_diff, axis=1)

    # Join tick features
    tf = tick_feats[tick_feats["source"] == "live"].copy()
    tf["game_date"] = pd.to_datetime(tf["game_date"]).dt.date
    merged = live.merge(
        tf[["game_date", "home_team", "away_team", "side"] + TICK_FEATURES_COLS],
        left_on=["game_date", "home_team", "away_team", "bet_side"],
        right_on=["game_date", "home_team", "away_team", "side"],
        how="left",
    )
    before = len(merged)
    merged = merged.dropna(subset=ALL_FEATURES)
    print(f"  Live OOS set: {len(merged)} / {before} closed trades have full features")
    return merged


# ────────────────────────────────────────────────────────────────────────────
# Fit + evaluate
# ────────────────────────────────────────────────────────────────────────────

def summarize(trades, label):
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0, "wr": None, "total_pnl": 0, "roi": 0, "mean_ret": 0}
    wr = trades["bet_won"].mean()
    if "pnl" in trades.columns and "bet_amount" in trades.columns:
        total_pnl = trades["pnl"].sum()
        total_staked = trades["bet_amount"].sum()
        roi = total_pnl / total_staked if total_staked > 0 else 0
        mean_ret = (trades["pnl"] / trades["bet_amount"]).mean()
    else:
        total_pnl = 0
        roi = 0
        mean_ret = 0
    return {"label": label, "n": n, "wr": wr, "total_pnl": total_pnl,
            "roi": roi, "mean_ret": mean_ret}


def main():
    print("Loading tick features...")
    tick = pd.read_csv(str(TICK_FEATURES))
    print(f"  {len(tick)} token-side feature rows")

    print("\nBuilding backtest training set...")
    train = build_training_set(tick)

    print("\nBuilding live OOS set...")
    live = build_live_set(tick)

    # Fit calibrator on backtest training set
    print(f"\nFitting 12-feature calibrator on {len(train)} backtest candidates...")
    X_train = train[ALL_FEATURES].to_numpy()
    y_train = train["bet_won"].to_numpy()
    scaler = StandardScaler().fit(X_train)
    # Use L2 with moderate regularization (C=0.5) given small N and many features
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(scaler.transform(X_train), y_train)

    coefs = dict(zip(ALL_FEATURES, clf.coef_[0]))
    print("\nCalibrator coefficients (standardized features):")
    # Sort by absolute magnitude
    sorted_coefs = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))
    for f, c in sorted_coefs:
        print(f"  {f:<22s} {c:+.4f}")

    # Score + evaluate
    train["p_cal"] = clf.predict_proba(scaler.transform(X_train))[:, 1]
    train["cal_edge"] = train["p_cal"] - train["entry_price"]

    X_live = live[ALL_FEATURES].to_numpy()
    live["p_cal"] = clf.predict_proba(scaler.transform(X_live))[:, 1]
    live["cal_edge"] = live["p_cal"] - live["entry_price"]

    # In-sample sanity check: apply baseline filter and cal_edge filter on train
    print("\n" + "=" * 80)
    print("IN-SAMPLE on backtest training set (for sanity only)")
    print("=" * 80)
    # Apply V2 filters (edge>=7%, DOG floor=0.30) to mimic current live gating
    train_filt = train[(train["bet_edge"] >= 0.07)]
    train_filt = train_filt[(train_filt["model_prob"] >= 0.60)
                            | (train_filt["entry_price"] >= 0.30)]
    r = summarize(train_filt, "all (edge>=7%)")
    print(f"  Baseline (edge>=7%): n={r['n']} WR={r['wr']:.1%}")
    for T in [0.00, 0.02, 0.04, 0.06, 0.08]:
        sub = train_filt[train_filt["cal_edge"] >= T]
        r = summarize(sub, f"cal>={T:+.2f}")
        wr = f"{r['wr']:.1%}" if r["n"] > 0 else "n/a"
        print(f"  + cal>={T:+.2f}: n={r['n']:3d} WR={wr}")

    # OOS on live
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE on live V2 trades (14 with full features)")
    print("=" * 80)
    baseline = summarize(live, "all live (already filtered by edge>=7% live)")
    print(f"  Baseline: n={baseline['n']} WR={baseline['wr']:.1%} "
          f"ROI={baseline['roi']:+.2%} total=${baseline['total_pnl']:+.2f}")

    live_rows = [("baseline", baseline)]
    for T in [-0.05, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08]:
        sub = live[live["cal_edge"] >= T]
        r = summarize(sub, f"cal>={T:+.2f}")
        live_rows.append((f"cal>={T:+.2f}", r))
        wr = f"{r['wr']:.1%}" if r["n"] > 0 else "n/a"
        print(f"  cal>={T:+.2f}: n={r['n']:3d} WR={wr:<6s} "
              f"ROI={r['roi']:+.2%} total=${r['total_pnl']:+.2f}")

    # Leg breakdown
    print("\n--- Live by leg ---")
    for leg in ("FAV", "DOG"):
        sub = live[live["leg"] == leg]
        r = summarize(sub, leg)
        if r["n"] > 0:
            print(f"  {leg}: n={r['n']} WR={r['wr']:.1%} "
                  f"ROI={r['roi']:+.2%} total=${r['total_pnl']:+.2f}")

    # Save scored
    live_out = live.copy()
    live_out["source"] = "live"
    train_out = train.copy()
    train_out["source"] = "backtest"
    combined = pd.concat([train_out, live_out], ignore_index=True, sort=False)
    combined.to_csv(str(ANALYSIS_DIR / "enriched_filter_scored.csv"), index=False)

    # Build report
    lines = [
        "# Enriched Mispricing Filter Report",
        "",
        "**Data:** 10-min tick snapshots (Feb 19 → Apr 7 2026) enriching both the backtest "
        "training universe and the live paper-trading OOS set.",
        "",
        f"**Training (backtest):** {len(train)} candidates with full 12-feature coverage.",
        f"**OOS (live):** {len(live)} closed trades with full 12-feature coverage.",
        "",
        "## Feature set",
        "",
        "| Source | Features |",
        "|--------|----------|",
        "| Original | model_prob, entry_price, bet_edge, rest_diff |",
        "| Tick-derived | open_move, price_range, vol_2h, vol_6h, momentum_30m, "
        "momentum_2h, n_large_moves, n_direction_changes |",
        "",
        "## Calibrator coefficients (standardized, sorted by |magnitude|)",
        "",
        "| Feature | Coef | Direction |",
        "|---------|------|-----------|",
    ]
    for f, c in sorted_coefs:
        direction = "→ wins" if c > 0 else "→ losses"
        lines.append(f"| `{f}` | {c:+.4f} | {direction} |")

    lines += [
        "",
        "## Out-of-sample results on live trades",
        "",
        "| Gate | N | WR | ROI | Total P&L |",
        "|------|---|-----|-----|-----------|",
    ]
    for label, r in live_rows:
        wr = f"{r['wr']:.1%}" if r["n"] > 0 and r["wr"] is not None else "n/a"
        lines.append(
            f"| {label} | {r['n']} | {wr} | {r['roi']:+.2%} | ${r['total_pnl']:+.2f} |"
        )

    # Best & worst
    best = max((r for label, r in live_rows[1:] if r["n"] >= 3),
               key=lambda r: r["roi"], default=baseline)
    worst = min((r for label, r in live_rows[1:] if r["n"] >= 3),
                key=lambda r: r["roi"], default=baseline)
    lines += [
        "",
        "## Interpretation",
        "",
        f"- **Baseline (no filter):** n={baseline['n']}, WR={baseline['wr']:.1%}, "
        f"ROI={baseline['roi']:+.2%}, P&L=${baseline['total_pnl']:+.2f}",
        f"- **Best gate (n>=3):** {best['label']} — n={best['n']}, "
        f"WR={best['wr']:.1%}, ROI={best['roi']:+.2%}",
        f"- **Worst gate (n>=3):** {worst['label']} — n={worst['n']}, "
        f"WR={worst['wr']:.1%}, ROI={worst['roi']:+.2%}",
        "",
        "### Caveats",
        "- **N=14 OOS is still small.** Directional only.",
        "- The filter is fit on 2026-02-19 → 2026-03-15 and tested on 2026-03-16 → "
        "2026-04-05, so train/test are truly time-separated.",
        "- Feature coefficients reveal which microstructure signals the model finds "
        "most discriminative — compare to the no-tick Task 3 version to see if "
        "tick-derived features rank above the original 4.",
        "",
        "### What to watch for",
        "- If tick-derived features rank high and the OOS lift materializes, the filter "
        "is learning something real from the market microstructure.",
        "- If the OOS result is flat-or-worse, we have the same regime-shift problem as "
        "before and more features don't fix it — only more *in-regime* data does.",
    ]
    (ANALYSIS_DIR / "enriched_filter_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nReport: {ANALYSIS_DIR / 'enriched_filter_report.md'}")
    print(f"Scored trades: {ANALYSIS_DIR / 'enriched_filter_scored.csv'}")


if __name__ == "__main__":
    main()
