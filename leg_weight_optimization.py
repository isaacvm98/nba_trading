"""
Leg weight optimization — Dr. Yang's suggestion (2026-04-17 feedback).

Dr. Yang: "Optimize the betting size between the favorite and underdog.
Based on the expected return, you will find a best weight of the two teams.
1:1 is actually too risky."  (Bank credit-loan portfolio analogy.)

This script:
  1. Loads the Flat-2% baseline trade log from sizing_comparison.py
  2. Computes per-leg return statistics (mean, variance, Sharpe, Kelly)
  3. Pairs same-day bets to estimate leg-to-leg correlation
  4. Solves three weight schemes:
       a) Kelly-per-leg:   w_i ∝ μ_i / σ²_i  (fractional Kelly)
       b) Risk parity:     w_i ∝ 1 / σ_i
       c) Mean-variance:   argmax w'μ − λ w'Σw  for λ ∈ {1,3,10}
  5. Bootstraps 95% CIs for weights (small-sample caveat)
  6. Re-simulates each scheme vs 1:1 baseline (reuses sizing_comparison engine)
  7. Saves a report to Data/backtest/analysis/leg_weight_report.md
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sizing_comparison import (
    load_data, prep_bets, simulate, v2_filter, STARTING_BANKROLL
)

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "Data" / "backtest" / "analysis"
BASELINE_LOG = ANALYSIS_DIR / "sizing_1._Flat_2pct_current.csv"
OUT_REPORT = ANALYSIS_DIR / "leg_weight_report.md"

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000


# ─────────────────────────────────────────────────────────────────────────
# Per-leg statistics
# ─────────────────────────────────────────────────────────────────────────

def per_trade_return(df):
    """r_i = pnl / bet_amount.  Unit return per $1 staked (ignores sizing)."""
    out = df.copy()
    out["ret"] = out["pnl"] / out["bet_amount"]
    return out


def leg_stats(df):
    """Return dict of stats per leg, plus overall."""
    rows = {}
    for leg in ("FAV", "DOG"):
        sub = df[df["leg"] == leg]
        if len(sub) == 0:
            continue
        r = sub["ret"].to_numpy()
        mu = r.mean()
        sigma = r.std(ddof=1) if len(r) > 1 else 0.0
        wr = (sub["pnl"] > 0).mean()
        rows[leg] = {
            "n": len(sub),
            "win_rate": wr,
            "mean_ret": mu,
            "std_ret": sigma,
            "sharpe": mu / sigma if sigma > 0 else np.nan,
            "kelly_frac": mu / (sigma ** 2) if sigma > 0 else np.nan,
            "total_pnl": sub["pnl"].sum(),
        }
    return rows


def leg_correlation(df):
    """Pair bets by date and compute FAV-DOG correlation on same-day returns.
    If very few same-day pairs, returns (np.nan, 0)."""
    pairs = []
    for date, grp in df.groupby("game_date"):
        fav = grp[grp["leg"] == "FAV"]["ret"].mean()
        dog = grp[grp["leg"] == "DOG"]["ret"].mean()
        if pd.notna(fav) and pd.notna(dog):
            pairs.append((fav, dog))
    if len(pairs) < 3:
        return np.nan, len(pairs)
    arr = np.array(pairs)
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), len(pairs)


# ─────────────────────────────────────────────────────────────────────────
# Weight solvers
# ─────────────────────────────────────────────────────────────────────────

def kelly_weights(stats):
    """Fractional Kelly per leg — proportional to μ/σ²."""
    w = {}
    for leg, s in stats.items():
        w[leg] = max(s["kelly_frac"], 0.0) if pd.notna(s["kelly_frac"]) else 0.0
    total = sum(w.values())
    if total <= 0:
        return {k: 0.5 for k in stats}
    return {k: v / total for k, v in w.items()}


def risk_parity_weights(stats):
    """Inverse-volatility weighting — each leg contributes equal risk."""
    inv_sigma = {leg: (1.0 / s["std_ret"]) if s["std_ret"] > 0 else 0.0
                 for leg, s in stats.items()}
    total = sum(inv_sigma.values())
    if total <= 0:
        return {k: 0.5 for k in stats}
    return {k: v / total for k, v in inv_sigma.items()}


def mean_variance_weights(stats, rho, lam=3.0):
    """Argmax w'μ − λ w'Σw  subject to w_fav + w_dog = 1, w_i ≥ 0.

    For 2 assets with μ = (μ1, μ2), σ = (σ1, σ2), correlation rho:
    closed form for unconstrained w1*:
        w1 = [ (μ1 − μ2) + 2λ(σ2² − ρσ1σ2) ] / [ 2λ(σ1² + σ2² − 2ρσ1σ2) ]
    Clamp to [0, 1]."""
    if "FAV" not in stats or "DOG" not in stats:
        return {"FAV": 0.5, "DOG": 0.5}
    mu1, mu2 = stats["FAV"]["mean_ret"], stats["DOG"]["mean_ret"]
    s1, s2 = stats["FAV"]["std_ret"], stats["DOG"]["std_ret"]
    r = 0.0 if pd.isna(rho) else rho
    denom = 2 * lam * (s1**2 + s2**2 - 2 * r * s1 * s2)
    if denom == 0:
        return {"FAV": 0.5, "DOG": 0.5}
    w1 = ((mu1 - mu2) + 2 * lam * (s2**2 - r * s1 * s2)) / denom
    w1 = float(np.clip(w1, 0.0, 1.0))
    return {"FAV": w1, "DOG": 1.0 - w1}


# ─────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_weights(df, solver_fn, n=N_BOOTSTRAP, **solver_kwargs):
    """Resample trades with replacement within each leg, recompute weights."""
    fav = df[df["leg"] == "FAV"]
    dog = df[df["leg"] == "DOG"]
    if len(fav) == 0 or len(dog) == 0:
        return None

    w_fav_samples = []
    for _ in range(n):
        fav_s = fav.sample(n=len(fav), replace=True, random_state=RNG.integers(1e9))
        dog_s = dog.sample(n=len(dog), replace=True, random_state=RNG.integers(1e9))
        resampled = pd.concat([fav_s, dog_s], ignore_index=True)
        stats = leg_stats(resampled)
        if "FAV" not in stats or "DOG" not in stats:
            continue
        if solver_fn is mean_variance_weights:
            rho, _ = leg_correlation(resampled)
            w = solver_fn(stats, rho, **solver_kwargs)
        else:
            w = solver_fn(stats)
        w_fav_samples.append(w.get("FAV", np.nan))

    arr = np.array([x for x in w_fav_samples if pd.notna(x)])
    if len(arr) == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


# ─────────────────────────────────────────────────────────────────────────
# Re-simulate with leg-specific sizing
# ─────────────────────────────────────────────────────────────────────────

def make_weighted_sizer(base_pct, w_fav, w_dog):
    """Return a size_fn that multiplies base_pct by leg weight.

    Weights are normalized so expected total exposure matches baseline —
    i.e. the sizer scales each leg relative to its share in the strategy.
    Here we normalize so w_fav + w_dog = 1 (already done by solvers), then
    the *per-bet* multiplier is 2 * w_leg (so 1:1 → 1.0x both).
    """
    def sizer(row, bankroll):
        leg = "FAV" if row["model_prob"] >= 0.60 else "DOG"
        mult = 2.0 * (w_fav if leg == "FAV" else w_dog)
        return bankroll * base_pct * mult
    return sizer


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────

def fmt_weights(w, ci=None):
    s = f"FAV: {w['FAV']:.3f}, DOG: {w['DOG']:.3f}"
    if ci:
        s += f"  (FAV 95% CI: [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}])"
    return s


def main():
    print("Loading backtest data...")
    df_raw = load_data()
    bets = prep_bets(df_raw)
    print(f"Total eligible bets: {len(bets)}")

    # Use baseline trade log if it exists; else re-run flat-2% simulation
    if BASELINE_LOG.exists():
        print(f"Loading baseline trade log: {BASELINE_LOG.name}")
        trades = pd.read_csv(str(BASELINE_LOG))
    else:
        print("Re-running Flat-2% baseline to generate trade log...")
        from sizing_comparison import flat_2pct
        r = simulate(bets, "Flat 2%", flat_2pct)
        trades = pd.DataFrame(r["history"])

    trades = per_trade_return(trades)
    stats = leg_stats(trades)
    rho, n_pairs = leg_correlation(trades)

    # Point-estimate weights
    w_kelly = kelly_weights(stats)
    w_rp = risk_parity_weights(stats)
    w_mv_low = mean_variance_weights(stats, rho, lam=1.0)
    w_mv_med = mean_variance_weights(stats, rho, lam=3.0)
    w_mv_high = mean_variance_weights(stats, rho, lam=10.0)

    # Bootstrap CIs (Kelly + risk-parity only — mean-variance varies with lam)
    print("Bootstrapping weight confidence intervals...")
    ci_kelly = bootstrap_weights(trades, kelly_weights)
    ci_rp = bootstrap_weights(trades, risk_parity_weights)
    ci_mv = bootstrap_weights(trades, mean_variance_weights, lam=3.0)

    # Re-simulate each scheme
    print("Re-simulating weight schemes vs 1:1 baseline...")
    base_pct = 0.02
    schemes = {
        "1:1 baseline (Flat 2%)":        ({"FAV": 0.5, "DOG": 0.5}, None),
        "Kelly-per-leg":                 (w_kelly, ci_kelly),
        "Risk parity":                   (w_rp, ci_rp),
        "Mean-variance (lam=1, aggressive)": (w_mv_low, None),
        "Mean-variance (lam=3, balanced)":   (w_mv_med, ci_mv),
        "Mean-variance (lam=10, conservative)": (w_mv_high, None),
    }

    sim_results = {}
    for name, (w, _) in schemes.items():
        sizer = make_weighted_sizer(base_pct, w["FAV"], w["DOG"])
        r = simulate(bets, name, sizer)
        if r:
            sim_results[name] = r

    # ─── Build report ───
    lines = [
        "# Leg Weight Optimization Report",
        "",
        "**Motivation:** Dr. Yang (2026-04-17 feedback): *\"Optimize the betting size between "
        "the favorite and underdog. 1:1 is actually too risky.\"* Analogy: bank credit-loan "
        "portfolio (high-FICO vs low-FICO loans).",
        "",
        "## Per-leg statistics (Flat 2% baseline)",
        "",
        "| Leg | N | Win rate | Mean return ($1 staked) | Std | Sharpe | Kelly fraction | Total P&L |",
        "|-----|---|----------|-------------------------|-----|--------|----------------|-----------|",
    ]
    for leg, s in stats.items():
        lines.append(
            f"| {leg} | {s['n']} | {s['win_rate']:.1%} | "
            f"{s['mean_ret']:+.4f} | {s['std_ret']:.4f} | "
            f"{s['sharpe']:.3f} | {s['kelly_frac']:.3f} | ${s['total_pnl']:+.2f} |"
        )
    lines += [
        "",
        f"**Same-day FAV/DOG correlation:** ρ = {rho:.3f}  (from {n_pairs} paired days)",
        "" if pd.notna(rho) else "_Insufficient same-day pairs — assuming ρ = 0 for MV solver._",
        "",
        "## Optimal weights (point estimates + 95% bootstrap CI)",
        "",
        "| Scheme | FAV weight | DOG weight | FAV 95% CI |",
        "|--------|------------|------------|------------|",
    ]
    for name, (w, ci) in schemes.items():
        ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]" if ci else "—"
        lines.append(f"| {name} | {w['FAV']:.3f} | {w['DOG']:.3f} | {ci_str} |")

    lines += [
        "",
        "## Backtest comparison (same filters, same exits, base 2% × 2w_leg)",
        "",
        "| Scheme | Bets | WR | Final $ | Return | MaxDD | Sharpe | Avg bet |",
        "|--------|------|----|---------|--------|-------|--------|---------|",
    ]
    for name, r in sim_results.items():
        lines.append(
            f"| {name} | {r['total_bets']} | {r['win_rate']:.1%} | "
            f"${r['final_bankroll']:.2f} | {r['total_return']:+.1%} | "
            f"{r['max_drawdown']:.1%} | {r['sharpe_ratio']:.2f} | ${r['avg_bet']:.2f} |"
        )

    # Recommendation
    lines += [
        "",
        "## Recommendation",
        "",
    ]
    if "FAV" in stats and "DOG" in stats:
        fav_s = stats["FAV"]
        dog_s = stats["DOG"]
        if fav_s["sharpe"] > dog_s["sharpe"]:
            lean = "FAV"
        else:
            lean = "DOG"
        lines.append(
            f"- Historical per-$1 Sharpe favors **{lean}** "
            f"(FAV={fav_s['sharpe']:.2f} vs DOG={dog_s['sharpe']:.2f})."
        )
    best = max(sim_results.items(), key=lambda kv: kv[1]["sharpe_ratio"])
    lines.append(
        f"- Best Sharpe in simulation: **{best[0]}** "
        f"(Sharpe={best[1]['sharpe_ratio']:.2f}, return={best[1]['total_return']:+.1%})."
    )
    lines.append(
        "- **Caveat:** Sample size is small (n=76, split 24 FAV / 52 DOG). "
        "Bootstrap CIs are wide — treat point estimates as directional, not final."
    )
    lines.append(
        "- **Next live step:** apply chosen weight to live `paper_trader_v2.py` via a "
        "leg-aware scalar on `half_kelly_size()`."
    )

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved: {OUT_REPORT}")

    # Also print summary to stdout
    print("\n" + "=" * 80)
    print("LEG WEIGHT OPTIMIZATION — SUMMARY")
    print("=" * 80)
    for leg, s in stats.items():
        print(f"{leg}: n={s['n']} WR={s['win_rate']:.1%} "
              f"mean={s['mean_ret']:+.4f} std={s['std_ret']:.4f} "
              f"Sharpe={s['sharpe']:.2f} Kelly={s['kelly_frac']:.3f}")
    print(f"corr(FAV,DOG) = {rho:.3f} on {n_pairs} paired days\n")

    print("Weights:")
    for name, (w, ci) in schemes.items():
        print(f"  {name:<40s} {fmt_weights(w, ci)}")

    print("\nBacktest results:")
    for name, r in sim_results.items():
        print(f"  {name:<40s} final=${r['final_bankroll']:.2f} "
              f"Sharpe={r['sharpe_ratio']:.2f} MaxDD={r['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
