"""
Monthly decomposition of V2 backtest + live trades.

Question: does the FAV-leg collapse seen in live V2 (Mar 16 -> Apr 5, WR 40%, ROI -63%)
have any precursor in the backtest data itself?

Tests:
  1. Group the 76 backtest-filtered trades by month (Jan / Feb / Mar)
  2. Compute FAV vs DOG stats per month
  3. Append the 25 live trades as "Mar-late / Apr" buckets
  4. Look for a monotonic trend in FAV performance

If FAV performance degrades steadily within backtest Feb -> Mar already,
that's corroborating evidence for the end-of-season regime hypothesis
(N is larger than just the 5 live FAV bets).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "Data" / "backtest" / "analysis"
BACKTEST_LOG = ANALYSIS_DIR / "sizing_1._Flat_2pct_current.csv"
LIVE_POSITIONS = BASE_DIR / "Data" / "paper_trading_v2" / "positions.json"


def load_backtest():
    df = pd.read_csv(str(BACKTEST_LOG))
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["source"] = "backtest"
    df["bet_return"] = df["pnl"] / df["bet_amount"]
    df["bet_won"] = df["pnl"] > 0
    return df[["game_date", "source", "leg", "entry_price",
               "bet_amount", "pnl", "bet_return", "bet_won"]]


def load_live():
    with open(LIVE_POSITIONS) as f:
        pos = json.load(f)
    rows = []
    for p in pos.values():
        if p.get("pnl") is None:
            continue
        leg = "FAV" if p["is_favorite"] else "DOG"
        rows.append({
            "game_date": pd.to_datetime(p["entry_time"][:10]),
            "source": "live",
            "leg": leg,
            "entry_price": p["entry_price"],
            "bet_amount": p["bet_amount"],
            "pnl": p["pnl"],
            "bet_return": p["pnl"] / p["bet_amount"] if p["bet_amount"] else 0,
            "bet_won": p["pnl"] > 0,
        })
    return pd.DataFrame(rows)


def bucket_stats(trades, label):
    """Return a stats dict for a slice of trades."""
    if len(trades) == 0:
        return None
    return {
        "bucket": label,
        "n": len(trades),
        "wr": trades["bet_won"].mean(),
        "mean_ret": trades["bet_return"].mean(),
        "roi": trades["pnl"].sum() / trades["bet_amount"].sum(),
        "total_pnl": trades["pnl"].sum(),
    }


def fmt_row(r):
    if r is None or r["n"] == 0:
        return "  (no trades)"
    return (f"  n={r['n']:3d} WR={r['wr']:.1%} "
            f"mean_ret={r['mean_ret']:+.4f} ROI={r['roi']:+.2%} "
            f"P&L=${r['total_pnl']:+.2f}")


def decompose_by_month(df, label):
    """Split a DataFrame into monthly buckets and return stats per leg per month."""
    df = df.copy()
    df["month"] = df["game_date"].dt.to_period("M")
    print(f"\n{'=' * 90}")
    print(f"{label}")
    print('=' * 90)
    for month in sorted(df["month"].unique()):
        sub = df[df["month"] == month]
        print(f"\n--- {month} (total n={len(sub)}) ---")
        for leg in ("FAV", "DOG"):
            leg_sub = sub[sub["leg"] == leg]
            r = bucket_stats(leg_sub, leg)
            print(f"  {leg}:")
            print(fmt_row(r))


def decompose_biweekly(df, label):
    """Split by 2-week windows for finer resolution."""
    df = df.copy()
    df["week_bucket"] = (
        df["game_date"]
        .apply(lambda d: (d - pd.Timestamp("2026-01-01")).days // 14)
    )
    print(f"\n{'=' * 90}")
    print(f"{label} — 2-week buckets")
    print('=' * 90)
    buckets = sorted(df["week_bucket"].unique())
    for b in buckets:
        sub = df[df["week_bucket"] == b]
        start = sub["game_date"].min().date()
        end = sub["game_date"].max().date()
        print(f"\n--- Weeks {b} ({start} to {end}, n={len(sub)}) ---")
        for leg in ("FAV", "DOG"):
            leg_sub = sub[sub["leg"] == leg]
            r = bucket_stats(leg_sub, leg)
            print(f"  {leg}:")
            print(fmt_row(r))


def main():
    print("Loading backtest + live trade logs...")
    bt = load_backtest()
    live = load_live()
    print(f"  backtest trades: {len(bt)} ({bt['game_date'].min().date()} to {bt['game_date'].max().date()})")
    print(f"  live trades:     {len(live)} ({live['game_date'].min().date()} to {live['game_date'].max().date()})")

    all_trades = pd.concat([bt, live], ignore_index=True).sort_values("game_date")
    print(f"  combined:        {len(all_trades)}")

    # Per-month decomposition
    decompose_by_month(all_trades, "COMBINED — monthly")

    # Per-two-week for finer resolution (late-season only)
    late = all_trades[all_trades["game_date"] >= pd.Timestamp("2026-02-15")]
    decompose_biweekly(late, "LATE SEASON (Feb 15 onward)")

    # Summary trend: FAV WR over time (rolling)
    print(f"\n{'=' * 90}")
    print("FAV-leg trend (rolling 10-bet window)")
    print('=' * 90)
    fav = all_trades[all_trades["leg"] == "FAV"].sort_values("game_date").reset_index(drop=True)
    print(f"Total FAV bets across backtest + live: {len(fav)}")
    if len(fav) >= 10:
        fav["roll_wr"] = fav["bet_won"].rolling(10).mean()
        fav["roll_roi"] = fav["pnl"].rolling(10).sum() / fav["bet_amount"].rolling(10).sum()
        for i in range(9, len(fav), 3):  # every 3rd row for readability
            row = fav.iloc[i]
            print(f"  {row['game_date'].date()}  (through bet #{i+1}): "
                  f"rolling 10-bet WR={row['roll_wr']:.1%}, ROI={row['roll_roi']:+.2%}")

    # Summary table for report
    print(f"\n{'=' * 90}")
    print("SUMMARY: FAV-leg performance per month")
    print('=' * 90)
    summary = []
    all_trades["month"] = all_trades["game_date"].dt.to_period("M")
    for month in sorted(all_trades["month"].unique()):
        fav_sub = all_trades[(all_trades["month"] == month) & (all_trades["leg"] == "FAV")]
        dog_sub = all_trades[(all_trades["month"] == month) & (all_trades["leg"] == "DOG")]
        print(f"\n{month}:")
        print(f"  FAV: {fmt_row(bucket_stats(fav_sub, 'FAV'))}")
        print(f"  DOG: {fmt_row(bucket_stats(dog_sub, 'DOG'))}")
        summary.append({
            "month": str(month),
            "fav_n": len(fav_sub),
            "fav_wr": fav_sub["bet_won"].mean() if len(fav_sub) else np.nan,
            "fav_roi": fav_sub["pnl"].sum() / fav_sub["bet_amount"].sum() if len(fav_sub) else np.nan,
            "dog_n": len(dog_sub),
            "dog_wr": dog_sub["bet_won"].mean() if len(dog_sub) else np.nan,
            "dog_roi": dog_sub["pnl"].sum() / dog_sub["bet_amount"].sum() if len(dog_sub) else np.nan,
        })

    summary_df = pd.DataFrame(summary)
    out = ANALYSIS_DIR / "monthly_decomposition.csv"
    summary_df.to_csv(str(out), index=False)

    # Build report
    lines = [
        "# Monthly Decomposition Report",
        "",
        "**Question:** Is the end-of-season regime shift visible in the backtest itself, "
        "or only in the live N=25 sample?",
        "",
        "## Per-month breakdown (backtest + live combined)",
        "",
        "| Month | FAV N | FAV WR | FAV ROI | DOG N | DOG WR | DOG ROI |",
        "|-------|-------|--------|---------|-------|--------|---------|",
    ]
    for s in summary:
        fav_wr = f"{s['fav_wr']:.1%}" if pd.notna(s["fav_wr"]) else "—"
        fav_roi = f"{s['fav_roi']:+.2%}" if pd.notna(s["fav_roi"]) else "—"
        dog_wr = f"{s['dog_wr']:.1%}" if pd.notna(s["dog_wr"]) else "—"
        dog_roi = f"{s['dog_roi']:+.2%}" if pd.notna(s["dog_roi"]) else "—"
        lines.append(
            f"| {s['month']} | {s['fav_n']} | {fav_wr} | {fav_roi} | "
            f"{s['dog_n']} | {dog_wr} | {dog_roi} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Look for a monotonic trend in FAV-WR / FAV-ROI as months progress.",
        "- If FAV-leg already degraded Feb -> Mar *within the backtest*, "
        "the live April crash is consistent with an ongoing regime shift (not just variance).",
        "- If FAV-WR stays flat through Mar-15 and only drops on Mar 16+, the regime shift "
        "is sharper — something specific about late March broke the model.",
        "",
        f"Summary CSV saved: `{out.name}`",
    ]
    (ANALYSIS_DIR / "monthly_decomposition_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"\nSummary CSV: {out}")
    print(f"Report: {ANALYSIS_DIR / 'monthly_decomposition_report.md'}")


if __name__ == "__main__":
    main()
