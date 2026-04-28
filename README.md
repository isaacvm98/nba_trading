# NBA Polymarket Dual-Leg Strategy — Code & Data

Companion repository for the report **"Quantitative Strategy Design & Implementation — NBA Polymarket Dual-Leg Strategy"** (Isaac Vergara, Duke University, 2026).

The PDF report is submitted separately. This repo contains every script and dataset needed to reproduce every table, figure, and number cited in it.

---

## What's here

```
.
├── main.py                       NBA prediction CLI (XGBoost on live stats)
├── config.toml                   Season date ranges
├── requirements.txt
│
├── build_*.py / fetch_*.py       Data pipeline (Polymarket, ESPN WP, Q1 scores, history)
├── backtest_*.py                 Backtest engines (Stages 1–4, late-season retro)
├── sizing_comparison.py          Flat / Half-Kelly / Edge-scaled sizing comparison
├── flip_backtest.py              Dr. Yang's flip-strategy test
├── monthly_decomposition.py      Monthly FAV / DOG performance decomposition
├── mispricing_filter.py          Logistic-regression filter on raw model_edge
├── enriched_filter.py            12-feature filter using tick microstructure
├── tick_features.py              10-min PM tick → microstructure features
├── leg_weight_optimization.py    FAV / DOG Kelly + risk-parity + mean-variance
├── backfill_market_context.py    Retroactive feature enrichment for live trades
│
├── src/
│   ├── DataProviders/            Polymarket Gamma, ESPN, NBA, SBR, CLOB price history
│   ├── Predict/                  XGBoost / NN inference runners
│   ├── Train-Models/             XGBoost / NN training scripts
│   ├── Process-Data/             Team / odds ingestion + feature engineering
│   ├── Utils/                    Kelly, EV, Backtester, drawdown, alerts
│   └── Polymarket/               Paper trader V2 (live system) + scheduler
│
├── Models/XGBoost_Models/        The trained ML model used in the report
├── Data/                         Schedule + committable backtest CSVs
└── nba_game_snapshots.parquet    10-min Polymarket tick snapshots (microstructure)
```

See **[REPRODUCE.md](REPRODUCE.md)** for the exact command that reproduces each table in the report.

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are required for any of the public data providers used here (Polymarket Gamma, NBA Stats, ESPN, SBR).

To re-run a backtest table from scratch, see `REPRODUCE.md`. To regenerate the underlying CSVs from upstream APIs, the data-pipeline scripts at the project root walk through it end-to-end.

---

## What's intentionally not here

- **Soccer-draw and CBB systems** — separate live trading systems described in the parent project; not part of this report.
- **`Data/TeamData.sqlite` / `Data/OddsData.sqlite`** — the team-stat and historical-odds databases used to *train* the XGBoost model. They are large (~hundreds of MB) and rebuildable with `python -m src.Process-Data.Get_Data --backfill` and `python -m src.Process-Data.Get_Odds_Data --backfill`. The backtest CSVs already shipped in `Data/backtest/` contain the model predictions, so the report's tables can be reproduced without rebuilding the DBs.
- **Live paper-trading state files** (`Data/paper_trading*/bankroll.json` etc.) — generated on first `--init` run.

---

## Author

Isaac Vergara (`isaacvergaram@hotmail.com`) — Duke University.
Strategy research supervised by **Dr. Hanchao Yang** (Duke MIDS; Kalshi trader).
