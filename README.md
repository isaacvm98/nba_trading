# NBA Polymarket Dual-Leg Strategy — Code & Data

## Abstract

An XGBoost classifier on 2012–2025 NBA team data scored 71.8% on the 2025-26 holdout, edging out Polymarket pregame (71.2%) and ESPN (67.0%). That accuracy alone is not a strategy: naive Kelly across 314 model-recommended bets lost 25–100% under every sizing method, because 75% of those bets were underdogs and the unfiltered win rate was 34%.

Filtering to model edge ≥ 7% and confidence ≥ 60% cut the universe to 24 bets at 66.7% win rate and +50.9% return. A dual-leg routing rule — favorites (conf ≥ 60%) held to resolution, underdogs (entry ≥ $0.30) actively managed with ESPN Win Probability exits — expanded the tradeable set to 76 bets at +39.8% return, 8.4% max drawdown, Sharpe 5.50.

### Key learnings

- **Filters are the primary alpha, not model complexity.** The Stage 1 → Stage 2 jump from −25.6% to +50.9% comes entirely from filtering. No ensemble, neural net, or Bayesian update is needed.
- **Conviction-based exits beat uniform exits.** Applying ESPN WP stop-loss to every bet *reduces* returns by cutting favorites that would have resolved as winners. Routing favorites to hold-to-resolution and underdogs to active management captures both alpha sources.
- **The flip-strategy intuition is wrong.** Across 15,639 historical games, underdogs leading at Q1 go on to win 47.9% of the time, not the 25% a "regression to favorite" narrative assumes. Selling the underdog at the Q1 mark is correct; flipping to buy the favorite is not.
- **Microstructure has sign-dependent momentum.** Slow 2-hour price grinds toward our side predict losses (edge is already priced in); sharp 30-minute moves predict wins (fresh sharp money). Coefficients: −0.43 vs +0.25.
- **Regime shifts kill filtered strategies.** Live performance degraded from mid-March onward as star rest, tanking, and post-trade-deadline lineups broke the model's season-to-date features. Safe operating window: ~Nov 20 → Mar 10.

Live forward tests confirmed both legs of the simulation: V1 (unfiltered Kelly) returned −96.4% on 181 positions, mapping to the simulated −100% Full Kelly; V2 (filtered + dual-leg) returned −1.0% on 22 closed positions and drifted to −17.8% on 25 as the FAV leg collapsed in late March.

---

## Layout

```
.
├── config.toml                   Season date ranges
├── requirements.txt
│
├── build_*.py / fetch_*.py       Polymarket, ESPN WP, Q1 scores, history
├── backtest_simulation.py        Stage 1–4 backtest
├── sizing_comparison.py          Flat / Half-Kelly / Edge-scaled comparison
├── flip_backtest.py              Flip-strategy test
├── monthly_decomposition.py      Monthly FAV / DOG performance
├── mispricing_filter.py          Logistic-regression filter on model_edge
├── enriched_filter.py            12-feature filter using tick microstructure
├── tick_features.py              10-min PM tick → microstructure features
├── leg_weight_optimization.py    FAV / DOG Kelly + risk-parity + mean-variance
├── retro_backtest_late_season.py Late-season FAV-only retro
├── backfill_market_context.py    Retro feature enrichment for live trades
│
├── src/
│   ├── DataProviders/            Polymarket Gamma, ESPN, NBA, SBR, CLOB
│   ├── Predict/                  XGBoost / NN inference runners
│   ├── Train-Models/             XGBoost / NN training scripts
│   ├── Process-Data/             Team / odds ingestion + features
│   ├── Utils/                    Kelly, EV, drawdown, alerts
│   └── Polymarket/               Paper trader V2 + scheduler
│
├── Models/XGBoost_Models/        Trained 69.8% ML model
├── Data/                         Schedule + backtest CSVs
└── nba_game_snapshots.parquet    10-min Polymarket tick snapshots
```

See [REPRODUCE.md](REPRODUCE.md) for the command that reproduces each table.

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

No API keys required.
