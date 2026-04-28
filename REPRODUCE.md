# Reproducing the report

All commands run from the repo root after `pip install -r requirements.txt`.

---

## Verifiability

### Fully reproducible from shipped data

| Element | Script |
|---|---|
| Predictive accuracy benchmark (XGBoost 71.8% / Polymarket 71.2% / ESPN 67.0%) | `backtest_simulation.py` |
| Stage 1–4 backtest tables | `backtest_simulation.py` |
| Sizing comparison | `sizing_comparison.py` |
| Flip strategy + 47.9% Q1-leader figure (15,639 games) | `flip_backtest.py` |
| Halftime stop-loss audit (89% true-loser rate) | `backtest_simulation.py` Stage 3 |
| Monthly FAV / DOG decomposition | `monthly_decomposition.py` |
| Tick coefficient table (`momentum_2h −0.43`, `momentum_30m +0.25`) | `tick_features.py` → `enriched_filter.py` |
| Mispricing-filter OOS validation | `mispricing_filter.py` |
| Leg-weight bootstrap CI [0, 1] | `leg_weight_optimization.py` |

### Two gaps

**Live paper-trading P&L (V1, V2).** Runtime artifacts of a live session, not derivable from any script. The verifiable claim is that the backtest predicts the live result; the backtest itself is fully reproducible above.

**`retro_backtest_late_season.py`.** Calls the live Polymarket CLOB price-history endpoint, which retains only 30 days of tick data. Reruns past that window will return empty for the earliest dates and produce a slightly different bet count. Upstream API limitation, not a script bug.

---

## Commands

### Stage 1–4 backtest + accuracy benchmark

```bash
python backtest_simulation.py
python sizing_comparison.py
```

### Flip strategy

```bash
python build_historical_backtest.py
python flip_backtest.py
```

### Tick microstructure + filters

```bash
python tick_features.py
python enriched_filter.py
python mispricing_filter.py
```

### Monthly decomposition

```bash
python monthly_decomposition.py
```

### Late-season retro

```bash
python retro_backtest_late_season.py
```

### Leg-weight optimization

```bash
python leg_weight_optimization.py
```

### Backtest dataset (regenerate from upstream APIs)

The CSVs under `Data/backtest/` are already shipped. To rebuild:

```bash
python build_backtest_dataset.py
python build_espn_wp.py
python fetch_q1_scores.py
python build_price_history.py
```

### Retrain the XGBoost model (optional)

The trained model is shipped at `Models/XGBoost_Models/`. To retrain from raw inputs (requires rebuilding `Data/TeamData.sqlite` + `Data/OddsData.sqlite`, ~hundreds of MB):

```bash
python -m src.Process-Data.Get_Odds_Data --backfill
python -m src.Process-Data.Get_Data --backfill
python -m src.Process-Data.Create_Games
cd src/Train-Models
python -m XGBoost_Model_ML --dataset dataset_2012-26 --trials 100 --splits 5 --calibration sigmoid
```
