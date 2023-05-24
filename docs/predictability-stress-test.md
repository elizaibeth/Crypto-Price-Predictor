# Stress test: can this project produce a useful Bitcoin predictor?

Date: 2026-09-21. Conclusion: **no demonstrated robust edge from the current
daily-close-only models; insufficient evidence to claim useful prediction is impossible.**

## What was tested

Freeze the strongest one-day validation challenger (`trees_7_shrunk`):
gradient-boosted trees, 14 return lags, 730 training samples and 0.25 shrinkage.
Evaluate its sensitivity to refit frequency and historical period. This is a
retrospective diagnostic using already explored history, not a fresh holdout
or a change to the production selection rules.

## Retraining frequency

Validation period: 2022-09-21 through 2024-09-19. Positive numbers mean lower MAE
than persistence. Choosing the best cadence after seeing this table would be
additional tuning, not an independent result.

| Refit every | MAE improvement |
| --- | ---: |
| 1 days | +0.081% |
| 7 days | +0.127% |
| 30 days | +0.097% |
| 90 days | +0.137% |

Daily refits do not rescue this particular candidate.

## Historical periods

| Target period (end exclusive) | MAE improvement |
| --- | ---: |
| 2018-09-21 to 2020-09-21 | -0.120% |
| 2020-09-21 to 2022-09-21 | -0.117% |
| 2022-09-21 to 2024-09-20 | +0.097% |
| 2024-09-20 to 2026-09-21 | -0.097% |

The tiny benefit in the selection period reverses in the other three periods.
This supports rejecting this candidate, not all possible models.

## Uncertainty around the apparent edge

Paired circular moving-block bootstrap of daily absolute errors, seed 42,
2,000 resamples. Both baseline and challenger use identical resampled dates.
The descriptive 2.5th–97.5th percentile ranges for validation MAE improvement are:

- 30-day blocks: -0.207% to +0.397%.
- 90-day blocks: -0.213% to +0.327%.

Both include zero. These are post-selection descriptive intervals: the model
was chosen on this same validation period, and the intervals do not correct
for searching multiple candidates. They are not confirmatory significance tests
or bounds on future performance. Block length and market nonstationarity matter.

## Why impossibility would be an overclaim

- Fifteen configurations are not fifteen independent information sources. They
  are largely transformations of the same daily closing prices.
- The search excludes volume, order books, derivatives, macroeconomic releases
  and other candidate information. Whether any adds value is untested.
- The automatic search did not retrain/tune a return-based LSTM. Its failure
  cannot be inferred from the original raw-price LSTM result.
- Only one main validation era selected models. Three horizons use overlapping
  multi-day returns, which reduces independence of the errors.
- Models fit log-return losses, while selection uses dollar-price MAE. These
  objectives are not identical; price levels can change which periods dominate.
- “Good” depends on the task. A 28-day mean-return strategy reduced validation
  RMSE by 1.584% but worsened MAE. That is exploratory evidence of metric
  sensitivity, not a qualified model or a reason to change the objective after
  seeing the result.
- The pipeline accepts learned models on predictable synthetic data, so a flat
  forecast is not simply hard-coded for every input. Synthetic success is only
  a positive control, not market evidence.

## What would change the conclusion?

First choose the product objective: point-price accuracy, probability of an
up/down move, calibrated price ranges, or volatility. These require different
losses, baselines and acceptance criteria. A point forecast need not move away
from the latest price to accompany a useful, calibrated uncertainty range.

For the current point-price objective, predeclare a bounded experiment with
a genuinely new information source; enforce point-in-time availability, data
revision handling and no future-feature leakage. Freeze its model and selection
policy, then log forecasts prospectively and score only after targets mature.
Require an improvement that is both useful and stable, with uncertainty that
accounts for dependent errors and model selection. Forecast error alone does
not establish trading profitability.

Do not keep expanding the historical search until something passes. That would
increase the risk of fitting noise. The defensible project claim is currently
“we built a reproducible system that can reject models without a robust edge.”

## Reproduce

Requires the original local report and matching CSV snapshot:

```sh
python -m scripts.stress_test
```

Inputs: `artifacts/strategy-search-final.json` and
`artifacts/strategy-search-final.prices.csv`. Output: `artifacts/stress-test.json`.
No network access, model release changes or updates to the cached market data
are performed. Artifacts are local and not committed.
