# Ten-iteration Bitcoin research result

Run date: 2026-09-21. **Completed all 10 iterations; no candidate qualified.**
The existing production persistence policy was not changed.

## Fixed objective and information

- Objective: next-day dollar MAE, at least 5% below persistence.
- Consistency: wins in at least 60% of 30-day blocks and positive improvement
  in both halves of validation and final evaluation.
- Final uncertainty gate: positive lower endpoints of both descriptive
  30- and 90-day paired block-bootstrap intervals.
- New inputs: daily open, high, low and volume, in addition to closing prices.
  These are additional fields from Yahoo, not an independent market-data provider.
- All features use completed candles available at each forecast origin.
- Nine fixed model recipes, then a tenth ensemble of the three best validation
  recipes, shrunk halfway toward persistence.

Dataset: 3652 daily bars, 2016-09-21 through 2026-09-20.
Normalized input SHA-256: `1bfedfe66e24f4b566b76edeacc0233201bf5cb7d45a0c2045f102a6b20fe86b`.
Validation starts 2020-09-20; final evaluation starts 2024-09-20.
The first 40% provides initial history; the next 40% validates candidates; the
final 20% diagnoses one frozen validation winner. Refits occur every 30 target
days with up to 1,095 mature labels. Seed: 42.

## Iteration log

Positive improvement means lower MAE than persistence. The validation baseline
MAE was $885.58.

| Iteration | Recipe | Validation MAE | Improvement | Blocks won |
| --- | --- | ---: | ---: | ---: |
| 1 | ridge_returns_control | $893.02 | -0.84% | 32.7% |
| 2 | ridge_volume_and_range | $896.49 | -1.23% | 28.6% |
| 3 | weighted_median_regression | $911.45 | -2.92% | 30.6% |
| 4 | trees_returns_control | $896.78 | -1.27% | 38.8% |
| 5 | trees_range | $894.33 | -0.99% | 42.9% |
| 6 | trees_volume | $895.88 | -1.16% | 40.8% |
| 7 | trees_volume_and_range | $896.50 | -1.23% | 36.7% |
| 8 | trees_recent_regime | $901.38 | -1.78% | 36.7% |
| 9 | extra_trees_volume_and_range | $884.13 | +0.16% | 57.1% |
| 10 | top_three_shrunk_ensemble | $884.80 | +0.09% | 49.0% |

Recipes 1–2 use standardized ridge regression (alpha 1000); recipe 3 uses
standardized median regression with alpha 0.0001. Recipes 4–8 use 80 rounds of
absolute-error histogram boosting, 7 leaves, minimum 40 samples per leaf,
learning rate 0.04 and L2 regularization 10. Recipe 8 additionally weights recent
history with a 180-day half-life. Recipe 9 uses 64 extra trees, depth 5, minimum
30 samples per leaf and absolute-error splits. All use origin-price weights;
for absolute simple-return loss these align fitting with dollar-price MAE.

## Final diagnostic

Frozen candidate: `extra_trees_volume_and_range`. It failed validation qualification,
but was evaluated once to diagnose whether its tiny apparent benefit survived.
No other candidates were subsequently evaluated or selected from the final period.

| Metric | Challenger | Persistence |
| --- | ---: | ---: |
| MAE | $1,422.21 | $1,418.23 |
| RMSE | $2,000.01 | $2,001.74 |

MAE improvement: -0.280%; blocks won: 56%.
The small RMSE improvement does not satisfy the predeclared MAE objective.

Descriptive bootstrap ranges for MAE improvement (2,000 samples):

- 30-day blocks: -0.879% to +0.312%.
- 90-day blocks: -0.848% to +0.254%.

Both ranges include zero. These retrospective, dependent-data intervals are not
independent proof of predictive skill and do not account for all prior research
on this dataset. The final period was already seen in previous experiments.

## Decision

Stop at the requested ten-iteration budget. Do not lower the threshold, switch
to RMSE after seeing the results, or promote a model merely because it emits
changing prices. The default remains persistence.

The result is negative for this bounded set of next-day, daily-OHLCV strategies.
It does not prove that Bitcoin has no predictable properties. Point-price
forecasting, volatility forecasting and calibrated prediction intervals are
different objectives and would require separate experiments. No fees, slippage
or trading returns were evaluated.

## Reproduce

```sh
docker compose run --build --rm research
```

The default reuses `artifacts/research/ohlcv.csv`; only a missing snapshot or
`--refresh-data` causes a download. Use the original matching CSV/hash for exact
data reproduction. Live provider revisions can change results. The input copy
and full report are saved as `artifacts/research/report.bars.csv` and
`artifacts/research/report.json`. They are local artifacts, not committed.

The runner enforces 1–10 iterations and checkpoints results. No additional
real-data iterations were run after this result. Unit tests use synthetic data
and mocks to verify future-feature isolation, training boundaries, the loop cap
and the single final-candidate evaluation.
