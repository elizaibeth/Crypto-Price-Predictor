# One-year directional experiment

Run date: 2026-09-21. Downloaded one year of BTC-USD daily closes directly from
Yahoo Finance; retained the existing full SQLite history unchanged.

Input: 365 rows, 2025-09-21 to 2026-09-20.
Input hash: `02a9f83b15fda1470cdb65a161029566a7569f0c07353ea0929610ab589235f4`.
Validation begins 2026-04-28; final test begins 2026-07-10.
The classifier, 65% probability threshold, horizon bands and quality gates
were identical to the ten-year run. The data window was the only changed setting.

## One-year results

| Horizon | Validation calls | Final calls / 73 | Final call accuracy | Final prior accuracy on same calls | Forecast |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 days | 1 | 0 | n/a | n/a | inconclusive |
| 7 days | 0 | 11 | 63.6% | 63.6% | inconclusive |
| 28 days | 0 | 5 | 60.0% | 0.0% | inconclusive |

The seven-day candidate made 11 final calls and got 7 right; the 28-day
candidate made 5 and got 3 right. Neither passed validation or the minimum
30-call gate. Higher call frequency is not enough to establish useful accuracy.

## Comparison caveat

The ordinary 60/20/20 splits give the one-year and ten-year runs different
evaluation lengths. Their aggregate accuracies are not an apples-to-apples
comparison. On the common final dates listed below, the ten-year model made
zero calls at every horizon, while the one-year model made the counts above:

- 1-day: 73 common dates, ten-year candidate calls: 0.
- 7-day: 73 common dates, ten-year candidate calls: 0.
- 28-day: 73 common dates, ten-year candidate calls: 0.

This comparison measures a behavior change, not independent evidence of
improvement. Both windows reuse previously explored history. With only one year,
the chronological training/calibration subsets are smaller; some folds lack
enough examples of all three outcomes to fit the calibrated classifier.

## Reproduce

```sh
docker compose run --build --rm predictor --ticker BTC-USD --offline \
  --model direction --output artifacts/btc-one-year.json
```

The CLI now defaults `--history-days` to 365. Or use the downloaded input
`artifacts/btc-one-year.prices.csv` with `--csv`.
The full cache is not truncated. Forecast labels remain inconclusive until
the unchanged confidence and quality gates pass.
