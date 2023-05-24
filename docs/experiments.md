# Recorded strategy comparison

Run date: 2026-09-21. This is a retrospective experiment, not an independent
live forecast validation. No final-period challenger search was performed.

## Data and protocol

- Source snapshot: Yahoo Finance BTC-USD; 3,652 daily closes.
- Dates: 2016-09-21 through 2026-09-20.
- Normalized CSV SHA-256: `8a836b5ad2a420b6ea6656847fb49eb43c2add90b788b913967ae763162832ff`.
- Validation: 2022-09-21 through 2024-09-19.
- Final evaluation: 2024-09-20 through 2026-09-20.
- Refits every 30 target dates, using only labels available at each origin.
- Direct 1-, 7- and 28-day log-return targets; MAE selects the model.
- Selection gates: at least 1% lower validation MAE and wins in 60% of blocks.
- Release gate: selected challenger must improve final MAE by at least 1%.

## Search iterations

The first pass compared 11 configurations: persistence, median/mean returns,
ridge regression and gradient-boosted trees, including shrinkage toward zero return.
The best one-day challenger improved validation MAE by only about 0.1%.
A second pass added four shorter-history configurations (30-day median, 90-day
mean, 365-day trees and 90-day ridge), based on validation results. The final
search contains 15 candidates per horizon. No challenger cleared the gates.

The 1% threshold was not relaxed after seeing the results. The final period was
already visible in the original LSTM experiment; it cannot be described as a
previously unseen test set. Further tuning would require fresh evaluation data.

## Validation results

| Horizon | Strategy | MAE (USD) | RMSE (USD) | Blocks beating baseline |
| --- | --- | ---: | ---: | ---: |
| 1 days | trees_7_shrunk | 718.49 | 1163.54 | 60% |
| 1 days | mean_365_shrunk | 719.09 | 1162.16 | 40% |
| 1 days | persistence | 719.19 | 1162.34 | 0% |
| 1 days | mean_90_shrunk | 719.35 | 1162.15 | 44% |
| 1 days | median_365 | 720.22 | 1163.74 | 44% |
| 1 days | trees_365_shrunk | 720.85 | 1165.38 | 44% |
| 1 days | median_90 | 720.88 | 1164.67 | 44% |
| 1 days | trees_15_1095 | 720.92 | 1172.17 | 44% |
| 1 days | trees_7_730 | 721.10 | 1172.61 | 52% |
| 1 days | ridge_7_90_shrunk | 721.39 | 1164.11 | 28% |
| 1 days | ridge_14_shrunk | 722.48 | 1166.38 | 40% |
| 1 days | median_30 | 723.62 | 1165.27 | 40% |
| 1 days | ridge_14_1095 | 723.78 | 1164.59 | 40% |
| 1 days | ridge_7_365 | 727.74 | 1170.25 | 32% |
| 1 days | ridge_30_730 | 752.70 | 1195.43 | 16% |
| 7 days | persistence | 1980.73 | 2952.58 | 0% |
| 7 days | mean_365_shrunk | 1986.35 | 2955.35 | 48% |
| 7 days | trees_7_shrunk | 1989.58 | 2954.20 | 40% |
| 7 days | mean_90_shrunk | 1992.27 | 2953.46 | 28% |
| 7 days | ridge_14_shrunk | 1995.25 | 2968.04 | 28% |
| 7 days | ridge_7_90_shrunk | 2004.15 | 2973.62 | 40% |
| 7 days | trees_365_shrunk | 2006.00 | 2985.95 | 28% |
| 7 days | ridge_14_1095 | 2009.08 | 2975.69 | 40% |
| 7 days | median_365 | 2010.07 | 2977.33 | 48% |
| 7 days | median_90 | 2028.49 | 3008.97 | 52% |
| 7 days | ridge_30_730 | 2061.07 | 2996.79 | 40% |
| 7 days | ridge_7_365 | 2094.88 | 3060.64 | 32% |
| 7 days | trees_15_1095 | 2097.33 | 3004.60 | 32% |
| 7 days | trees_7_730 | 2116.36 | 3021.58 | 28% |
| 7 days | median_30 | 2337.97 | 3448.96 | 32% |
| 28 days | persistence | 3965.03 | 5656.35 | 0% |
| 28 days | mean_365_shrunk | 4000.20 | 5627.55 | 52% |
| 28 days | ridge_14_1095 | 4004.02 | 5679.70 | 56% |
| 28 days | mean_90_shrunk | 4009.69 | 5566.75 | 48% |
| 28 days | ridge_7_90_shrunk | 4010.58 | 5602.44 | 44% |
| 28 days | trees_7_shrunk | 4038.68 | 5704.09 | 36% |
| 28 days | ridge_14_shrunk | 4042.33 | 5745.74 | 44% |
| 28 days | trees_365_shrunk | 4090.12 | 5676.40 | 44% |
| 28 days | median_365 | 4173.02 | 5676.71 | 40% |
| 28 days | ridge_30_730 | 4357.49 | 6197.77 | 44% |
| 28 days | trees_15_1095 | 4459.36 | 6146.47 | 28% |
| 28 days | median_90 | 4536.33 | 6145.28 | 52% |
| 28 days | ridge_7_365 | 4603.73 | 6215.60 | 36% |
| 28 days | trees_7_730 | 4644.84 | 6218.51 | 20% |
| 28 days | median_30 | 5434.19 | 8590.51 | 40% |

## Final evaluation of the selected strategies

| Horizon | Selected strategy | MAE (USD) | RMSE (USD) |
| --- | --- | ---: | ---: |
| 1 days | persistence | 1418.23 | 2001.74 |
| 7 days | persistence | 3808.02 | 5111.61 |
| 28 days | persistence | 8275.49 | 10916.35 |

The earlier raw-price LSTM had one-day MAE $7,144.00
and RMSE $9,060.74 on the same final dates.
Selecting persistence avoids that model's larger error. It does not establish a
new predictive advantage: the selected strategy is the baseline itself.

The forecasts are flat at the latest observed close because no challenger
qualified. This expresses the fallback policy, not confidence that price will
remain unchanged. Multi-day errors overlap and are not independent observations.

## Reproduce

Use the saved CSV; a fresh Yahoo download can change the date range or values.

```sh
docker compose run --rm predictor --csv artifacts/btc.prices.csv --model auto \
  --output artifacts/replay-auto.json
```

The source CSV and full JSON reports are local artifacts and are not committed.
For another checkout, supply a matching CSV and verify the hash. The synthetic
fixture only verifies execution. The full report includes every candidate setting,
fold metric, selected strategy, holdout prediction and runtime version.

Next evidence to collect: freeze this policy, record forecasts as new data
arrives, and assess them after targets mature. Repeated historical search alone
would risk fitting the validation period.
