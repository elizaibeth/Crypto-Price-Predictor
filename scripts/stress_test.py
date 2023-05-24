"""Retrospective diagnostic, not a new model-selection or release procedure.

Run from repository root: python -m scripts.stress_test
The challenger is frozen to the best one-day validation candidate in the recorded
search. These diagnostics must not be represented as an untouched test set.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from crypto_predictor.pipeline import validate_prices
from crypto_predictor.search import Strategy, backtest


def bootstrap_gain(actual, forecast, baseline, block, seed=42, repeats=2000):
    """Paired circular moving-block bootstrap; descriptive, post-selection interval."""
    model_error = np.abs(actual - forecast)
    baseline_error = np.abs(actual - baseline)
    rng = np.random.default_rng(seed)
    n = len(actual)
    gains = []
    for _ in range(repeats):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        gains.append(100 * (1 - model_error[idx].mean() / baseline_error[idx].mean()))
    return np.quantile(gains, [0.025, 0.975]).tolist()


def main():
    report = json.loads(Path('artifacts/strategy-search-final.json').read_text())
    frame = validate_prices(pd.read_csv('artifacts/strategy-search-final.prices.csv'))
    p = frame.Close.to_numpy()
    best = min((r for r in report['selection']['1']['leaderboard']
                if r['strategy']['kind'] != 'persistence'), key=lambda r: r['metrics']['mae'])
    strategy = Strategy(**best['strategy'])
    start, stop = int(len(p) * .6), int(len(p) * .8)
    result = {'challenger': best['strategy'], 'cadence': [], 'periods': [], 'intervals': {}}
    with threadpool_limits(limits=1):
        for cadence in (1, 7, 30, 90):
            print('Validation refit cadence', cadence, flush=True)
            run = backtest(p[:stop], strategy, 1, start, stop, refit_days=cadence)
            baseline = np.abs(p[start:stop] - p[start-1:stop-1]).mean()
            result['cadence'].append({'refit_days': cadence, 'mae': run['metrics']['mae'],
                                     'baseline_mae': baseline,
                                     'improvement_percent': 100 * (1-run['metrics']['mae']/baseline)})
            if cadence == 30:
                for block in (30, 90):
                    result['intervals'][str(block)] = bootstrap_gain(p[start:stop], run['predictions'],
                                                                    p[start-1:stop-1], block)
        for begin, end in (('2018-09-21', '2020-09-21'), ('2020-09-21', '2022-09-21'),
                           ('2022-09-21', '2024-09-20'), ('2024-09-20', '2026-09-21')):
            a = int(frame.Date.searchsorted(pd.Timestamp(begin, tz='UTC')))
            b = int(frame.Date.searchsorted(pd.Timestamp(end, tz='UTC')))
            print('Retrospective period', begin, end, flush=True)
            run = backtest(p[:b], strategy, 1, a, b)
            baseline = np.abs(p[a:b] - p[a-1:b-1]).mean()
            result['periods'].append({'start': begin, 'end_exclusive': end,
                                     'mae': run['metrics']['mae'], 'baseline_mae': baseline,
                                     'improvement_percent': 100 * (1-run['metrics']['mae']/baseline)})
    Path('artifacts/stress-test.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
