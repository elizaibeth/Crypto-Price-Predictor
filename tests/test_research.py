import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_predictor.research import (
    Recipe, fit_predict, make_features, run, score, validate_bars,
)


def bars(n=600):
    close = 100*np.exp(np.arange(n)*0.001 + np.sin(np.arange(n))*.01)
    return pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n, tz='UTC'),
                         'Open': close*.999, 'High': close*1.02, 'Low': close*.98,
                         'Close': close, 'Volume': 1000+np.arange(n)%13})


class ResearchTests(unittest.TestCase):
    def test_features_do_not_see_future_bars(self):
        frame = bars()
        changed = frame.copy()
        changed.loc[301:, ['Open','High','Low','Close','Volume']] *= 2
        for name, matrix in make_features(frame).items():
            np.testing.assert_allclose(matrix[:301], make_features(changed)[name][:301], equal_nan=True)
            self.assertTrue(np.isfinite(matrix[90:]).all())

    def test_fitted_prediction_ignores_unobserved_prices_and_volume(self):
        frame = bars()
        changed = frame.copy()
        changed.loc[301:, ['Open','High','Low','Close','Volume']] *= 2
        result = [fit_predict(f.Close.to_numpy(), make_features(f)['all'], 300,
                              np.array([300]), Recipe('test','ridge')) for f in (frame,changed)]
        np.testing.assert_allclose(result[0], result[1])

    def test_ten_iteration_limit_and_only_one_final_challenger(self):
        calls = []
        def baseline(prices, features, recipe, start, stop, seed):
            calls.append((len(prices), start, stop))
            return prices[start-1:stop-1]
        with tempfile.TemporaryDirectory() as directory, \
             patch('crypto_predictor.research.walk_forward', side_effect=baseline):
            report = run(bars(), output=Path(directory)/'report.json', progress=lambda *a, **kw: None)
        self.assertEqual(len(report['iterations']), 10)
        self.assertEqual(calls[:9], [(480,240,480)]*9)
        self.assertEqual(calls[9:], [(600,480,600)])
        self.assertFalse(report['qualified'])
        self.assertFalse(report['production_policy_changed'])
        self.assertEqual(report['status'], 'completed')

    def test_iteration_limit_is_enforced(self):
        for limit in (0,11):
            with self.assertRaises(ValueError):
                run(bars(), max_loops=limit)

    def test_gate_requires_gain_and_consistency(self):
        p = np.arange(1,401,dtype=float)
        perfect = score(p, p[200:400], 200,400)
        self.assertTrue(perfect['passes'])
        baseline = score(p, p[199:399], 200,400)
        self.assertFalse(baseline['passes'])

    def test_invalid_bars_rejected(self):
        frame = bars()
        for column,value in [('Volume',-1),('High',1),('Low',1e9),('Open',np.nan)]:
            changed=frame.copy()
            changed.loc[0,column]=value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_bars(changed)
        with self.assertRaises(ValueError):
            validate_bars(frame.drop(index=5))
