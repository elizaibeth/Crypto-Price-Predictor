import unittest
import numpy as np
import pandas as pd
from crypto_predictor.pipeline import experiment, prepare, validate_prices


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'Date': pd.date_range('2024-01-01', periods=50),
                                   'Close': np.arange(1, 51, dtype=float)})

    def test_baseline_metrics_and_forecast(self):
        report = experiment(self.frame, lookback=5, horizon=3)
        self.assertEqual(report['evaluation']['model'], {'mae': 1.0, 'rmse': 1.0})
        self.assertEqual(len(report['holdout']), 10)
        self.assertEqual(report['holdout'][0]['actual'], 41)
        self.assertEqual(report['holdout'][0]['prediction'], 40)
        self.assertEqual([r['close'] for r in report['forecast']], [50.0] * 3)
        self.assertEqual(report['forecast'][0]['date'], '2024-02-20')

    def test_scaler_never_fits_holdout_and_windows_align(self):
        values = np.array([1, 2, 3, 4, 5, 100, 200], dtype=float)[:, None]
        scaler, x, y, test = prepare(values, lookback=2, split=5)
        self.assertEqual(scaler.data_max_[0], 5)
        np.testing.assert_allclose(test[0, :, 0], [0.75, 1.0])
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(len(test), 2)
        self.assertGreater(test[1, -1, 0], 1)

    def test_invalid_prices_rejected(self):
        for value in [np.nan, np.inf, 0, -1]:
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame.loc[0, 'Close'] = value
                with self.assertRaises(ValueError):
                    validate_prices(frame)

    def test_dates_and_missing_columns_rejected(self):
        for frame in [self.frame.drop(index=3),
                      pd.concat([self.frame, self.frame.iloc[:1]]),
                      self.frame.drop(columns='Close')]:
            with self.assertRaises(ValueError):
                validate_prices(frame)

    def test_boundary_and_config_rejected(self):
        for kwargs in [{'lookback': 40}, {'horizon': 0}, {'epochs': 0}, {'model': 'unknown'}]:
            with self.assertRaises(ValueError):
                experiment(self.frame, **kwargs)

    def test_sorting_and_fingerprint_stability(self):
        self.assertEqual(experiment(self.frame, lookback=5),
                         experiment(self.frame.iloc[::-1], lookback=5))
