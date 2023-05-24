import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_predictor.search import (
    Strategy, backtest, compare, features, fit_predict, select_strategy, training_indices,
)


class SearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 240)))
        self.frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=240),
                                   "Close": self.prices})

    def test_features_use_only_observed_returns(self):
        changed = self.prices.copy()
        changed[101:] *= 3
        np.testing.assert_allclose(features(self.prices, 7)[:101],
                                   features(changed, 7)[:101], equal_nan=True)
        np.testing.assert_allclose(features(self.prices, 7)[100, :7],
                                   np.diff(np.log(self.prices[93:101]))[::-1])

    def test_training_targets_mature_before_origin_and_window_is_bounded(self):
        strategy = Strategy("ridge", "ridge", window=40, lags=7)
        for horizon in (1, 7, 28):
            indices = training_indices(150, horizon, strategy)
            self.assertEqual(len(indices), 40)
            self.assertEqual(indices[-1] + horizon, 150)
            self.assertTrue(np.all(indices + horizon <= 150))

    def test_future_mutation_cannot_change_prediction_at_origin(self):
        strategy = Strategy("ridge", "ridge", lags=7)
        changed = self.prices.copy()
        changed[151:] *= 2
        predictions = [fit_predict(p, features(p, 7), 150, np.array([150]), 7, strategy, 42)
                       for p in (self.prices, changed)]
        np.testing.assert_allclose(*predictions)

    def test_scaler_only_sees_training_features(self):
        from sklearn.pipeline import make_pipeline
        strategy = Strategy("ridge", "ridge", window=40, lags=7)
        x = features(self.prices, 7)
        captured = []
        def capture(*steps):
            pipeline = make_pipeline(*steps)
            captured.append(pipeline)
            return pipeline
        with patch("crypto_predictor.search.make_pipeline", side_effect=capture):
            fit_predict(self.prices, x, 150, np.array([150]), 7, strategy, 42)
        rows = training_indices(150, 7, strategy)
        np.testing.assert_allclose(captured[0][0].mean_, x[rows].mean(axis=0))

    def test_successful_challenger_is_refit_for_future_forecast(self):
        self.frame["Close"] = 100 * np.exp(np.arange(240) * 0.001)
        report = compare(self.frame, strategies=(Strategy("persistence"), Strategy("mean", "mean")))
        for forecast in report["forecast"]:
            self.assertEqual(forecast["strategy"], "mean")
            expected = self.frame["Close"].iloc[-1] * np.exp(forecast["horizon_days"] * 0.001)
            self.assertAlmostEqual(forecast["close"], expected)

    def test_failed_release_gate_uses_baseline_without_selecting_another_model(self):
        candidates = (Strategy("persistence"), Strategy("mean", "mean"))
        real_backtest = backtest
        def simulated_shift(prices, strategy, horizon, start, stop, *args):
            run = real_backtest(prices, strategy, horizon, start, stop, *args)
            if strategy.kind == "mean":
                if start == int(len(self.frame) * 0.6):
                    run["metrics"] = {"mae": 0, "rmse": 0}
                    for fold in run["folds"]:
                        fold["model"] = {"mae": 0, "rmse": 0}
                else:
                    run["metrics"] = {"mae": 1e6, "rmse": 1e6}
            return run
        with patch("crypto_predictor.search.backtest", side_effect=simulated_shift):
            report = compare(self.frame, strategies=candidates)
        for result in report["evaluation"]["horizons"].values():
            self.assertEqual(result["selected"], "mean")
            self.assertEqual(result["forecast_strategy"], "persistence")
            self.assertEqual(result["status"], "baseline_fallback")

    def test_baseline_is_horizon_aligned(self):
        p = np.arange(1, 241, dtype=float)
        run = backtest(p, Strategy("persistence"), 7, 180, 240)
        self.assertEqual(run["metrics"]["mae"], 7)
        np.testing.assert_array_equal(run["predictions"], p[173:233])
        self.assertEqual([f["fit_origin_index"] for f in run["folds"]], [173, 203])

    def test_direct_return_model_recovers_constant_growth(self):
        p = 100 * np.exp(np.arange(240) * 0.001)
        run = backtest(p, Strategy("mean", "mean"), 7, 180, 240)
        self.assertLess(run["metrics"]["mae"], 1e-10)

    def test_selection_requires_material_and_consistent_improvement(self):
        def row(name, mae, wins):
            return {"strategy": {"name": name}, "metrics": {"mae": mae},
                    "fold_win_fraction": wins}
        base = row("persistence", 10, 0)
        for challenger in [row("tiny", 9.99, 1), row("inconsistent", 8, 0.5), row("worse", 11, 1)]:
            self.assertEqual(select_strategy([base, challenger])[0], base)
        good = row("good", 9, 0.8)
        self.assertEqual(select_strategy([base, good])[0], good)
        self.assertEqual(select_strategy([row("persistence", 0, 0), row("tie", 0, 1)])[0]["strategy"]["name"], "persistence")

    def test_final_holdout_does_not_change_selection(self):
        candidates = (Strategy("persistence"), Strategy("median", "median", window=90))
        original = compare(self.frame, strategies=candidates)
        changed = self.frame.copy()
        changed.loc[192:, "Close"] *= 2
        altered = compare(changed, strategies=candidates)
        self.assertEqual(original["selection"], altered["selection"])
        self.assertEqual(len(original["forecast"]), 3)
        self.assertEqual([f["horizon_days"] for f in original["forecast"]], [1, 7, 28])

    def test_constant_prices_fall_back_and_remain_json_serializable(self):
        import json
        self.frame["Close"] = 100.0
        report = compare(self.frame, strategies=(Strategy("persistence"), Strategy("median", "median")))
        json.dumps(report, allow_nan=False)
        for row in report["evaluation"]["horizons"].values():
            self.assertEqual(row["forecast_strategy"], "persistence")
            self.assertEqual(row["model"]["mae"], 0)
        self.assertTrue(all(f["close"] == 100 for f in report["forecast"]))

    def test_rejects_short_data_and_invalid_configuration(self):
        with self.assertRaises(ValueError):
            compare(self.frame.iloc[:100])
        for kwargs in ({"horizon": 0}, {"horizon": 91}, {"refit_days": 0}, {"strategies": ()}):
            with self.assertRaises(ValueError):
                compare(self.frame, **kwargs)
