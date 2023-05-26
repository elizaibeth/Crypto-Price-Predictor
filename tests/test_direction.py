import unittest

from crypto_predictor.direction import direction_decision


class DirectionDecisionTests(unittest.TestCase):
    def test_confident_qualified_prediction_is_up(self):
        result = direction_decision({'up': .8, 'down': .1, 'same': .1}, qualified=True)
        self.assertEqual(result['label'], 'up')
        self.assertEqual(result['confidence'], .8)

    def test_unqualified_model_cannot_claim_same_even_with_high_probability(self):
        result = direction_decision({'up': .01, 'down': .01, 'same': .98}, qualified=False)
        self.assertEqual(result['label'], 'inconclusive')
        self.assertEqual(result['reason'], 'model_not_qualified')

    def test_low_confidence_abstains_instead_of_guessing(self):
        result = direction_decision({'up': .4, 'down': .35, 'same': .25}, qualified=True)
        self.assertEqual(result['label'], 'inconclusive')
        self.assertEqual(result['reason'], 'low_confidence')

    def test_invalid_probabilities_and_thresholds_are_rejected(self):
        for probabilities in ({'up': .9}, {'up': -.1, 'down': .8, 'same': .3},
                              {'up': .8, 'down': .8, 'same': .8},
                              {'up': float('nan'), 'down': .5, 'same': .5}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                direction_decision(probabilities, qualified=True)
        for threshold in (.5, 1.1, float('nan')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                direction_decision({'up': .8, 'down': .1, 'same': .1}, qualified=True,
                                   confidence_threshold=threshold)

    def test_qualified_down_and_same_including_confidence_boundary(self):
        for probabilities, expected in [({'up': .1, 'down': .8, 'same': .1}, 'down'),
                                        ({'up': .2, 'down': .15, 'same': .65}, 'same')]:
            self.assertEqual(direction_decision(probabilities, qualified=True)['label'], expected)

    def test_persistence_without_probabilities_is_inconclusive(self):
        result = direction_decision(None, qualified=False)
        self.assertEqual(result['label'], 'inconclusive')
        self.assertIsNone(result['confidence'])
        self.assertIsNone(result['probabilities'])

    def test_same_band_includes_its_boundaries(self):
        from crypto_predictor.direction import movement_label
        for price, expected in [(102, 'up'), (98, 'down'), (100, 'same'), (101, 'same'), (99, 'same')]:
            self.assertEqual(movement_label(100, price, .01), expected)

    def test_invalid_prices_and_bands_are_rejected(self):
        from crypto_predictor.direction import movement_label
        for args in [(0, 100, .01), (100, float('nan'), .01), (100, 100, -1),
                     (100, 100, 1), (100, 100, float('inf'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                movement_label(*args)


class DirectionExperimentTests(unittest.TestCase):
    def test_ohlcv_features_use_completed_candle_without_future_leakage(self):
        import numpy as np
        import pandas as pd
        from crypto_predictor.direction import direction_features
        n = 600
        close = 100 * np.exp(np.arange(n) * .001)
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=n),
                              'Open': close * .999, 'High': close * 1.02,
                              'Low': close * .98, 'Close': close,
                              'Volume': np.arange(n) + 1000.})
        changed = frame.copy()
        changed.loc[301:, ['Open', 'High', 'Low', 'Close', 'Volume']] *= 2
        original, altered = direction_features(frame), direction_features(changed)
        self.assertGreater(original.shape[1], 18)
        np.testing.assert_allclose(original[:301], altered[:301])

    def test_predictable_three_class_series_can_qualify_and_report_coverage(self):
        import numpy as np
        import pandas as pd
        from crypto_predictor.direction import direction_experiment
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                              'Close': np.tile([100., 103., 103., 100.], 150)})
        result = direction_experiment(frame, bands={1: .01})
        forecast = result['forecast'][0]
        self.assertEqual(forecast['label'], 'same')
        self.assertEqual(forecast['flat_band'], .01)
        self.assertTrue(result['evaluation']['horizons']['1']['qualified'])
        self.assertGreaterEqual(result['evaluation']['horizons']['1']['final']['call_accuracy'], .9)
        self.assertGreaterEqual(result['evaluation']['horizons']['1']['final']['call_coverage'], .5)
        self.assertEqual(set(forecast['probabilities']), {'up', 'down', 'same'})
        self.assertAlmostEqual(sum(forecast['probabilities'].values()), 1)

    def test_invalid_data_and_configuration_fail_before_training(self):
        import numpy as np
        import pandas as pd
        from crypto_predictor.direction import direction_experiment
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                              'Close': np.tile([100., 103., 103., 100.], 150)})
        for kwargs in ({'bands': {}}, {'bands': {0: .01}}, {'bands': {1: -.01}},
                       {'confidence_threshold': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                direction_experiment(frame, **kwargs)
        with self.assertRaises(ValueError):
            direction_experiment(frame.iloc[:100])
        with self.assertRaises(ValueError):
            direction_experiment(frame.iloc[:300], bands={90: .1})
        with self.assertRaises(ValueError):
            direction_experiment(frame.drop(index=10))

    def test_insufficient_class_diversity_abstains_without_invented_confidence(self):
        import pandas as pd
        from crypto_predictor.direction import direction_experiment
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600), 'Close': 100.})
        result = direction_experiment(frame)
        for forecast in result['forecast']:
            self.assertEqual(forecast['label'], 'inconclusive')
            self.assertIsNone(forecast['probabilities'])
        self.assertEqual(result['evaluation']['horizons']['1']['final']['call_coverage'], 0)
        self.assertIsNone(result['evaluation']['horizons']['1']['final']['call_accuracy'])

    def test_later_prices_cannot_change_earlier_predictions_or_validation(self):
        import numpy as np
        import pandas as pd
        from crypto_predictor.direction import direction_experiment
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                              'Close': np.tile([100., 103., 103., 100.], 150)})
        original = direction_experiment(frame, bands={1: .01})
        frame.loc[480:, 'Close'] *= 1.2
        changed = direction_experiment(frame, bands={1: .01})
        self.assertEqual(original['evaluation']['horizons']['1']['validation'],
                         changed['evaluation']['horizons']['1']['validation'])
        self.assertEqual(original['holdout']['1'][0]['probabilities'],
                         changed['holdout']['1'][0]['probabilities'])

    def test_legacy_persistence_and_auto_fallback_are_explicitly_inconclusive(self):
        import pandas as pd
        from crypto_predictor.pipeline import experiment
        from crypto_predictor.search import compare, Strategy
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=300), 'Close': 100.})
        for result in [experiment(frame), compare(frame, strategies=(Strategy('persistence'),))]:
            self.assertTrue(all(row['label'] == 'inconclusive' for row in result['forecast']))
            self.assertTrue(all(row['probabilities'] is None for row in result['forecast']))
