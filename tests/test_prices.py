import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from crypto_predictor.prices import DataUnavailable, load_prices


def prices(start='2024-01-01', end='2024-01-10', value=100):
    dates = pd.date_range(start, end, tz='UTC')
    return pd.DataFrame({'Date': dates, 'Close': [float(value)] * len(dates)})


def bars(start='2024-01-01', end='2024-01-10', value=100):
    dates = pd.date_range(start, end, tz='UTC')
    close = pd.Series(float(value), index=dates)
    return pd.DataFrame({'Date': dates, 'Open': close * .99, 'High': close * 1.02,
                         'Low': close * .98, 'Close': close, 'Volume': 1000.})


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'prices.sqlite3'

    def load(self, downloader, **kwargs):
        return load_prices('BTC-USD', self.path, today='2024-01-11',
                           downloader=downloader, progress=lambda _: None, **kwargs)

    def seed(self):
        return self.load(Mock(return_value=prices()))

    def test_initial_load_and_incremental_overlap_updates_without_duplicates(self):
        fetch = Mock(return_value=prices())
        self.load(fetch)
        fetch.assert_called_once_with('BTC-USD', '2012-01-11', '2024-01-11')
        update = Mock(return_value=prices('2024-01-04', '2024-01-12', value=200))
        frame, metadata = load_prices('btc-usd', self.path, today='2024-01-13', downloader=update,
                                      progress=lambda _: None)
        update.assert_called_once_with('BTC-USD', '2024-01-04', '2024-01-13')
        self.assertEqual(len(frame), 12)
        self.assertEqual(frame['Close'].iloc[0], 100)
        self.assertEqual(frame['Close'].iloc[3], 200)
        self.assertEqual(metadata['as_of'], '2024-01-12')
        self.assertEqual(metadata['missing_recent_days'], 0)

    def test_ohlcv_is_persisted_and_advertised_as_feature_context(self):
        frame, metadata = self.load(Mock(return_value=bars()))
        self.assertEqual(metadata['feature_context'], 'ohlcv')
        self.assertEqual(list(frame.columns), ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT volume FROM prices LIMIT 1').fetchone()[0], 1000.)

    def test_today_is_excluded(self):
        frame, _ = self.load(Mock(return_value=prices(end='2024-01-11')))
        self.assertEqual(len(frame), 10)

    def test_provider_failure_uses_cache_and_exposes_age(self):
        self.seed()
        messages = []
        frame, metadata = load_prices('BTC-USD', self.path, today='2024-01-15',
            downloader=Mock(side_effect=DataUnavailable('rate-limited')), progress=messages.append)
        self.assertEqual(len(frame), 10)
        self.assertEqual(metadata['mode'], 'cache_fallback')
        self.assertEqual(metadata['missing_recent_days'], 4)
        self.assertTrue(any('2024-01-10' in message for message in messages))

    def test_offline_never_downloads(self):
        self.seed()
        fetch = Mock(side_effect=AssertionError('Network access'))
        _, metadata = self.load(fetch, offline=True)
        fetch.assert_not_called()
        self.assertEqual(metadata['mode'], 'offline')

    def test_no_cache_fails_clearly(self):
        with self.assertRaisesRegex(DataUnavailable, 'No cached data'):
            self.load(Mock(side_effect=DataUnavailable('rate-limited')))
        with self.assertRaisesRegex(ValueError, 'No cached prices'):
            self.load(Mock(), offline=True)

    def test_gap_is_requested_and_repaired(self):
        self.seed()
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM prices WHERE date='2024-01-02'")
        fetch = Mock(return_value=prices('2024-01-02'))
        frame, _ = self.load(fetch)
        fetch.assert_called_once_with('BTC-USD', '2024-01-02', '2024-01-11')
        self.assertEqual(len(frame), 10)

    def test_unrepaired_gap_rejects_merge_and_preserves_cache(self):
        self.seed()
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM prices WHERE date='2024-01-02'")
        with self.assertRaisesRegex(ValueError, 'gaps'):
            self.load(Mock(return_value=prices('2024-01-04', value=200)))
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT MAX(close) FROM prices').fetchone()[0], 100)
        with self.assertRaisesRegex(ValueError, 'gaps'):
            self.load(Mock(side_effect=DataUnavailable('offline')))

    def test_full_refresh_requests_history_and_updates_old_correction(self):
        self.seed()
        fetch = Mock(return_value=prices(value=300))
        frame, _ = self.load(fetch, full_refresh=True)
        fetch.assert_called_once_with('BTC-USD', '2012-01-11', '2024-01-11')
        self.assertEqual(frame['Close'].iloc[0], 300)

    def test_tickers_are_isolated(self):
        self.seed()
        frame, _ = load_prices('ETH-USD', self.path, today='2024-01-11',
                              downloader=Mock(return_value=prices(value=50)), progress=lambda _: None)
        self.assertEqual(frame['Close'].iloc[0], 50)
        bitcoin, _ = self.load(Mock(), offline=True)
        self.assertEqual(bitcoin['Close'].iloc[0], 100)

    def test_invalid_update_rolls_back(self):
        self.seed()
        with self.assertRaises(ValueError):
            self.load(Mock(return_value=prices('2024-01-04', value=-1)))
        frame, _ = self.load(Mock(), offline=True)
        self.assertEqual(frame['Close'].iloc[-1], 100)

    def test_conflicting_flags_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.load(Mock(), full_refresh=True, offline=True)
