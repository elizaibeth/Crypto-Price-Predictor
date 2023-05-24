"""Exercise provider failures at the CLI boundary without hitting Yahoo."""
import contextlib
import io
import sys
import types
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from crypto_predictor.__main__ import main


class RateLimitError(Exception):
    pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.argv = ['crypto_predictor', '--ticker', 'BTC-USD', '--cache',
                     self.directory.name + '/prices.sqlite3']

    def test_rate_limit_becomes_actionable_cli_error(self):
        yf = types.ModuleType('yfinance')
        yf.Ticker = Mock()
        yf.Ticker.return_value.history.side_effect = RateLimitError('Too Many Requests')
        exceptions = types.ModuleType('yfinance.exceptions')
        exceptions.YFRateLimitError = RateLimitError
        stderr = io.StringIO()
        with patch.dict(sys.modules, {'yfinance': yf, 'yfinance.exceptions': exceptions}), \
             patch.object(sys, 'argv', self.argv), \
             contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        self.assertIn('rate-limited', stderr.getvalue())
        self.assertIn('--csv', stderr.getvalue())
        yf.Ticker.return_value.history.assert_called_once()

    def test_empty_download_returns_clear_error(self):
        yf = types.ModuleType('yfinance')
        yf.Ticker = Mock()
        yf.Ticker.return_value.history.return_value = pd.DataFrame()
        exceptions = types.ModuleType('yfinance.exceptions')
        exceptions.YFRateLimitError = RateLimitError
        stderr = io.StringIO()
        with patch.dict(sys.modules, {'yfinance': yf, 'yfinance.exceptions': exceptions}), \
             patch.object(sys, 'argv', self.argv), \
             contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        self.assertIn('returned no prices', stderr.getvalue())
