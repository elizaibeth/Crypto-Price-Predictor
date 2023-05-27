"""Exercise the real command and its public output, without internal mocks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class DirectionCLITests(unittest.TestCase):
    def test_missing_source_defaults_to_bitcoin_instead_of_argparse_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([
                sys.executable, '-m', 'crypto_predictor', '--offline',
                '--cache', str(Path(directory)/'empty.sqlite3'), '--history-days', '730'],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertIn('No cached prices for BTC-USD', completed.stderr)
            self.assertNotIn('one of the arguments --csv --ticker is required', completed.stderr)

    def test_saved_forecasts_and_terminal_labels_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                          'Close': np.tile([100.,103.,103.,100.],150)}).to_csv(path/'prices.csv', index=False)
            command = [sys.executable, '-m', 'crypto_predictor', '--csv', str(path/'prices.csv'),
                       '--model', 'direction', '--history-days', '600',
                       '--output', str(path/'report.json')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((path/'report.json').read_text())
            printed = json.loads(completed.stdout.split('Forecasts:\n',1)[1])
            self.assertEqual(printed, report['forecast'])
            self.assertIn('1-day: same', completed.stdout)
            self.assertIn('Feature context: close_only', completed.stdout)
            self.assertIn('call coverage', completed.stdout)
            self.assertEqual(report['forecast'][0]['label'], 'same')
            self.assertEqual(report['forecast'][0]['date'], '2021-08-23')

    def test_invalid_threshold_fails_clearly_without_overwriting_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                          'Close': 100.}).to_csv(path/'prices.csv', index=False)
            (path/'report.json').write_text('previous report')
            completed = subprocess.run([sys.executable, '-m', 'crypto_predictor',
                '--csv', str(path/'prices.csv'), '--model', 'direction',
                '--confidence-threshold', '.2', '--output', str(path/'report.json')],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertIn('Confidence threshold', completed.stderr)
            self.assertNotIn('Traceback', completed.stderr)
            self.assertEqual((path/'report.json').read_text(), 'previous report')

    def test_custom_flat_band_and_missing_data_are_reported_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                          'Close': 100.}).to_csv(path/'prices.csv', index=False)
            command = [sys.executable, '-m', 'crypto_predictor', '--csv', str(path/'prices.csv'),
                       '--model', 'direction', '--flat-band-1d', '.02', '--output', str(path/'report.json')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            r = json.loads((path/'report.json').read_text())
            self.assertEqual(r['forecast'][0]['flat_band'], .02)
            self.assertTrue(all(f['label'] == 'inconclusive' for f in r['forecast']))
            self.assertIn('1-day: inconclusive', completed.stdout)
            (path/'prices.csv').write_text('Date,Close\n2024-01-01,\n')
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn('missing', failed.stderr)
            self.assertNotIn('Traceback', failed.stderr)

    def test_history_window_defaults_to_one_year_and_preserves_source_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=600),
                          'Close': 100.}).to_csv(path/'prices.csv', index=False)
            command = [sys.executable, '-m', 'crypto_predictor', '--csv', str(path/'prices.csv'),
                       '--output', str(path/'report.json')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((path/'report.json').read_text())
            self.assertEqual(report['data']['rows'], 365)
            self.assertEqual(report['data']['start'], '2020-08-23')
            self.assertEqual(report['config']['history_days'], 365)
            self.assertEqual(report['config']['model'], 'direction')
            self.assertEqual(len(pd.read_csv(path/'prices.csv')), 600)
            failed = subprocess.run(command + ['--history-days', '0'], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn('History days must be positive', failed.stderr)
