"""At most ten preregistered OHLCV experiments; one final retrospective evaluation.

Research output never changes the production forecast policy automatically.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .pipeline import metrics, validate_prices

WARMUP = 90
MIN_GAIN = 0.05
MIN_WINS = 0.6


@dataclass(frozen=True)
class Recipe:
    name: str
    kind: str
    feature_set: str = 'all'
    window: int = 1095
    half_life: int = 0


RECIPES = (
    Recipe('ridge_returns_control', 'ridge', 'returns'),
    Recipe('ridge_volume_and_range', 'ridge'),
    Recipe('weighted_median_regression', 'quantile'),
    Recipe('trees_returns_control', 'boost', 'returns'),
    Recipe('trees_range', 'boost', 'range'),
    Recipe('trees_volume', 'boost', 'volume'),
    Recipe('trees_volume_and_range', 'boost'),
    Recipe('trees_recent_regime', 'boost', half_life=180),
    Recipe('extra_trees_volume_and_range', 'extra'),
)


def validate_bars(frame):
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not set(columns).issubset(frame.columns):
        raise ValueError('Research needs Date, Open, High, Low, Close and Volume')
    frame = frame[columns].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], utc=True).dt.normalize()
    frame = frame.sort_values('Date').reset_index(drop=True)
    validate_prices(frame)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(frame[columns[1:]].to_numpy()).all():
        raise ValueError('OHLCV values must be finite')
    if (frame[['Open', 'High', 'Low', 'Close']] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError('Prices must be positive; volume must be nonnegative')
    if ((frame.Low > frame[['Open', 'Close']].min(axis=1)) |
        (frame.High < frame[['Open', 'Close']].max(axis=1)) | (frame.High < frame.Low)).any():
        raise ValueError('Inconsistent daily high/low bounds')
    return frame


def make_features(frame):
    """All features are available at the completed daily candle's close."""
    close = frame.Close
    returns = np.log(close).diff()
    base = pd.DataFrame({f'return_lag_{i}': returns.shift(i) for i in range(14)})
    for window in (7, 30, 90):
        base[f'momentum_{window}'] = np.log(close / close.shift(window))
        base[f'volatility_{window}'] = returns.rolling(window).std(ddof=0)
    ranges = pd.DataFrame({
        'range': np.log(frame.High / frame.Low),
        'body': np.log(close / frame.Open),
        'close_position': (close-frame.Low) / (frame.High-frame.Low).replace(0, np.nan),
    })
    ranges['close_position'] = ranges['close_position'].fillna(0.5)
    ranges['range_mean_7'] = ranges['range'].rolling(7).mean()
    ranges['range_mean_30'] = ranges['range'].rolling(30).mean()
    log_volume = np.log1p(frame.Volume)
    volume = pd.DataFrame({'volume_change': log_volume.diff()})
    for window in (7, 30, 90):
        volume[f'volume_deviation_{window}'] = log_volume - log_volume.rolling(window).mean()
    groups = {'returns': base, 'range': pd.concat([base, ranges], axis=1),
              'volume': pd.concat([base, volume], axis=1),
              'all': pd.concat([base, ranges, volume], axis=1)}
    return {name: values.to_numpy(dtype=float) for name, values in groups.items()}


def fit_predict(prices, x, origin, queries, recipe, seed=42):
    # Label at t is next-day simple return. Its close at t+1 must be observed.
    rows = np.arange(max(WARMUP, origin-recipe.window), origin)
    if len(rows) < 30:
        raise ValueError('Not enough mature training examples')
    y = prices[rows+1] / prices[rows] - 1
    # Price-weighted absolute return error equals absolute price error exactly.
    weights = prices[rows] / prices[rows].mean()
    if recipe.half_life:
        weights *= np.exp2(-(origin-1-rows) / recipe.half_life)
    weights /= weights.mean()
    if recipe.kind in {'ridge', 'quantile'}:
        estimator = (Ridge(alpha=1000) if recipe.kind == 'ridge'
                     else QuantileRegressor(quantile=0.5, alpha=0.0001, solver='highs'))
        model = make_pipeline(StandardScaler(), estimator)
        key = 'ridge' if recipe.kind == 'ridge' else 'quantileregressor'
        model.fit(x[rows], y, **{f'{key}__sample_weight': weights})
    else:
        if recipe.kind == 'boost':
            model = HistGradientBoostingRegressor(
                loss='absolute_error', max_iter=80, max_leaf_nodes=7,
                min_samples_leaf=40, l2_regularization=10, learning_rate=0.04,
                early_stopping=False, random_state=seed)
        elif recipe.kind == 'extra':
            model = ExtraTreesRegressor(n_estimators=64, max_depth=5, min_samples_leaf=30,
                                        criterion='absolute_error', n_jobs=1, random_state=seed)
        else:
            raise ValueError(f'Unknown recipe: {recipe.kind}')
        model.fit(x[rows], y, sample_weight=weights)
    predicted = prices[queries] * (1 + model.predict(x[queries]))
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError('Model produced invalid prices')
    return predicted


def walk_forward(prices, features, recipe, start, stop, seed=42):
    predictions = []
    for target in range(start, stop, 30):
        queries = np.arange(target-1, min(target+30, stop)-1)
        predictions.extend(fit_predict(prices, features[recipe.feature_set], target-1, queries, recipe, seed))
    return np.asarray(predictions)


def score(prices, predicted, start, stop):
    actual, baseline = prices[start:stop], prices[start-1:stop-1]
    model, base = metrics(actual, predicted), metrics(actual, baseline)
    gain = 1-model['mae']/base['mae'] if base['mae'] else 0.0
    folds = []
    for a in range(0, len(actual), 30):
        m = metrics(actual[a:a+30], predicted[a:a+30])
        b = metrics(actual[a:a+30], baseline[a:a+30])
        folds.append({'model_mae': m['mae'], 'baseline_mae': b['mae']})
    middle = len(actual)//2
    era_gains = []
    for sl in (slice(0, middle), slice(middle, None)):
        b = metrics(actual[sl], baseline[sl])['mae']
        m = metrics(actual[sl], predicted[sl])['mae']
        era_gains.append(1-m/b if b else 0.0)
    wins = sum(f['model_mae'] < f['baseline_mae'] for f in folds)/len(folds)
    return {'model': model, 'persistence': base, 'mae_improvement_fraction': gain,
            'fold_win_fraction': wins, 'era_improvements': era_gains, 'folds': folds,
            'passes': bool(gain >= MIN_GAIN and wins >= MIN_WINS and min(era_gains) > 0)}


def uncertainty(prices, prediction, start, stop, seed=42):
    actual, base = prices[start:stop], prices[start-1:stop-1]
    e, b = np.abs(actual-prediction), np.abs(actual-base)
    rng = np.random.default_rng(seed)
    result = {}
    for block in (30, 90):
        samples = []
        for _ in range(2000):
            starts = rng.integers(0, len(e), int(np.ceil(len(e)/block)))
            indices = ((starts[:, None]+np.arange(block)) % len(e)).ravel()[:len(e)]
            denominator = b[indices].mean()
            samples.append(1-e[indices].mean()/denominator if denominator else 0.0)
        result[str(block)] = np.quantile(samples, [.025, .975]).tolist()
    return result


def run(frame, max_loops=10, output=Path('artifacts/research/report.json'), seed=42, progress=print):
    if not 1 <= max_loops <= 10:
        raise ValueError('max-loops must be between 1 and 10')
    frame = validate_bars(frame)
    if len(frame) < 600:
        raise ValueError('Research needs at least 600 complete daily bars')
    prices = frame.Close.to_numpy(dtype=float)
    start, stop = int(len(prices)*.4), int(len(prices)*.8)
    # Features for validation are physically isolated from the final 20%.
    validation_features = make_features(frame.iloc[:stop])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix('.bars.csv'), index=False)
    report = {'schema_version': 1, 'status': 'running', 'max_loops': max_loops,
              'seed': seed, 'refit_days': 30, 'target': 'next_day_simple_return',
              'data': {'sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                       'rows': len(frame), 'start': str(frame.Date.iloc[0].date()),
                       'end': str(frame.Date.iloc[-1].date()),
                       'validation_start': str(frame.Date.iloc[start].date()),
                       'final_start': str(frame.Date.iloc[stop].date())},
              'criteria': {'min_gain': MIN_GAIN, 'min_fold_wins': MIN_WINS,
                           'positive_gain_in_both_halves': True,
                           'positive_final_bootstrap_lower_bounds': True},
              'plan': [asdict(recipe) for recipe in RECIPES] +
                      [{'name': 'top_three_shrunk_ensemble', 'shrink': .5}],
              'runtime': {'python': platform.python_version(),
                          'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn']}},
              'iterations': []}
    def checkpoint():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)
    checkpoint()
    predictions = {}
    with threadpool_limits(limits=1):
        for index, recipe in enumerate(RECIPES[:max_loops], start=1):
            progress(f'Iteration {index}/{max_loops}: {recipe.name}', flush=True)
            pred = walk_forward(prices[:stop], validation_features, recipe, start, stop, seed)
            predictions[recipe.name] = pred
            result = score(prices, pred, start, stop)
            report['iterations'].append({'iteration': index, 'recipe': asdict(recipe), 'validation': result})
            progress(f"  MAE gain {result['mae_improvement_fraction']:+.2%}; block wins {result['fold_win_fraction']:.0%}", flush=True)
            checkpoint()
        if max_loops == 10:
            ranked = sorted(report['iterations'], key=lambda row: row['validation']['model']['mae'])
            members = [row['recipe']['name'] for row in ranked[:3]]
            pred = .5*np.mean([predictions[name] for name in members], axis=0)+.5*prices[start-1:stop-1]
            predictions['top_three_shrunk_ensemble'] = pred
            result = score(prices, pred, start, stop)
            report['iterations'].append({'iteration': 10, 'recipe': {
                'name': 'top_three_shrunk_ensemble', 'kind': 'ensemble', 'members': members, 'shrink': .5},
                'validation': result})
            progress(f"Iteration 10/10: ensemble; MAE gain {result['mae_improvement_fraction']:+.2%}", flush=True)
            checkpoint()
        eligible = [row for row in report['iterations'] if row['validation']['passes']]
        # Even on failure, diagnose ONE frozen validation winner on the final period.
        winner = min(eligible or report['iterations'], key=lambda row: row['validation']['model']['mae'])
        report['selected'] = winner['recipe']
        report['validation_qualified'] = bool(eligible)
        checkpoint()
        progress(f"Final evaluation once: {winner['recipe']['name']}", flush=True)
        all_features = make_features(frame)
        if winner['recipe']['kind'] == 'ensemble':
            recipes = [r for r in RECIPES if r.name in winner['recipe']['members']]
            pred = .5*np.mean([walk_forward(prices, all_features, r, stop, len(prices), seed)
                              for r in recipes], axis=0)+.5*prices[stop-1:-1]
        else:
            pred = walk_forward(prices, all_features, Recipe(**winner['recipe']), stop, len(prices), seed)
        report['final'] = score(prices, pred, stop, len(prices))
        report['final']['descriptive_gain_intervals'] = uncertainty(prices, pred, stop, len(prices), seed)
        interval_pass = all(interval[0] > 0 for interval in report['final']['descriptive_gain_intervals'].values())
        report['qualified'] = bool(eligible and report['final']['passes'] and interval_pass)
        report['status'] = 'completed'
        report['production_policy_changed'] = False
        report['final_predictions'] = [{'date': str(frame.Date.iloc[i].date()), 'actual': float(prices[i]),
                                        'prediction': float(p), 'persistence': float(prices[i-1])}
                                       for i,p in zip(range(stop,len(prices)),pred)]
        checkpoint()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('artifacts/research/ohlcv.csv'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/research/report.json'))
    parser.add_argument('--max-loops', type=int, choices=range(1,11), default=10)
    parser.add_argument('--refresh-data', action='store_true', help='Explicitly replace the frozen Bitcoin OHLCV snapshot')
    args = parser.parse_args()
    if not args.csv.exists() or args.refresh_data:
        from .prices import download_prices

        today = pd.Timestamp.now(tz='UTC').normalize()
        print('Downloading one Bitcoin OHLCV snapshot; reused by all iterations.', flush=True)
        frame = download_prices('BTC-USD', (today-pd.DateOffset(years=10)).strftime('%Y-%m-%d'),
                                today.strftime('%Y-%m-%d'),
                                columns=('Date', 'Open', 'High', 'Low', 'Close', 'Volume'))
        frame = validate_bars(frame)
        frame = frame.loc[frame.Date < today]
        if frame.empty:
            raise ValueError('No completed daily bars returned')
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.csv.with_suffix('.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(args.csv)
    result = run(pd.read_csv(args.csv), args.max_loops, args.output)
    print(f"Completed {len(result['iterations'])} iterations. Qualified: {result['qualified']}.")
    print(f"Final MAE: {result['final']['model']['mae']:.2f}; persistence: {result['final']['persistence']['mae']:.2f}")
    print(f'Report: {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f'Research failed: {exc}', file=sys.stderr)
        sys.exit(1)
