"""Directional forecasts with explicit confidence and model-quality abstention."""
import math
import hashlib

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .pipeline import validate_prices
from .search import features



def direction_decision(probabilities, *, qualified, confidence_threshold=.65):
    if not math.isfinite(confidence_threshold) or not .5 < confidence_threshold <= 1:
        raise ValueError('Confidence threshold must be greater than 0.5 and at most 1')
    if probabilities is None and not qualified:
        return {'label': 'inconclusive', 'confidence': None,
                'probabilities': None, 'reason': 'model_not_qualified'}
    if (not isinstance(probabilities, dict) or set(probabilities) != {'up', 'down', 'same'} or
        any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()) or
        not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-8)):
        raise ValueError('Probabilities must contain up/down/same, be finite and sum to 1')
    label = max(probabilities, key=probabilities.get)
    if not qualified:
        return {'label': 'inconclusive', 'confidence': None,
                'probabilities': probabilities, 'reason': 'model_not_qualified'}
    if probabilities[label] < confidence_threshold:
        return {'label': 'inconclusive', 'confidence': None,
                'probabilities': probabilities, 'reason': 'low_confidence'}
    return {'label': label, 'confidence': probabilities[label],
            'probabilities': probabilities, 'reason': 'qualified_confident_prediction'}


def movement_label(reference_price, actual_price, band):
    """Ground-truth class; the roughly-flat band includes its endpoints."""
    if (not all(math.isfinite(v) for v in (reference_price, actual_price, band))
        or reference_price <= 0 or actual_price <= 0 or not 0 < band < 1):
        raise ValueError('Prices must be finite and positive; movement band must be between 0 and 1')
    if actual_price > reference_price * (1 + band):
        return 'up'
    if actual_price < reference_price * (1 - band):
        return 'down'
    return 'same'


CLASSES = ('down', 'same', 'up')
DEFAULT_BANDS = {1: .01, 7: .03, 28: .05}


def _fit(prices, x, origin, horizon, band, seed):
    rows = np.arange(14, origin-horizon+1)
    y = np.array([movement_label(prices[t], prices[t+horizon], band) for t in rows])
    prior = {label: float(np.mean(y == label)) for label in CLASSES}
    split = int(len(rows)*.8)
    # Purge overlapping future labels at the train/calibration boundary.
    train_stop = split-horizon+1
    train, calibration = rows[:max(0, train_stop)], rows[split:]
    if (len(train) < 60 or len(calibration) < 30 or
        any(np.sum(y[:train_stop] == c) < 5 or np.sum(y[split:] == c) < 5 for c in CLASSES)):
        return None, prior
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=seed))
    model.fit(x[train], y[:train_stop])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
    calibrated.fit(x[calibration], y[split:])
    return calibrated, prior


def _probabilities(model, x):
    if model is None:
        return [None] * len(x)
    return [{label: float(p) for label, p in zip(model.classes_, row)}
            for row in model.predict_proba(x)]


def _backtest(frame, horizon, band, start, stop, seed):
    prices = frame.Close.to_numpy(dtype=float)
    x = features(prices, 14)
    rows = []
    for target in range(start, stop, 30):
        indices = np.arange(target, min(target+30, stop))
        origins = indices-horizon
        model, prior = _fit(prices, x, int(origins[0]), horizon, band, seed)
        for index, origin, probabilities in zip(indices, origins, _probabilities(model, x[origins])):
            rows.append({'date': frame.Date.iloc[index].strftime('%Y-%m-%d'),
                         'origin_date': frame.Date.iloc[origin].strftime('%Y-%m-%d'),
                         'actual': movement_label(prices[origin], prices[index], band),
                         'probabilities': probabilities, 'baseline_probabilities': prior})
    return rows


def _score(rows, threshold):
    calls, correct, baseline_correct, errors, baseline_errors = 0, 0, 0, [], []
    for row in rows:
        probabilities = row['probabilities']
        if probabilities is None:
            continue
        decision = direction_decision(probabilities, qualified=True, confidence_threshold=threshold)
        baseline = max(row['baseline_probabilities'], key=row['baseline_probabilities'].get)
        if decision['label'] != 'inconclusive':
            calls += 1
            correct += decision['label'] == row['actual']
            baseline_correct += baseline == row['actual']
        errors.append(sum((probabilities[c] - (row['actual'] == c))**2 for c in CLASSES))
        baseline_errors.append(sum((row['baseline_probabilities'][c] - (row['actual'] == c))**2 for c in CLASSES))
    accuracy = correct/calls if calls else None
    baseline_accuracy = baseline_correct/calls if calls else None
    coverage = calls/len(rows)
    brier = float(np.mean(errors)) if errors else None
    baseline_brier = float(np.mean(baseline_errors)) if baseline_errors else None
    qualified = bool(calls >= 30 and coverage >= .1 and accuracy >= .65 and
                     accuracy >= baseline_accuracy+.05 and brier <= baseline_brier)
    return {'rows': len(rows), 'calls': calls, 'call_coverage': coverage,
            'call_accuracy': accuracy, 'baseline_accuracy_on_calls': baseline_accuracy,
            'multiclass_brier': brier, 'baseline_multiclass_brier': baseline_brier,
            'probability_coverage': len(errors)/len(rows), 'passes': qualified}


def direction_experiment(frame, *, bands=None, confidence_threshold=.65, seed=42):
    """Evaluate a calibrated three-class model, abstaining when evidence is weak.

    Up/down lie strictly outside a fixed, horizon-specific flat band. Inconclusive
    is a decision to abstain, never a training label. No price predictions are
    interpreted as probabilities.
    """
    bands = DEFAULT_BANDS.copy() if bands is None else dict(bands)
    if not bands or any(type(h) is not int or not 1 <= h <= 90 for h in bands):
        raise ValueError('Provide at least one integer horizon between 1 and 90 days')
    for band in bands.values():
        movement_label(100, 100, band)
    direction_decision(None, qualified=False, confidence_threshold=confidence_threshold)
    frame = validate_prices(frame)
    if len(frame) < 300:
        raise ValueError('Directional evaluation needs at least 300 consecutive daily prices')
    prices = frame.Close.to_numpy(dtype=float)
    start, stop = int(len(frame)*.6), int(len(frame)*.8)
    if start - 2*max(bands) - 14 < 30:
        raise ValueError('Insufficient mature history for the requested directional horizons')
    report = {'schema_version': 3,
              'config': {'model': 'direction', 'bands': bands, 'confidence_threshold': confidence_threshold,
                         'seed': seed, 'refit_days': 30, 'calibration': 'chronological_sigmoid',
                         'gates': {'min_calls': 30, 'min_coverage': .1, 'min_accuracy': .65,
                                   'min_accuracy_gain': .05, 'brier_no_worse_than_prior': True}},
              'data': {'rows': len(frame), 'start': frame.Date.iloc[0].strftime('%Y-%m-%d'),
                       'end': frame.Date.iloc[-1].strftime('%Y-%m-%d'),
                       'validation_start': frame.Date.iloc[start].strftime('%Y-%m-%d'),
                       'test_start': frame.Date.iloc[stop].strftime('%Y-%m-%d'),
                       'sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()},
              'evaluation': {'protocol': 'rolling_direction_with_abstention', 'horizons': {}},
              'holdout': {}, 'forecast': []}
    with threadpool_limits(limits=1):
        for horizon, band in sorted(bands.items()):
            validation = _backtest(frame.iloc[:stop], horizon, band, start, stop, seed)
            validation_score = _score(validation, confidence_threshold)
            final = _backtest(frame, horizon, band, stop, len(frame), seed)
            final_score = _score(final, confidence_threshold)
            qualified = validation_score['passes'] and final_score['passes']
            report['evaluation']['horizons'][str(horizon)] = {
                'validation': validation_score, 'final': final_score, 'qualified': qualified}
            report['holdout'][str(horizon)] = final
            model, _ = _fit(prices, features(prices, 14), len(prices)-1, horizon, band, seed)
            probabilities = _probabilities(model, features(prices, 14)[-1:])[0]
            decision = direction_decision(probabilities, qualified=qualified and model is not None,
                                          confidence_threshold=confidence_threshold)
            report['forecast'].append({
                'date': (frame.Date.iloc[-1]+pd.Timedelta(days=horizon)).strftime('%Y-%m-%d'),
                'horizon_days': horizon, 'flat_band': band,
                'reference_close': float(prices[-1]), **decision})
    return report
