"""Bounded strategy search with chronological selection and rolling refits.

An origin is the index of the latest observed close. Training labels must mature
at or before that origin, including for multi-day direct-return targets.
"""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .pipeline import metrics, validate_prices


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str = "persistence"
    window: int = 730
    lags: int = 14
    alpha: float = 100.0
    shrink: float = 1.0
    leaves: int = 7


# Fixed before inspecting the final evaluation period. No unbounded tuning loop.
STRATEGIES = (
    Strategy("persistence"),
    Strategy("median_90", "median", window=90),
    Strategy("median_365", "median", window=365),
    Strategy("mean_365_shrunk", "mean", window=365, shrink=0.25),
    Strategy("ridge_7_365", "ridge", window=365, lags=7, alpha=100),
    Strategy("ridge_30_730", "ridge", window=730, lags=30, alpha=100),
    Strategy("ridge_14_1095", "ridge", window=1095, lags=14, alpha=1000),
    Strategy("ridge_14_shrunk", "ridge", window=730, lags=14, alpha=100, shrink=0.25),
    Strategy("trees_7_730", "trees", window=730, leaves=7),
    Strategy("trees_15_1095", "trees", window=1095, leaves=15),
    Strategy("trees_7_shrunk", "trees", window=730, leaves=7, shrink=0.25),
    Strategy("median_30", "median", window=30),
    Strategy("mean_90_shrunk", "mean", window=90, shrink=0.25),
    Strategy("trees_365_shrunk", "trees", window=365, leaves=7, shrink=0.25),
    Strategy("ridge_7_90_shrunk", "ridge", window=90, lags=7, alpha=100, shrink=0.25),
)


def features(prices, lags):
    """Row t contains only returns observable through the close at t."""
    returns = np.diff(np.log(prices))
    result = np.full((len(prices), lags + 4), np.nan)
    for t in range(lags, len(prices)):
        recent = returns[t - lags:t]
        result[t] = np.r_[recent[::-1], recent.mean(), recent.std(),
                          recent.min(), recent.max()]
    return result


def training_indices(origin, horizon, strategy):
    # t+h <= origin ensures every target is observed when the model is fitted.
    last = origin - horizon
    first = max(strategy.lags, last - strategy.window + 1)
    return np.arange(first, last + 1)


def fit_predict(prices, x, origin, query_origins, horizon, strategy, seed):
    if strategy.kind == "persistence":
        return prices[query_origins].copy()
    train = training_indices(origin, horizon, strategy)
    if len(train) < 30:
        raise ValueError("Insufficient mature training targets for strategy search")
    y = np.log(prices[train + horizon] / prices[train])
    if strategy.kind in {"median", "mean"}:
        estimate = np.median(y) if strategy.kind == "median" else np.mean(y)
        predicted = np.full(len(query_origins), estimate)
    else:
        if strategy.kind == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=strategy.alpha))
        elif strategy.kind == "trees":
            model = HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=60, learning_rate=0.05,
                max_leaf_nodes=strategy.leaves, min_samples_leaf=30,
                l2_regularization=10, early_stopping=False, random_state=seed,
            )
        else:
            raise ValueError(f"Unknown strategy kind: {strategy.kind}")
        model.fit(x[train], y)
        predicted = model.predict(x[query_origins])
    forecast = prices[query_origins] * np.exp(strategy.shrink * predicted)
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ValueError(f"Invalid forecast from {strategy.name}")
    return forecast


def backtest(prices, strategy, horizon, start, stop, refit_days=30, seed=42):
    """Evaluate target dates [start, stop), refitting at each block's first origin."""
    x = features(prices, strategy.lags)
    predicted, folds = [], []
    for target_start in range(start, stop, refit_days):
        target_stop = min(target_start + refit_days, stop)
        targets = np.arange(target_start, target_stop)
        origins = targets - horizon
        forecasts = fit_predict(prices, x, int(origins[0]), origins, horizon, strategy, seed)
        predicted.extend(forecasts)
        folds.append({
            "target_start_index": target_start, "target_stop_index": target_stop,
            "fit_origin_index": int(origins[0]),
            "model": metrics(prices[targets], forecasts),
            "persistence": metrics(prices[targets], prices[origins]),
        })
    targets = np.arange(start, stop)
    return {"metrics": metrics(prices[targets], predicted), "folds": folds,
            "predictions": np.asarray(predicted)}


def select_strategy(rows, min_improvement=0.01, min_win_fraction=0.6):
    baseline = next(row for row in rows if row["strategy"]["name"] == "persistence")
    baseline_mae = baseline["metrics"]["mae"]
    eligible = [row for row in rows
                if row["strategy"]["name"] != "persistence"
                and baseline_mae > 0
                and row["metrics"]["mae"] <= baseline_mae * (1 - min_improvement)
                and row["fold_win_fraction"] >= min_win_fraction]
    if not eligible:
        return baseline, "No challenger passed the validation improvement and consistency gates"
    return min(eligible, key=lambda row: row["metrics"]["mae"]), "Passed validation gates"


def compare(frame, horizon=28, seed=42, refit_days=30, strategies=STRATEGIES, progress=None):
    if horizon < 1 or horizon > 90 or refit_days < 1:
        raise ValueError("Search horizon must be 1–90 days; refit interval must be positive")
    if not strategies or len({s.name for s in strategies}) != len(strategies):
        raise ValueError("Strategies must be nonempty with unique names")
    if not any(s.name == "persistence" and s.kind == "persistence" for s in strategies):
        raise ValueError("Strategy search requires a persistence baseline")
    frame = validate_prices(frame)
    prices = frame["Close"].to_numpy(dtype=float)
    validation_start, test_start = int(len(prices) * 0.6), int(len(prices) * 0.8)
    if validation_start - horizon - max(s.lags for s in strategies) < 30 or len(prices) < 180:
        raise ValueError("Strategy search needs at least 180 rows and enough mature training targets")
    horizons = sorted({1, min(7, horizon), horizon})
    dates = frame["Date"].dt.strftime("%Y-%m-%d")
    report = {
        "schema_version": 2,
        "config": {"model": "auto", "seed": seed, "horizons": horizons,
                   "refit_days": refit_days, "validation_fraction": [0.6, 0.8],
                   "selection_min_mae_improvement": 0.01, "selection_min_fold_win_fraction": 0.6,
                   "release_min_mae_improvement": 0.01,
                   "strategies": [asdict(s) for s in strategies]},
        "data": {"rows": len(frame), "start": dates.iloc[0], "end": dates.iloc[-1],
                 "validation_start": dates.iloc[validation_start],
                 "validation_end": dates.iloc[test_start - 1], "test_start": dates.iloc[test_start],
                 "sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()},
        "selection": {}, "evaluation": {"protocol": "rolling_direct_return", "horizons": {}},
        "holdout": {}, "forecast": [],
    }
    # Selection is completed for every horizon before any final-test scoring.
    chosen = {}
    with threadpool_limits(limits=1):
        for h in horizons:
            rows = []
            for strategy in strategies:
                if progress:
                    progress(f"Validation: {h}-day horizon / {strategy.name}")
                # Physically exclude final holdout data during strategy selection.
                run = backtest(prices[:test_start], strategy, h, validation_start, test_start, refit_days, seed)
                wins = sum(f["model"]["mae"] < f["persistence"]["mae"] for f in run["folds"])
                rows.append({"strategy": asdict(strategy), "metrics": run["metrics"],
                             "fold_win_fraction": wins / len(run["folds"]), "folds": run["folds"]})
            selected, reason = select_strategy(rows)
            chosen[h] = Strategy(**selected["strategy"])
            report["selection"][str(h)] = {"selected": chosen[h].name, "reason": reason,
                                             "leaderboard": sorted(rows, key=lambda r: r["metrics"]["mae"])}
        for h, strategy in chosen.items():
            if progress:
                progress(f"Final evaluation and refit: {h}-day horizon / {strategy.name}")
            run = backtest(prices, strategy, h, test_start, len(prices), refit_days, seed)
            targets = np.arange(test_start, len(prices))
            baseline = prices[targets - h]
            baseline_metrics = metrics(prices[targets], baseline)
            base_mae = baseline_metrics["mae"]
            gain = 1 - run["metrics"]["mae"] / base_mae if base_mae else None
            # A predeclared release gate may fall back, but never chooses another challenger.
            released = strategy.kind != "persistence" and gain is not None and gain >= 0.01
            forecast_strategy = strategy if released else Strategy("persistence")
            report["evaluation"]["horizons"][str(h)] = {
                "selected": strategy.name, "model": run["metrics"],
                "persistence": baseline_metrics, "mae_improvement_fraction": gain,
                "forecast_strategy": forecast_strategy.name,
                "status": "passed_release_gate" if released else "baseline_fallback",
                "folds": run["folds"],
            }
            report["holdout"][str(h)] = [
                {"date": dates.iloc[t], "origin_date": dates.iloc[t-h],
                 "actual": float(prices[t]), "prediction": float(p), "persistence": float(b)}
                for t, p, b in zip(targets, run["predictions"], baseline)
            ]
            # Forecast models are refitted on every available, fully matured label.
            forecast = fit_predict(prices, features(prices, forecast_strategy.lags), len(prices)-1,
                                   np.array([len(prices)-1]), h, forecast_strategy, seed)[0]
            report["forecast"].append({
                "date": (frame["Date"].iloc[-1] + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                "horizon_days": h, "close": float(forecast), "strategy": forecast_strategy.name,
                "label": "inconclusive", "confidence": None, "probabilities": None,
                "reason": "point_forecast_has_no_directional_confidence",
            })
    return report
