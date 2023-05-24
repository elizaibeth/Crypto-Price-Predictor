"""Keep data validation and evaluation independent of network and model imports."""

import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def validate_prices(frame):
    if not {"Date", "Close"}.issubset(frame.columns):
        raise ValueError("Input must contain Date and Close columns")
    frame = frame[["Date", "Close"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise", utc=True)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="raise")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Prices must be nonempty with no missing dates or closes")
    if not np.isfinite(frame["Close"]).all() or (frame["Close"] <= 0).any():
        raise ValueError("Close prices must be finite and positive")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")
    if not (frame["Date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Expected consecutive daily observations; gaps are not imputed")
    return frame


def prepare(values, lookback, split):
    """Fit on training rows only; the first test window uses training context."""
    scaler = MinMaxScaler().fit(values[:split])
    scaled = scaler.transform(values).ravel()
    x = np.array([scaled[i - lookback:i] for i in range(lookback, len(values))])
    y = scaled[lookback:]
    boundary = split - lookback
    return scaler, x[:boundary, :, None], y[:boundary], x[boundary:, :, None]


def metrics(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    if not np.isfinite(error).all():
        raise ValueError("Model produced non-finite predictions")
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def experiment(frame, model="persistence", lookback=90, horizon=28, epochs=10, seed=42):
    if model not in {"persistence", "lstm"}:
        raise ValueError("Model must be persistence or lstm")
    if min(lookback, horizon, epochs) < 1:
        raise ValueError("Lookback, horizon and epochs must be positive")
    frame = validate_prices(frame)
    values = frame["Close"].to_numpy(dtype=float).reshape(-1, 1)
    split = int(len(values) * 0.8)
    if split <= lookback or split == len(values):
        raise ValueError("Insufficient rows: training partition must exceed lookback")
    actual = values[split:, 0]
    baseline = values[split - 1:-1, 0]
    predicted = baseline.copy()
    future = [float(values[-1, 0])] * horizon
    if model == "lstm":
        # Optional dependency: CSV baseline experiments never load TensorFlow.
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        scaler, x_train, y_train, x_test = prepare(values, lookback, split)
        network = tf.keras.Sequential([
            tf.keras.Input(shape=(lookback, 1)),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1),
        ])
        network.compile(optimizer="adam", loss="mean_squared_error")
        network.fit(x_train, y_train, epochs=epochs, batch_size=32, shuffle=False, verbose=0)
        predicted = scaler.inverse_transform(network.predict(x_test, verbose=0))[:, 0]
        window = scaler.transform(values[-lookback:]).ravel()
        future = []
        for _ in range(horizon):
            next_scaled = float(network(window[None, :, None], training=False).numpy()[0, 0])
            future.append(float(scaler.inverse_transform([[next_scaled]])[0, 0]))
            window = np.append(window[1:], next_scaled)
    if not np.isfinite(future).all():
        raise ValueError("Model produced non-finite forecasts")
    dates = frame["Date"].dt.strftime("%Y-%m-%d")
    forecast_dates = pd.date_range(frame["Date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    return {
        "schema_version": 1,
        "config": {"model": model, "lookback": lookback, "horizon": horizon,
                   "epochs": epochs, "seed": seed, "train_fraction": 0.8},
        "data": {"rows": len(frame), "start": dates.iloc[0], "end": dates.iloc[-1],
                 "train_end": dates.iloc[split - 1], "test_start": dates.iloc[split],
                 "sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()},
        "evaluation": {"protocol": "one_step_observed_history",
                       "model": metrics(actual, predicted), "persistence": metrics(actual, baseline)},
        "holdout": [{"date": date, "actual": float(a), "prediction": float(p),
                     "persistence": float(b)}
                    for date, a, p, b in zip(dates.iloc[split:], actual, predicted, baseline)],
        "forecast": [{"date": date.strftime("%Y-%m-%d"), "close": price,
                      "label": "inconclusive", "confidence": None, "probabilities": None,
                      "reason": "point_forecast_has_no_directional_confidence"}
                     for date, price in zip(forecast_dates, future)],
    }
