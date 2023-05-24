"""CLI adapters for CSV/Yahoo input and a portable JSON experiment report."""

import argparse
import importlib.metadata
import json
import platform
import sqlite3
from pathlib import Path
import sys

import pandas as pd

from .pipeline import experiment, validate_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--ticker", help="Daily crypto ticker, e.g. BTC-USD")
    parser.add_argument("--model", choices=["persistence", "lstm", "auto", "direction"], default="direction")
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-days", type=int, default=365,
                        help="Use only the latest N daily observations; default: 365; retain the full cache")
    parser.add_argument("--confidence-threshold", type=float, default=.65)
    parser.add_argument("--flat-band-1d", type=float, default=.01)
    parser.add_argument("--flat-band-7d", type=float, default=.03)
    parser.add_argument("--flat-band-28d", type=float, default=.05)
    parser.add_argument("--output", type=Path, default=Path("artifacts/report.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/prices.sqlite3"))
    parser.add_argument("--full-refresh", action="store_true", help="Refresh all stored history from Yahoo")
    parser.add_argument("--offline", action="store_true", help="Use cached ticker prices without network requests")
    args = parser.parse_args()
    if args.csv is None and args.ticker is None:
        args.ticker = "BTC-USD"
    if args.csv and (args.offline or args.full_refresh):
        parser.error("--offline and --full-refresh apply only to --ticker")
    try:
        if args.history_days is not None and args.history_days < 1:
            raise ValueError('History days must be positive')
        if args.csv:
            frame = pd.read_csv(args.csv)
        else:
            from .prices import load_prices

            frame, data_access = load_prices(
                args.ticker, args.cache, full_refresh=args.full_refresh, offline=args.offline,
                progress=lambda message: print(message, flush=True),
            )
        frame = validate_prices(frame)
        if args.history_days is not None:
            frame = frame.tail(args.history_days).reset_index(drop=True)
        print(f"Loaded {len(frame)} daily closes", flush=True)
        if args.model == "direction":
            from .direction import direction_experiment

            result = direction_experiment(frame, seed=args.seed, confidence_threshold=args.confidence_threshold,
                                          bands={1: args.flat_band_1d, 7: args.flat_band_7d, 28: args.flat_band_28d})
        elif args.model == "auto":
            from .search import compare

            result = compare(frame, horizon=args.horizon, seed=args.seed,
                             progress=lambda message: print(message, flush=True))
        else:
            result = experiment(frame, args.model, args.lookback, args.horizon, args.epochs, args.seed)
        result["source"] = str(args.csv) if args.csv else args.ticker
        result['config']['history_days'] = args.history_days
        if args.ticker:
            result["data_access"] = data_access
        packages = ["numpy", "pandas", "scikit-learn"]
        if args.model == "lstm":
            packages.append("tensorflow")
        if args.ticker and not args.offline:
            packages.append("yfinance")
        result["runtime"] = {"python": platform.python_version(),
                             "packages": {p: importlib.metadata.version(p) for p in packages}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output.with_suffix(".prices.csv"), index=False)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, ImportError, KeyError, sqlite3.Error) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {args.output}")
    if args.model == "direction":
        for forecast in result['forecast']:
            confidence = (f"{forecast['confidence']:.0%}" if forecast['confidence'] is not None else 'n/a')
            print(f"{forecast['horizon_days']}-day: {forecast['label']} "
                  f"(confidence {confidence}; flat band ±{forecast['flat_band']:.1%}; {forecast['reason']})")
            scores = result['evaluation']['horizons'][str(forecast['horizon_days'])]
            for period in ('validation', 'final'):
                score = scores[period]
                accuracy = f"{score['call_accuracy']:.0%}" if score['call_accuracy'] is not None else 'n/a'
                print(f"  {period} candidate call coverage {score['call_coverage']:.0%}; "
                      f"accuracy {accuracy}; calls {score['calls']}; gate {'passed' if score['passes'] else 'failed'}")
        print('Forecasts:')
        print(json.dumps(result['forecast'], indent=2))
    elif args.model == "auto":
        for horizon, result_row in result["evaluation"]["horizons"].items():
            print(f"{horizon}-day: selected={result_row['selected']}, "
                  f"MAE={result_row['model']['mae']:.2f}, "
                  f"baseline MAE={result_row['persistence']['mae']:.2f}, "
                  f"forecast={result_row['forecast_strategy']} ({result_row['status']})")
        print("Forecasts:")
        print(json.dumps(result["forecast"], indent=2))
    else:
        print(json.dumps(result["evaluation"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
