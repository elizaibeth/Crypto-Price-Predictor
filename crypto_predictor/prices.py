"""Persistent daily Yahoo closes with transactional, validated incremental updates."""
from pathlib import Path
import sqlite3

import pandas as pd

from .pipeline import validate_prices


# Keep a long research window in the cache. The CLI can still select a shorter
# training window (365 days by default) without throwing away older cycles.
CACHE_HISTORY_YEARS = 12


class DataUnavailable(ValueError):
    """The remote provider could not supply usable data."""


BAR_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def download_prices(ticker, start, end, columns=("Date", *BAR_COLUMNS)):
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    try:
        raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    except YFRateLimitError as exc:
        raise DataUnavailable("Yahoo Finance rate-limited the download") from exc
    except Exception as exc:
        # Translate failures only at the external provider boundary, not model/DB bugs.
        raise DataUnavailable(f"Yahoo Finance download failed ({type(exc).__name__}): {exc}") from exc
    if raw.empty:
        raise DataUnavailable("Yahoo Finance returned no prices")
    return raw.reset_index()[list(columns)]


def load_prices(ticker, cache_path=Path("artifacts/prices.sqlite3"), *, full_refresh=False,
                offline=False, today=None, downloader=download_prices, progress=print):
    if full_refresh and offline:
        raise ValueError("--full-refresh cannot be combined with --offline")
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be empty")
    today = pd.Timestamp.now(tz="UTC") if today is None else pd.Timestamp(today)
    today = (today.tz_localize("UTC") if today.tzinfo is None else today.tz_convert("UTC")).normalize()
    end = today.strftime("%Y-%m-%d")  # Yahoo's end is exclusive.
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL, date TEXT NOT NULL, close REAL NOT NULL CHECK(close > 0),
            open REAL, high REAL, low REAL, volume REAL,
            PRIMARY KEY (ticker, date)
        )''')
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(prices)")}
        for column in ("open", "high", "low", "volume"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE prices ADD COLUMN {column} REAL")
        cached = pd.read_sql_query(
            'SELECT date AS Date, open AS Open, high AS High, low AS Low, '
            'close AS Close, volume AS Volume FROM prices WHERE ticker = ? AND date < ? ORDER BY date',
            connection, params=(ticker, end),
        )
        cached["Date"] = pd.to_datetime(cached["Date"], utc=True)
        cached_has_bars = not cached.empty and cached[list(BAR_COLUMNS)].notna().all().all()
        cached_for_frame = cached if cached_has_bars else cached[["Date", "Close"]]
        cutoff = today - pd.DateOffset(years=CACHE_HISTORY_YEARS)
        start = cutoff
        gaps = []
        if not cached.empty:
            expected = pd.date_range(cached["Date"].iloc[0], cached["Date"].iloc[-1], freq="D")
            gaps = expected.difference(cached["Date"])
            # Refresh the last seven stored candles, plus any newly missing days.
            start = max(cached["Date"].iloc[0], cached["Date"].iloc[-1] - pd.Timedelta(days=6))
            if len(gaps):
                start = min(start, gaps[0])
            if full_refresh:
                start = min(cutoff, cached["Date"].iloc[0])
        warning = None
        requested = None
        mode = "offline" if offline else "updated"
        if offline:
            if cached.empty:
                raise ValueError(f"No cached prices for {ticker}; run online first or use --csv")
            frame = validate_prices(cached_for_frame)
        else:
            requested = {"start": start.strftime("%Y-%m-%d"), "end_exclusive": end}
            progress(f"Updating {ticker}: {requested['start']} through {(today-pd.Timedelta(days=1)).date()}"
                     + (f"; repairing {len(gaps)} missing dates" if len(gaps) else ""))
            try:
                downloaded = downloader(ticker, requested["start"], end).copy()
                if downloaded.empty:
                    raise DataUnavailable("Yahoo Finance returned no prices")
            except DataUnavailable as exc:
                if cached.empty:
                    raise DataUnavailable(f"{exc}. No cached data for {ticker}. Wait before retrying or use --csv.") from exc
                # An incomplete cache must not silently become training data.
                frame = validate_prices(cached_for_frame)
                warning = str(exc)
                mode = "cache_fallback"
            else:
                downloaded["Date"] = pd.to_datetime(downloaded["Date"], utc=True).dt.normalize()
                downloaded = downloaded.loc[(downloaded["Date"] >= start) & (downloaded["Date"] < today)]
                downloaded = validate_prices(downloaded)
                downloaded_columns = ["Date", "Close"] + [c for c in BAR_COLUMNS if c != "Close" and c in downloaded]
                if cached.empty:
                    merged = downloaded[downloaded_columns].copy()
                else:
                    # Legacy caches may have all-NA OHLCV columns after the
                    # schema migration. Exclude those columns from the left
                    # input to avoid pandas' all-NA concat warning.
                    cached_merge = cached[downloaded_columns]
                    if not cached_has_bars:
                        cached_merge = cached[["Date", "Close"]]
                    merged = pd.concat([cached_merge, downloaded[downloaded_columns]], ignore_index=True)
                merged = merged.drop_duplicates("Date", keep="last")
                merged_has_bars = set(BAR_COLUMNS).issubset(merged.columns) and merged[list(BAR_COLUMNS)].notna().all().all()
                frame = validate_prices(merged if merged_has_bars else merged[["Date", "Close"]])
                # Validate the entire merged series before committing any changed rows.
                if set(BAR_COLUMNS).issubset(downloaded.columns):
                    connection.executemany(
                        'INSERT INTO prices(ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?) '
                        'ON CONFLICT(ticker, date) DO UPDATE SET open=excluded.open, high=excluded.high, '
                        'low=excluded.low, close=excluded.close, volume=excluded.volume',
                        [(ticker, row.Date.strftime("%Y-%m-%d"), float(row.Open), float(row.High),
                          float(row.Low), float(row.Close), float(row.Volume))
                         for row in downloaded[["Date", "Open", "High", "Low", "Close", "Volume"]].itertuples(index=False)],
                    )
                else:
                    connection.executemany(
                        'INSERT INTO prices(ticker, date, close) VALUES (?, ?, ?) '
                        'ON CONFLICT(ticker, date) DO UPDATE SET close=excluded.close',
                        [(ticker, date.strftime("%Y-%m-%d"), float(close))
                         for date, close in downloaded[["Date", "Close"]].itertuples(index=False, name=None)],
                    )
        last = frame["Date"].iloc[-1]
        lag = int(((today-pd.Timedelta(days=1)) - last).days)
        metadata = {"provider": "Yahoo Finance", "ticker": ticker, "cache_path": str(path),
                    "mode": mode, "requested_range": requested, "as_of": last.strftime("%Y-%m-%d"),
                    "missing_recent_days": lag, "warning": warning,
                    "feature_context": "ohlcv" if set(BAR_COLUMNS).issubset(frame.columns) else "close_only"}
        if warning:
            progress(f"WARNING: {warning}. Using cached real prices through {metadata['as_of']}.")
        if lag > 0:
            progress(f"WARNING: Data is {lag} day(s) behind the latest completed UTC day. "
                     "Forecast dates start after the last stored close, not today.")
        progress(f"Data through {metadata['as_of']} ({mode}; {len(frame)} rows)")
    return frame, metadata
