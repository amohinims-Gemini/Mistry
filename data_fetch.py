"""
data_fetch.py
--------------
Fetches historical candle data for multiple instruments and multiple
timeframes from OANDA, with local CSV caching so repeated runs don't
re-download years of data every time.

READ-ONLY: only ever calls `instruments.InstrumentsCandles` (a plain GET
request for historical candles). No orders, trades, or account changes
are ever made, and no order/trade/account endpoints are imported.

Requests bid, ask, AND mid prices in one call (OANDA's "BAM" price
component) so the backtest engine can use real historical spreads and
realistic bid/ask fills, rather than a guessed constant spread.

If live fetching fails for any reason (missing/invalid credentials, no
internet, OANDA outage, etc.), this falls back to SYNTHETIC random-walk
data for every instrument so the project still runs end to end. That
fallback is clearly labeled wherever it's used, and none of the
strategy-validity requirements in run_backtest.py (150+ trades, positive
out-of-sample results, etc.) mean anything when run on it - it exists
only to prove the code runs, not to evaluate the strategy.
"""

import os
import time
import datetime as dt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.instruments as instruments_api

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS

load_dotenv()

OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN")
OANDA_ENVIRONMENT = "practice"  # demo account - this project only ever reads candles anyway

PAGE_SIZE = 5000               # OANDA's max candles allowed per request
YEARS_OF_HISTORY = 6           # how far back we *ask* for; actual availability may be less -
                                # we report whatever OANDA actually gives us, honestly
REQUEST_PAUSE_SECONDS = 0.3    # be polite to OANDA between paginated requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def get_client():
    """Create an authenticated, read-only OANDA API client. Raises a
    clear error if the token is missing, so callers can fall back to
    synthetic data instead of crashing with a cryptic error."""
    if not OANDA_API_TOKEN:
        raise RuntimeError(
            "OANDA_API_TOKEN is not set. Copy .env.example to .env and fill "
            "in your OANDA demo account credentials."
        )
    return oandapyV20.API(access_token=OANDA_API_TOKEN, environment=OANDA_ENVIRONMENT)


def _cache_path(oanda_symbol, granularity):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{oanda_symbol}_{granularity}.csv")


def _candles_to_dataframe(candles):
    """Turn OANDA's raw candle JSON (with bid/ask/mid components) into a
    tidy DataFrame indexed by candle OPEN time."""
    rows = []
    for c in candles:
        if not c.get("complete", True):
            continue  # skip any still-forming candle - we only want finished bars
        bid, ask, mid = c["bid"], c["ask"], c["mid"]
        rows.append({
            "Timestamp": pd.to_datetime(c["time"]),
            "Open": float(mid["o"]), "High": float(mid["h"]),
            "Low": float(mid["l"]), "Close": float(mid["c"]),
            "Bid_Open": float(bid["o"]), "Bid_High": float(bid["h"]),
            "Bid_Low": float(bid["l"]), "Bid_Close": float(bid["c"]),
            "Ask_Open": float(ask["o"]), "Ask_High": float(ask["h"]),
            "Ask_Low": float(ask["l"]), "Ask_Close": float(ask["c"]),
            "Volume": float(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.set_index("Timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def _fetch_paginated(client, oanda_symbol, granularity, since):
    """Fetch every complete candle from `since` (a UTC datetime) up to
    now, paginating with the `from`/`includeFirst` params since a single
    request is capped at PAGE_SIZE candles.

    The ONLY reliable "no more data" signal is an EMPTY page. A page
    returning fewer than PAGE_SIZE candles does NOT mean we've reached
    the present - OANDA can legitimately hand back a slightly short page
    for mundane reasons (e.g. a daylight-saving-time week has one fewer
    hourly candle) even when there are years of history still ahead.
    Treating "short page" as "done" was an earlier bug here that silently
    truncated years of history - don't reintroduce that shortcut.
    """
    all_candles = []
    params = {
        "granularity": granularity,
        "price": "BAM",  # Bid, Ask, and Mid OHLC in one response
        "count": PAGE_SIZE,
        "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    max_pages = 200  # safety backstop against a pathological infinite loop, not a real limit
    for _ in range(max_pages):
        request = instruments_api.InstrumentsCandles(instrument=oanda_symbol, params=params)
        client.request(request)  # read-only GET
        page = request.response.get("candles", [])
        page = [c for c in page if c.get("complete", True)]
        if not page:
            break  # truly no more data - this is the only trustworthy stop condition

        all_candles.extend(page)
        print(f"    {oanda_symbol} {granularity}: fetched page of {len(page)} "
              f"(running total: {len(all_candles)})")

        last_time = pd.to_datetime(page[-1]["time"])
        params = {
            "granularity": granularity,
            "price": "BAM",
            "count": PAGE_SIZE,
            "from": last_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "includeFirst": False,  # don't re-fetch the candle we already have
        }
        time.sleep(REQUEST_PAUSE_SECONDS)
    else:
        print(f"    {oanda_symbol} {granularity}: WARNING - hit the {max_pages}-page safety "
              f"cap; stopped early. Re-run to fetch more via the cache's incremental update.")

    return _candles_to_dataframe(all_candles)


def get_instrument_data(oanda_symbol, granularity):
    """
    Return a DataFrame of historical candles for one instrument/timeframe,
    using and updating a local CSV cache:
      - No cache yet -> fetch the full YEARS_OF_HISTORY span from OANDA.
      - Cache exists -> fetch only candles newer than what's cached and
        append them (fast - this is what makes repeated runs quick).

    Note: this does NOT go back and backfill older history if you later
    increase YEARS_OF_HISTORY. Delete the cache file to force a full
    re-fetch if you want more history than what's already cached.
    """
    client = get_client()
    path = _cache_path(oanda_symbol, granularity)

    if os.path.exists(path):
        cached = pd.read_csv(path, index_col="Timestamp", parse_dates=True)
        since = cached.index.max() + pd.Timedelta(seconds=1)
        print(f"  {oanda_symbol} {granularity}: cache has {len(cached)} candles up to "
              f"{cached.index.max()}, fetching anything newer...")
        new_data = _fetch_paginated(client, oanda_symbol, granularity, since)
        if not new_data.empty:
            combined = pd.concat([cached, new_data])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = cached
    else:
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365 * YEARS_OF_HISTORY)
        print(f"  {oanda_symbol} {granularity}: no cache found, fetching up to "
              f"~{YEARS_OF_HISTORY} years of history from {since.date()}...")
        combined = _fetch_paginated(client, oanda_symbol, granularity, since)

    combined.to_csv(path)
    return combined


def generate_synthetic_data(oanda_symbol, granularity, periods, start_price, seed):
    """
    SYNTHETIC (fake) OHLC + bid/ask data for one instrument/timeframe - an
    independent random walk, NOT correlated with other instruments and
    NOT reflecting any real historical market regime.

    Used only so the project still runs end to end without real OANDA
    credentials. None of run_backtest.py's strategy-validity checks
    (150+ trades, positive out-of-sample results, drawdown, profit
    factor) mean anything on this data - it exists purely to prove the
    code runs, not to evaluate the strategy.
    """
    rng = np.random.default_rng(seed)

    # Rough per-candle volatility scaled by timeframe (4H moves ~2x a 1H
    # candle, daily ~4x - not exactly sqrt(n) since real markets aren't a
    # pure random walk, but close enough for a smoke-test fallback).
    base_hourly_vol = 0.0006 if start_price < 100 else 0.0004  # forex vs gold/JPY-scale prices
    vol_scale = {"H4": 2.0, "D": 4.0}.get(granularity, 1.0)
    vol = base_hourly_vol * vol_scale

    returns = rng.normal(loc=0.0, scale=vol, size=periods)
    close = start_price * np.cumprod(1 + returns)

    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, vol / 2, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, vol / 2, periods))
    volume = rng.uniform(100, 1000, periods)

    # A rough, fixed synthetic spread (1.5 pips-equivalent) around mid.
    spec = next(s for s in INSTRUMENTS.values() if s.oanda_symbol == oanda_symbol)
    half_spread = spec.pip_size * 0.75

    freq = {"H4": "4h", "D": "1D"}.get(granularity, "h")
    index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq=freq)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Bid_Open": open_ - half_spread, "Bid_High": high - half_spread,
        "Bid_Low": low - half_spread, "Bid_Close": close - half_spread,
        "Ask_Open": open_ + half_spread, "Ask_High": high + half_spread,
        "Ask_Low": low + half_spread, "Ask_Close": close + half_spread,
        "Volume": volume,
    }, index=index)
    df.index.name = "Timestamp"
    return df


# Rough starting prices for synthetic data only, so fallback series look
# plausible (not used anywhere real data is available).
_SYNTHETIC_START_PRICES = {
    "EUR_USD": 1.10, "GBP_USD": 1.27, "USD_JPY": 150.0, "XAU_USD": 2400.0,
}


def fetch_all(instrument_symbols, granularities=("H1", "H4")):
    """
    Fetch (or update) cached candle data for every instrument x timeframe
    combination requested. Returns (data, is_synthetic) where data is
    {symbol: {granularity: DataFrame}}.

    If ANY live fetch fails partway through (network drop, bad
    credentials, etc.), the WHOLE result falls back to synthetic data for
    every instrument/timeframe - kept simple and predictable rather than
    mixing some real, some fake series together.
    """
    result = {}
    try:
        for symbol in instrument_symbols:
            spec = INSTRUMENTS[symbol]
            result[symbol] = {}
            print(f"\n{symbol}:")
            for granularity in granularities:
                result[symbol][granularity] = get_instrument_data(spec.oanda_symbol, granularity)
        return result, False
    except Exception as e:
        print(f"\nCould not fetch live OANDA data ({e}).")
        print("Falling back to SYNTHETIC sample data for ALL instruments (NOT real prices).")
        synthetic = {}
        periods_by_granularity = {
            "H1": 365 * YEARS_OF_HISTORY * 16, "H4": 365 * YEARS_OF_HISTORY * 4,
            "D": 365 * YEARS_OF_HISTORY,
        }
        for i, symbol in enumerate(instrument_symbols):
            spec = INSTRUMENTS[symbol]
            synthetic[symbol] = {}
            for granularity in granularities:
                synthetic[symbol][granularity] = generate_synthetic_data(
                    spec.oanda_symbol, granularity,
                    periods=periods_by_granularity[granularity],
                    start_price=_SYNTHETIC_START_PRICES.get(symbol, 1.0),
                    seed=hash((symbol, granularity)) % (2**32 - 1),
                )
        return synthetic, True


if __name__ == "__main__":
    data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS)
    label = "SYNTHETIC" if is_synthetic else "REAL"
    print(f"\n=== {label} DATA SUMMARY ===")
    for symbol, tfs in data.items():
        for granularity, df in tfs.items():
            span = f"{df.index.min()} to {df.index.max()}" if not df.empty else "EMPTY"
            print(f"{symbol} {granularity}: {len(df)} candles ({span})")
