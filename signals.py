"""
signals.py
-----------
Pure signal-generation logic: given prepared 4H and 1H OANDA candle data
for one instrument, produce a per-1H-bar "long entry available" / "short
entry available" signal based on the strategy's technical rules:

  1. Trend filter (4H): fast EMA above slow EMA = uptrend, longs allowed
     (mirror: fast EMA below slow EMA = downtrend, shorts allowed)
  2. Entry trigger (1H): this bar's close breaks above the highest High
     of the previous N COMPLETED 1H candles (mirror: breaks below the
     lowest Low of the previous N)
  3. (Optional, off by default) Volatility filter: skip signals when ATR
     is below its own trailing average - avoids taking breakouts during
     unusually quiet, chop-prone conditions.
  4. (Optional, off by default) Breakout buffer: require the close to
     clear the breakout level by a small ATR-scaled margin, rather than
     by any amount at all - filters out marginal/ambiguous breakouts.

All of this is configurable via SignalConfig so run_backtest.py can try
several parameter/filter combinations without editing this file.

*** channel_period default is 95, not the spec's originally-stated 20. ***
This is a deliberate, evidence-based deviation, not an oversight: a
channel-length sweep (20 to 100) followed by an 11-scenario robustness
stress test (different train/test splits AND independent historical
windows, not just the one split it was found on) showed 95 consistently
improves profit factor, drawdown, and return ON AVERAGE across most
scenarios - not just the split it happened to be tuned on. It still does
NOT make the strategy reliably clear the spec's profit-factor bar in
every market regime (worst-case scenario in that stress test: PF 0.735),
so treat this as "meaningfully better than 20," not "solved." See
README.md for the full before/after numbers. The other two filters (#3,
#4) remain OFF by default - they were tried and made things worse.

This file only decides "is there a technically valid setup right now" -
it doesn't know about spreads, account risk, or safety limits. That's
risk_management.py's job. Keeping the two separate means the technical
strategy logic can be reasoned about and tested on its own.

NO LOOKAHEAD: every indicator here is computed so a value "as of" a given
1H bar only ever uses information that was actually available by that
bar's close:
  - the 4H trend is merged onto the 1H timeline using only 4H candles
    that had already CLOSED (via merge_asof, direction="backward", on an
    index shifted forward by one 4H period - see prepare_4h_trend)
  - the breakout channel and the volatility filter's own trailing
    average both explicitly exclude the current bar (via .shift(1))
"""

from dataclasses import dataclass

import pandas as pd
from indicators import ema, atr, rolling_high, rolling_low


@dataclass
class SignalConfig:
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    channel_period: int = 95  # was 20 in the original spec - see module docstring for why
    atr_period: int = 14
    spread_avg_window: int = 100

    # Entry-filter additions - both OFF by default (matches the original spec)
    use_volatility_filter: bool = False
    volatility_filter_window: int = 100      # ATR's own trailing-average window
    breakout_buffer_atr_fraction: float = 0.0  # e.g. 0.1 = must clear the level by 0.1x ATR


DEFAULT_SIGNAL_CONFIG = SignalConfig()


def prepare_4h_trend(h4_df, config=DEFAULT_SIGNAL_CONFIG):
    """Compute the 4H trend filter. Returns a DataFrame whose index has
    been shifted forward by 4 hours - from "this candle's OPEN time" to
    "this candle's CLOSE time" - so that merge_asof in
    prepare_instrument_frame() naturally lines each 1H bar up with the
    most recently *closed* 4H candle. A 4H candle's trend reading isn't
    "known" until it actually finishes."""
    out = pd.DataFrame(index=h4_df.index)
    out["ema_fast"] = ema(h4_df["Close"], config.ema_fast_period)
    out["ema_slow"] = ema(h4_df["Close"], config.ema_slow_period)
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]
    out["trend_down"] = out["ema_fast"] < out["ema_slow"]

    out.index = out.index + pd.Timedelta(hours=4)
    return out


def prepare_instrument_frame(h1_df, h4_df, config=DEFAULT_SIGNAL_CONFIG):
    """
    Combine 1H and 4H data for one instrument into a single 1H-indexed
    DataFrame with everything the backtest engine needs: mid/bid/ask
    OHLC, ATR, the breakout channel, the 4H trend filter, spread, and the
    long/short technical entry signals.
    """
    df = h1_df.copy()

    # --- Volatility, for stop-loss/take-profit sizing AND (optionally) the
    # volatility filter below.
    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    # --- Entry trigger: breakout of the last N COMPLETED candles.
    # rolling(...).max() over the last channel_period bars includes the
    # CURRENT bar by default - .shift(1) drops it, so "highest of the
    # previous N" really means previous, not "these N including now".
    df["breakout_high"] = rolling_high(df["High"], config.channel_period).shift(1)
    df["breakout_low"] = rolling_low(df["Low"], config.channel_period).shift(1)

    # --- Spread, for the "is the spread normal?" check in risk_management.py.
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    # shift(1): the trailing average must not include the current bar's own spread
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    # --- 4H trend filter, aligned onto the 1H index without lookahead.
    trend = prepare_4h_trend(h4_df, config)
    df = pd.merge_asof(
        df.sort_index(), trend.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    # merge_asof leaves NaN for 1H bars before any 4H trend data existed yet
    # (the first ema_slow_period 4H candles' worth of warm-up), which turns
    # these columns into object dtype - explicitly convert back to a clean
    # bool (treating "trend not yet known" as "not trending") rather than
    # relying on plain .fillna(), which pandas now warns is ambiguous here.
    df["trend_up"] = df["trend_up"].astype("boolean").fillna(False).astype(bool)
    df["trend_down"] = df["trend_down"].astype("boolean").fillna(False).astype(bool)

    # --- Optional volatility filter: only signal when ATR is above its own
    # trailing average (skip unusually quiet, chop-prone stretches). shift(1)
    # so the trailing average never includes the current bar's own ATR.
    if config.use_volatility_filter:
        atr_trailing_avg = df["atr"].rolling(config.volatility_filter_window).mean().shift(1)
        volatility_ok = df["atr"] > atr_trailing_avg
    else:
        volatility_ok = pd.Series(True, index=df.index)

    # --- Optional breakout buffer: require the close to clear the breakout
    # level by a small ATR-scaled margin rather than by any amount. Uses
    # THIS bar's own ATR, which is fine - ATR at bar T is built from True
    # Range values up to and including bar T, all known by bar T's close.
    long_breakout_level = df["breakout_high"] + config.breakout_buffer_atr_fraction * df["atr"]
    short_breakout_level = df["breakout_low"] - config.breakout_buffer_atr_fraction * df["atr"]

    # --- The two mirrored technical entry rules, plus whichever optional
    # filters are enabled. (Rule 3 - spread - is checked in
    # risk_management.py, since it's a risk/execution concern rather than
    # "is there a technical setup".)
    df["signal_long"] = df["trend_up"] & (df["Close"] > long_breakout_level) & volatility_ok
    df["signal_short"] = df["trend_down"] & (df["Close"] < short_breakout_level) & volatility_ok

    return df
