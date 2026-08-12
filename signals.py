"""
signals.py
-----------
Pure signal-generation logic: given prepared 4H and 1H OANDA candle data
for one instrument, produce a per-1H-bar "long entry available" / "short
entry available" signal based on the strategy's technical rules:

  1. Trend filter (4H): 50 EMA above 200 EMA = uptrend, longs allowed
     (mirror: 50 EMA below 200 EMA = downtrend, shorts allowed)
  2. Entry trigger (1H): this bar's close breaks above the highest High
     of the previous 20 COMPLETED 1H candles (mirror: breaks below the
     lowest Low of the previous 20)

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
  - the 20-bar breakout channel explicitly excludes the current bar
    (via .shift(1))
"""

import pandas as pd
from indicators import ema, atr, rolling_high, rolling_low

EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
CHANNEL_PERIOD = 20
ATR_PERIOD = 14
SPREAD_AVG_WINDOW = 100


def prepare_4h_trend(h4_df):
    """Compute the 4H trend filter. Returns a DataFrame whose index has
    been shifted forward by 4 hours - from "this candle's OPEN time" to
    "this candle's CLOSE time" - so that merge_asof in
    prepare_instrument_frame() naturally lines each 1H bar up with the
    most recently *closed* 4H candle. A 4H candle's trend reading isn't
    "known" until it actually finishes."""
    out = pd.DataFrame(index=h4_df.index)
    out["ema_fast"] = ema(h4_df["Close"], EMA_FAST_PERIOD)
    out["ema_slow"] = ema(h4_df["Close"], EMA_SLOW_PERIOD)
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]
    out["trend_down"] = out["ema_fast"] < out["ema_slow"]

    out.index = out.index + pd.Timedelta(hours=4)
    return out


def prepare_instrument_frame(h1_df, h4_df):
    """
    Combine 1H and 4H data for one instrument into a single 1H-indexed
    DataFrame with everything the backtest engine needs: mid/bid/ask
    OHLC, ATR, the 20-bar breakout channel, the 4H trend filter, spread,
    and the long/short technical entry signals.
    """
    df = h1_df.copy()

    # --- Volatility, for stop-loss/take-profit sizing ---
    df["atr"] = atr(df["High"], df["Low"], df["Close"], ATR_PERIOD)

    # --- Entry trigger: breakout of the last N COMPLETED candles.
    # rolling(...).max() over the last CHANNEL_PERIOD bars includes the
    # CURRENT bar by default - .shift(1) drops it, so "highest of the
    # previous 20" really means previous, not "these 20 including now".
    df["rolling_high_20"] = rolling_high(df["High"], CHANNEL_PERIOD).shift(1)
    df["rolling_low_20"] = rolling_low(df["Low"], CHANNEL_PERIOD).shift(1)

    # --- Spread, for the "is the spread normal?" check in risk_management.py.
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    # shift(1): the trailing average must not include the current bar's own spread
    df["avg_spread_100"] = df["spread_close"].rolling(SPREAD_AVG_WINDOW).mean().shift(1)

    # --- 4H trend filter, aligned onto the 1H index without lookahead.
    trend = prepare_4h_trend(h4_df)
    df = pd.merge_asof(
        df.sort_index(), trend.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    # merge_asof leaves NaN for 1H bars before any 4H trend data existed yet
    # (the first ~200 4H candles' worth of warm-up), which turns these
    # columns into object dtype - explicitly convert back to a clean bool
    # (treating "trend not yet known" as "not trending") rather than
    # relying on plain .fillna(), which pandas now warns is ambiguous here.
    df["trend_up"] = df["trend_up"].astype("boolean").fillna(False).astype(bool)
    df["trend_down"] = df["trend_down"].astype("boolean").fillna(False).astype(bool)

    # --- The two mirrored technical entry rules. (Rule 3 - spread - is
    # checked in risk_management.py, since it's a risk/execution concern
    # rather than "is there a technical setup".)
    df["signal_long"] = df["trend_up"] & (df["Close"] > df["rolling_high_20"])
    df["signal_short"] = df["trend_down"] & (df["Close"] < df["rolling_low_20"])

    return df
