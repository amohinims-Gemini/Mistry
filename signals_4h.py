"""
signals_4h.py
----------------
Pure signal-generation logic for the SINGLE-TIMEFRAME 4H trend-following
strategy - trend filter AND entry trigger both computed from the SAME 4H
series, unlike signals.py (4H trend filter / 1H entry trigger, two
different clocks).

Built specifically to remove a diagnosed structural blind spot: a market
that satisfies the 4H trend filter while chopping on a faster entry
timeframe could let bad entries through, and no filter tried on the
faster timeframe (efficiency ratio, volatility, breakout buffer) fully
fixed that - see README.md's "Tuning history" for the full record. Using
ONE timeframe for both trend and entry removes the mismatch at its
source instead of patching around it on the faster side.

Rules (long; short is the exact mirror):
  1. Trend filter: 50 EMA above 200 EMA (both on 4H)
  2. Entry trigger: this 4H bar's close breaks above the highest High of
     the previous N COMPLETED 4H candles
  3. Spread within a normal range for the instrument (checked in
     risk_management.py, same mechanism as every other strategy here)

Starting parameters (EMA 50/200, channel=20, ATR 14) are the literal
original spec values, never tuned for any specific timeframe - a clean,
unbiased round-1 baseline, not a reused 1H-tuned number.

NO LOOKAHEAD: everything here uses only data up to and including the
current 4H bar's close. No cross-timeframe merge is needed at all
(unlike signals.py) - one of the simplifications of using a single
timeframe, not just a workaround.
"""

from dataclasses import dataclass

from indicators import ema, atr, rolling_high, rolling_low


@dataclass
class Signal4HConfig:
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    channel_period: int = 20
    atr_period: int = 14
    spread_avg_window: int = 100


DEFAULT_SIGNAL_4H_CONFIG = Signal4HConfig()


def prepare_instrument_frame(h4_df, config=DEFAULT_SIGNAL_4H_CONFIG):
    """
    Build a single 4H-indexed DataFrame with everything backtest_engine.py
    needs: mid/bid/ask OHLC (already in h4_df), ATR, the trend filter, the
    breakout channel, spread, and the long/short entry signals - all
    computed from ONE timeframe, no merge_asof required.

    Produces the same required columns as signals.prepare_instrument_frame
    (atr, spread_close, avg_spread_100, signal_long, signal_short) so
    backtest_engine.py works unchanged regardless of which strategy's
    frame it's given.
    """
    df = h4_df.copy()

    # --- Volatility, for stop-loss/take-profit sizing ---
    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    # --- Trend filter: both EMAs on the SAME 4H series as the entry trigger ---
    ema_fast = ema(df["Close"], config.ema_fast_period)
    ema_slow = ema(df["Close"], config.ema_slow_period)
    df["trend_up"] = ema_fast > ema_slow
    df["trend_down"] = ema_fast < ema_slow

    # --- Entry trigger: breakout of the last N COMPLETED 4H candles.
    # .shift(1) drops the current bar, so "highest of the previous N"
    # really means previous, not "these N including now".
    df["breakout_high"] = rolling_high(df["High"], config.channel_period).shift(1)
    df["breakout_low"] = rolling_low(df["Low"], config.channel_period).shift(1)

    # --- Spread, for the "is the spread normal?" check in risk_management.py ---
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    # --- The two mirrored entry rules ---
    df["signal_long"] = df["trend_up"] & (df["Close"] > df["breakout_high"])
    df["signal_short"] = df["trend_down"] & (df["Close"] < df["breakout_low"])

    return df
