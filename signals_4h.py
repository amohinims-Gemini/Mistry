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

    # Volatility-expansion confirmation - OFF by default (same pattern as
    # every other optional filter in this project). Requires ATR to be
    # genuinely expanding (above its own recent average), not just a
    # breakout happening during otherwise-normal/quiet conditions. A
    # volatility filter was tried once before on the ORIGINAL 1H strategy
    # (signals.py) and made things worse there - but that was a different
    # construction (1H entry, 4H trend, two clocks) tested before several
    # unrelated fixes landed; worth a genuine re-test on this single-
    # timeframe 4H construction, not assumed to fail the same way.
    use_volatility_filter: bool = False
    volatility_avg_window: int = 20         # matches the channel-period convention used elsewhere
    volatility_expansion_multiple: float = 1.0  # ATR must exceed this x its own rolling average -
                                                  # 1.0 is the simplest literal reading of "expanding
                                                  # above its recent average", not yet a tuned value


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
    signal_long = df["trend_up"] & (df["Close"] > df["breakout_high"])
    signal_short = df["trend_down"] & (df["Close"] < df["breakout_low"])

    # --- Optional volatility-expansion confirmation ---
    # No .shift() needed: ATR (and its own rolling average) legitimately
    # includes the current bar - both are fully known as of this bar's
    # close, the same no-lookahead reasoning already applied to ATR and
    # the Efficiency Ratio elsewhere in this project.
    if config.use_volatility_filter:
        avg_atr = df["atr"].rolling(config.volatility_avg_window).mean()
        volatility_expanding = df["atr"] > config.volatility_expansion_multiple * avg_atr
        signal_long = signal_long & volatility_expanding
        signal_short = signal_short & volatility_expanding

    df["signal_long"] = signal_long
    df["signal_short"] = signal_short

    return df
