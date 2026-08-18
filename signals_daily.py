"""
signals_daily.py
--------------------
Pure signal-generation logic for the SINGLE-TIMEFRAME DAILY trend-
following strategy - the daily-timeframe fallback, parallel to
signals_4h.py (single 4H) and the original signals.py (4H trend/1H
entry). Trend filter and entry trigger are both computed from the same
daily series, no merge_asof required.

Parameters carried forward LITERALLY from signals_4h.py (EMA 50/200,
20-bar channel, ATR period 14) rather than rescaled by clock-hours -
unlike the 1H->4H pivot, this is arguably MORE natural on daily than it
was on 4H: 50-day/200-day EMA is the standard "golden/death cross"
convention, and a 20-day channel is a standard monthly breakout window.
Re-tuning is still expected to happen via sweeps, same discipline as
every other pivot in this project - these are the round-1 baseline, not
a claim that they're already correct for daily volatility.

ATR-based stop-loss/take-profit MULTIPLES reset to the literal original
spec numbers (1.5x/3.0x) in RiskConfig - NOT the 4H-tuned values, same
reasoning applied when 1H moved to 4H: a different timeframe's noise
characteristics shouldn't inherit another timeframe's tuning.

NO LOOKAHEAD: everything here uses only data up to and including the
current daily bar's close.
"""

from dataclasses import dataclass

from indicators import ema, atr, rolling_high, rolling_low


@dataclass
class SignalDailyConfig:
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    channel_period: int = 20
    atr_period: int = 14
    spread_avg_window: int = 100


DEFAULT_SIGNAL_DAILY_CONFIG = SignalDailyConfig()


def prepare_instrument_frame(daily_df, config=DEFAULT_SIGNAL_DAILY_CONFIG):
    """Build a single daily-indexed DataFrame with everything
    backtest_engine.py needs: mid/bid/ask OHLC (already in daily_df),
    ATR, EMA trend, breakout channel, spread, and long/short entry
    signals - all computed from ONE timeframe."""
    df = daily_df.copy()

    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    ema_fast = ema(df["Close"], config.ema_fast_period)
    ema_slow = ema(df["Close"], config.ema_slow_period)
    df["trend_up"] = ema_fast > ema_slow
    df["trend_down"] = ema_fast < ema_slow

    df["breakout_high"] = rolling_high(df["High"], config.channel_period).shift(1)
    df["breakout_low"] = rolling_low(df["Low"], config.channel_period).shift(1)

    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    df["signal_long"] = df["trend_up"] & (df["Close"] > df["breakout_high"])
    df["signal_short"] = df["trend_down"] & (df["Close"] < df["breakout_low"])

    return df
