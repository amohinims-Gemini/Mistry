"""
indicators.py
--------------
Plain pandas indicator functions shared by the trend filter (4H EMAs),
entry trigger (1H breakout channel), and stop-loss sizing (1H ATR).

No lookahead protection is baked in here - these functions just compute
the raw indicator values. It's signals.py's job to shift/align things so
a decision made "at" time T only ever uses information that was actually
available at T.
"""

import pandas as pd


def ema(close, period):
    """Exponential Moving Average - like a simple average, but weights
    recent prices more heavily so it reacts faster to new trends."""
    return close.ewm(span=period, adjust=False).mean()


def atr(high, low, close, period=14):
    """
    Average True Range - a volatility measure, used here to size
    stop-losses/take-profits relative to how much the market is actually
    moving (a wide ATR means wider stops, so normal noise doesn't stop us
    out; a narrow ATR means tighter stops).

    True Range for one candle is the largest of:
      - high - low                    (this candle's own range)
      - abs(high - previous close)    (a gap up from the prior close)
      - abs(low - previous close)     (a gap down from the prior close)
    ATR is simply a rolling average of True Range.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def rolling_high(high, period):
    """Highest High over the last `period` candles (INCLUDING the
    current one - see signals.py, which shifts this to exclude it) -
    the basis for the breakout entry trigger."""
    return high.rolling(period).max()


def rolling_low(low, period):
    """Lowest Low over the last `period` candles - mirror of
    rolling_high, for short entries."""
    return low.rolling(period).min()
