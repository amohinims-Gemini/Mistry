"""
signals_momentum_4h.py
--------------------------
Pure signal-generation logic for a SINGLE-TIMEFRAME 4H TIME-SERIES
MOMENTUM strategy - a genuinely different signal class from everything
else in this project. Channel breakout (signals_4h.py) and its EMA
trend filter react to a specific PRICE LEVEL being crossed (a rolling
high/low, or one EMA crossing another). Mean-reversion
(mean_reversion_signals_4h.py) reacts to price stretching away from a
band. Time-series momentum reacts to neither - it just asks whether
price is higher or lower than it was N bars ago, and trades in that
direction. This is a distinct, well-established family in the
systematic-trading literature (Moskowitz/Ooi/Pedersen-style time-series
momentum), not a variant of anything already tried in this project.

Rule: go long if the trailing `lookback_period`-bar return is positive,
short if negative. Deliberately no magnitude/dead-zone threshold and no
trend/EMA filter layered on top for round 1 - either would just be
re-testing the already-tried EMA-filter idea in a different wrapper.
Since momentum's sign is defined on almost every bar (only exactly zero
would produce neither signal), this strategy wants to be in a position
almost continuously, re-entering in whatever direction momentum points
the moment any prior position closes - a genuinely different trading
character from channel-breakout's/mean-reversion's much sparser
signals, worth knowing going in, not a bug.

Starting lookback (20 bars) matches signals_4h.py's channel window for
comparability across strategies, not because it's assumed correct -
same "start from a literal, unbiased default" discipline used for every
other round-1 config in this project.

ATR stop/target multiples live in RiskConfig, same split as every other
strategy here. Exits are ATR stop/target only - no signal-flip exit -
consistent with how every other strategy in this project works, and
requires no new backtest-engine mechanics.

NO LOOKAHEAD: momentum at bar t compares Close[t] to Close[t-lookback] -
both known as of bar t's own close. No additional shift is needed
beyond what pct_change() already does.
"""

from dataclasses import dataclass

from indicators import atr


@dataclass
class MomentumConfig:
    lookback_period: int = 20   # in 4H bars
    atr_period: int = 14
    spread_avg_window: int = 100


DEFAULT_MOMENTUM_CONFIG = MomentumConfig()


def prepare_instrument_frame(h4_df, config=DEFAULT_MOMENTUM_CONFIG):
    """Build a single 4H-indexed DataFrame with everything
    backtest_engine.py needs: mid/bid/ask OHLC (already in h4_df), ATR,
    trailing-return momentum, spread, and the long/short entry signals -
    all computed from ONE timeframe, no merge_asof required."""
    df = h4_df.copy()

    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    df["momentum"] = df["Close"].pct_change(config.lookback_period)

    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    # NaN during warmup evaluates to False on both comparisons - no
    # signal until momentum is actually computable, same effect as the
    # NaN-channel warmup in signals_4h.py.
    df["signal_long"] = df["momentum"] > 0
    df["signal_short"] = df["momentum"] < 0

    return df
