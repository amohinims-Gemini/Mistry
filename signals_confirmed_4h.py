"""
signals_confirmed_4h.py
---------------------------
Requires TWO INDEPENDENT signals to agree before entering - trend-
following's channel breakout (signals_4h.py) AND time-series momentum
(signals_momentum_4h.py) must both point the same direction on the same
bar. Neither signal class cleared the validation bars alone (see
README's "Systematic strategy search"), but they've never been tested
as a joint confirmation filter - the hypothesis is that the intersection
of two independently-reasoned conditions is more selective than either
alone, even if neither has enough edge by itself.

Both source signals are computed on the SAME 4H series with their own
existing, already-tested default parameters (EMA 50/200 + 20-bar
channel for trend; 20-bar trailing-return sign for momentum) -
deliberately NOT re-tuned for this combination, since the question this
round is "does requiring agreement help," not "what are the best
parameters for each side." Both modules already default to a 20-bar
lookback/channel and 14-period ATR, so their OHLC/ATR/spread columns are
identical and can be taken from either source frame.

Combination rule: signal_long = trend.signal_long AND momentum.signal_long
(mirror for short). No "OR" case, no tie-break needed - this is pure
intersection, stricter than either input alone by construction, so
trade frequency will be LOWER than either standalone strategy, not
higher (the opposite of combined_signals_4h.py's OR-based portfolio
combination, which took a signal if EITHER strategy fired).

NO LOOKAHEAD: both source signals are already no-lookahead by
construction (see their own docstrings); ANDing them introduces none of
its own.
"""

from signals_4h import prepare_instrument_frame as _trend_frame, Signal4HConfig
from signals_momentum_4h import prepare_instrument_frame as _momentum_frame, MomentumConfig

_BASE_COLUMNS = [
    "Open", "High", "Low", "Close",
    "Bid_Open", "Bid_High", "Bid_Low", "Bid_Close",
    "Ask_Open", "Ask_High", "Ask_Low", "Ask_Close",
    "atr", "spread_close", "avg_spread_100",
]


def prepare_instrument_frame(h4_df, trend_config=Signal4HConfig(), momentum_config=MomentumConfig()):
    """Build the confirmed-entry frame for ONE instrument's 4H data -
    trend-following's signal AND momentum's signal, both required."""
    trend_df = _trend_frame(h4_df, config=trend_config)
    momentum_df = _momentum_frame(h4_df, config=momentum_config)

    df = trend_df[_BASE_COLUMNS].copy()

    trend_long = trend_df["signal_long"].fillna(False)
    trend_short = trend_df["signal_short"].fillna(False)
    momentum_long = momentum_df["signal_long"].reindex(df.index).fillna(False)
    momentum_short = momentum_df["signal_short"].reindex(df.index).fillna(False)

    df["signal_long"] = trend_long & momentum_long
    df["signal_short"] = trend_short & momentum_short

    return df
