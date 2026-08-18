"""
combined_signals_4h.py
--------------------------
Merges the two SINGLE-TIMEFRAME 4H strategies (signals_4h.py trend-
following, mean_reversion_signals_4h.py mean-reversion) into ONE
combined signal frame per instrument, so both can trade out of a single
shared PortfolioAccount at once, sharing every existing risk/safety
limit (max open positions, total open risk cap, correlation groups,
daily/weekly/drawdown circuit breakers) - see run_combined_backtest_4h.py.

Motivation - measured BEFORE building this, not assumed: standalone,
each strategy is close to breakeven and neither clears the 4 validation
bars alone, but a direct check of their historical trade records found
the daily P&L correlation between the two is -0.02 (essentially zero),
and only ~2.7% of trades land on the same instrument on the same day.
They fire under close-to-mutually-exclusive conditions by construction
- trend-following needs a breakout WITH the EMA trend; mean-reversion
needs a low-efficiency, range-bound market stretched to a Bollinger
extreme - so this isn't two versions of the same edge, and a genuine
diversification benefit is at least plausible.

Combination rule, per instrument, per bar:
  - Only trend fires               -> take it, signal_source="trend",
                                       sized with the TREND strategy's
                                       own ATR stop/target multiples.
  - Only mean-reversion fires      -> take it, signal_source=
                                       "mean_reversion", sized with MR's
                                       own multiples.
  - Both fire, SAME direction      -> take it once, signal_source=
                                       "both", sized with the trend
                                       multiples (an arbitrary but
                                       documented tiebreak - irrelevant
                                       in practice, given how rarely the
                                       two agree at all).
  - Both fire, OPPOSITE directions -> skip. Genuine ambiguity (one
                                       strategy says buy, the other says
                                       sell, at the same instrument, same
                                       bar) - not worth guessing at, and
                                       rarer still than the "both agree"
                                       case.

What this module does NOT do: it does not duplicate or split the
account-level risk limits between the two strategies. Both strategies
draw from the SAME shared PortfolioAccount, the SAME max-open-positions/
total-open-risk-cap/correlation-group pool, exactly as if a single
strategy were trading all 4 instruments - that sharing is the entire
point of combining them. This module's only job is deciding, per
instrument per bar, WHICH signal (if any) is on offer and what stop/
target distance applies if it's taken. backtest_engine.py's existing
"one open position per symbol at a time" rule is what actually prevents
the two strategies from double-trading the same instrument - no special
handling needed here for that case.

NO LOOKAHEAD: both source frames are already built with no lookahead
(see signals_4h.py / mean_reversion_signals_4h.py); merging them
row-by-row on the same index introduces none of its own.
"""

import numpy as np

from signals_4h import prepare_instrument_frame as _trend_frame, Signal4HConfig
from mean_reversion_signals_4h import prepare_instrument_frame as _mr_frame, MeanReversion4HConfig

# Columns backtest_engine.py actually needs, carried over from the trend
# frame as the shared base (OHLC/bid-ask/atr/spread are identical
# regardless of which strategy "owns" a given bar's signal - both source
# frames are built from the same underlying h4_df with the same
# atr_period default of 14).
_BASE_COLUMNS = [
    "Open", "High", "Low", "Close",
    "Bid_Open", "Bid_High", "Bid_Low", "Bid_Close",
    "Ask_Open", "Ask_High", "Ask_Low", "Ask_Close",
    "atr", "spread_close", "avg_spread_100",
]


def build_combined_frame(
    h4_df,
    trend_config=Signal4HConfig(),
    trend_stop_atr_multiple=1.5,
    trend_target_atr_multiple=3.0,
    mr_config=MeanReversion4HConfig(),
    mr_stop_atr_multiple=1.5,
    mr_target_atr_multiple=1.0,
):
    """Build the combined signal frame for ONE instrument's 4H data.
    `trend_stop_atr_multiple`/`trend_target_atr_multiple` and their `mr_`
    counterparts are each strategy's OWN R:R, applied only to entries
    that strategy generates (see module docstring) - kept as explicit
    parameters here rather than buried in a RiskConfig, since a single
    RiskConfig object no longer has "the" ATR multiple once two
    strategies with two different R:Rs are trading side by side.
    """
    trend_df = _trend_frame(h4_df, config=trend_config)
    mr_df = _mr_frame(h4_df, config=mr_config)

    df = trend_df[_BASE_COLUMNS].copy()

    trend_long = trend_df["signal_long"].fillna(False)
    trend_short = trend_df["signal_short"].fillna(False)
    mr_long = mr_df["signal_long"].reindex(df.index).fillna(False)
    mr_short = mr_df["signal_short"].reindex(df.index).fillna(False)

    # Opposite-direction disagreement on the same instrument, same bar -
    # skip both rather than guess which one to believe.
    conflict = (trend_long & mr_short) | (trend_short & mr_long)

    trend_fires = (trend_long | trend_short) & ~conflict
    mr_fires = (mr_long | mr_short) & ~conflict
    mr_only_fires = mr_fires & ~trend_fires
    both_agree = trend_fires & mr_fires

    df["signal_long"] = (trend_long | mr_long) & ~conflict
    df["signal_short"] = (trend_short | mr_short) & ~conflict

    df["signal_source"] = np.select(
        [both_agree, trend_fires, mr_only_fires],
        ["both", "trend", "mean_reversion"],
        default="none",
    )

    df["stop_distance_override"] = np.select(
        [trend_fires, mr_only_fires],
        [trend_stop_atr_multiple * df["atr"], mr_stop_atr_multiple * df["atr"]],
        default=np.nan,
    )
    df["target_distance_override"] = np.select(
        [trend_fires, mr_only_fires],
        [trend_target_atr_multiple * df["atr"], mr_target_atr_multiple * df["atr"]],
        default=np.nan,
    )

    return df
