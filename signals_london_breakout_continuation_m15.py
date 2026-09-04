"""
signals_london_breakout_continuation_m15.py
------------------------------------------------
Candidate 2 of the post-Hypothesis-4 structural search: Asian/London
Range Breakout - Continuation. The deliberate mirror opposite of
signals_london_sweep_m15.py (V1, REJECTED) - same Asian-range
infrastructure, same session windows, same buffer/R:R conventions
(reused, not retuned), but the OPPOSITE entry logic: trade WITH a
confirmed breakout, not against it.

Research question: does London-session order flow tend to EXTEND a
genuine overnight Asian-session range breakout (momentum transfer, as
new desks enter aligned with the move) rather than reject it (the
stop-hunt/reversal story V1 tested and V1's own diagnostic found NOT
supported)? This is a genuinely different, opposite economic mechanism
from V1/V2 - not a parameter variation of either.

Rule (short is the exact mirror of long):
  1. Build the Asian session's high/low (00:00-07:00 Europe/London
     LOCAL time, DST-aware) from COMPLETED bars only - REUSES
     signals_london_sweep_m15.py's _session_masks/_compute_asian_range
     UNCHANGED (imported, not copied), including its no-lookahead
     freeze-at-session-close behavior. See that module's docstring for
     the exact mechanism.
  2. During the London entry window (07:00-10:00 Europe/London local
     time, same window as V1), watch for a CONFIRMED breakout: a
     candle's Close beyond the Asian high (bullish continuation) or
     below the Asian low (bearish continuation) by AT LEAST
     breakout_buffer_atr_fraction x ATR. Unlike V1, the breakout
     itself (once confirmed by a CLOSE, not just an intrabar touch) IS
     the entry signal - there is no reversal/reject-then-confirm step,
     because this hypothesis is betting on continuation, not
     rejection.
  3. At most ONE trade per instrument per London-local day - the FIRST
     confirmed breakout in either direction fires; no further bars
     that day are evaluated once a signal fires (same one-trade/day
     discipline as V1).

Deliberately excludes RSI/MACD/EMA or any other indicator confirmation
- the confirmed-close-beyond-the-range structure IS the signal, same
minimalist discipline as V1. No sweep-penetration-style extra
measurement here (that was specific to V1's reversal thesis); nothing
analogous is recorded since there's no "how far beyond, before
reverting" question this hypothesis needs answered.

Stop-loss is STRUCTURAL: placed beyond the OPPOSITE side of the Asian
range (not the side that was broken) - i.e. for a long continuation
entry (breakout above the high), the stop sits below the Asian LOW,
plus a small ATR buffer. This is a deliberate, structural choice: if
price reverses all the way back through the FAR side of the range, the
entire "the range held as support/resistance and flow is continuing"
thesis has failed, not just a shallow pullback - a stop at the near
(broken) boundary would exit on ordinary retest noise. This produces a
WIDER stop than V1's (V1's stop sits just beyond the sweep's own
extreme, close to the entry) - reflected honestly in smaller position
sizes via the unchanged calculate_position_size() risk-per-trade logic,
not adjusted or compensated for here. Take-profit is a fixed 1:1
multiple of that same distance, same convention as V1/V2 - not
searched or tuned for this candidate.

Both stop and target flow through risk_management.py's UNCHANGED
calculate_position_size() via stop_distance_override /
target_distance_override - the same mechanism V1/V2/combined_signals_4h.py
already established. Because a signal on bar T's CLOSE fills at bar
T+1's Open (backtest_engine.py's standard, unmodified fill convention),
this strategy runs through the SAME run_backtest() used everywhere else
in this project - unlike Hypothesis 4's genuine tick-instant trigger,
which needed a standalone simulator. No lookahead: bar T's Close is
fully known at signal time; the fill at T+1's Open is strictly later.

NO LOOKAHEAD (Asian range): identical mechanism and guarantee to
signals_london_sweep_m15.py - reused, not reimplemented. See that
module's docstring and tests/test_london_sweep_signals.py's own
no-lookahead tests, which this module's tests explicitly build on
rather than re-deriving from scratch.
"""

from dataclasses import dataclass

import pandas as pd

from indicators import atr
from signals_london_sweep_m15 import _session_masks, _compute_asian_range


@dataclass
class LondonBreakoutContinuationConfig:
    asian_start_hour: int = 0
    asian_end_hour: int = 7            # exclusive - same Asian session window as V1
    london_window_start_hour: int = 7  # inclusive - same London entry window as V1
    london_window_end_hour: int = 10   # exclusive
    atr_period: int = 14
    breakout_buffer_atr_fraction: float = 0.1  # Close must clear the Asian range by this many ATRs -
                                                # SAME VALUE as V1's stop_buffer_atr_fraction, reused for
                                                # consistency, not independently tuned for this candidate.
    stop_buffer_atr_fraction: float = 0.1      # stop sits this many ATRs beyond the OPPOSITE range boundary
    target_rr_multiple: float = 1.0            # fixed 1:1 reward:risk, same as V1/V2 - not tuned
    spread_avg_window: int = 100


DEFAULT_BREAKOUT_CONTINUATION_CONFIG = LondonBreakoutContinuationConfig()


def _detect_day_breakouts(day_df, config):
    """For ONE instrument's ONE London-local day's London-window bars,
    in chronological order: fire on the FIRST candle whose Close clears
    the Asian range (either side) by the configured ATR buffer, then
    stop entirely - no further bars this day are evaluated once a
    signal fires (one trade per instrument per day, same discipline as
    V1's _detect_day_confirmations)."""
    n = len(day_df)
    signal_long = [False] * n
    signal_short = [False] * n

    rows = list(day_df.itertuples())
    for i, row in enumerate(rows):
        asian_high, asian_low, atr_now = row.asian_high, row.asian_low, row.atr
        if pd.isna(asian_high) or pd.isna(asian_low) or pd.isna(atr_now) or atr_now <= 0:
            continue  # Asian range or ATR not available yet - be safe, same guard as V1

        buffer_distance = config.breakout_buffer_atr_fraction * atr_now

        # --- Bullish continuation: Close clears the Asian high by the buffer ---
        if row.Close > asian_high + buffer_distance:
            signal_long[i] = True
            break  # one trade per instrument per day

        # --- Bearish continuation: Close clears the Asian low by the buffer ---
        if row.Close < asian_low - buffer_distance:
            signal_short[i] = True
            break  # one trade per instrument per day

    return pd.DataFrame({"signal_long": signal_long, "signal_short": signal_short}, index=day_df.index)


def prepare_instrument_frame(m15_df, config=DEFAULT_BREAKOUT_CONTINUATION_CONFIG):
    """Build the M15-indexed signal frame for one instrument: ATR, the
    Asian range (REUSED unchanged from signals_london_sweep_m15.py, see
    module docstring), confirmed-breakout detection within the London
    entry window (one trade/day), the structural stop/target distances
    (opposite-boundary stop, fixed 1:1 target), spread, and the
    long/short entry signals - everything backtest_engine.py needs via
    the same stop_distance_override/target_distance_override mechanism
    already established elsewhere in this project."""
    df = m15_df.copy()

    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    is_asian, is_london_window, date_key, hour = _session_masks(df, config)
    df["asian_high"], df["asian_low"] = _compute_asian_range(df, is_asian, date_key, hour, config)

    df["signal_long"] = False
    df["signal_short"] = False

    london_window_df = df[is_london_window]
    london_window_dates = date_key[is_london_window]
    for _day, day_df in london_window_df.groupby(london_window_dates):
        result = _detect_day_breakouts(day_df, config)
        df.loc[result.index, "signal_long"] = result["signal_long"]
        df.loc[result.index, "signal_short"] = result["signal_short"]

    # --- Structural stop/target distances - only meaningful on rows where a signal fired.
    # Long: stop below the Asian LOW (opposite side of the breakout) minus buffer.
    # Short: stop above the Asian HIGH (opposite side) plus buffer.
    # Reference price is this (confirmation) bar's own Close, same "estimate at
    # signal time" convention used everywhere else in this project - the engine
    # fills at the NEXT bar's open and re-derives absolute levels from the real
    # fill price, preserving this DISTANCE, not this estimated level.
    has_signal = df["signal_long"] | df["signal_short"]
    long_stop_distance = (df["Close"] - df["asian_low"]) + config.stop_buffer_atr_fraction * df["atr"]
    short_stop_distance = (df["asian_high"] - df["Close"]) + config.stop_buffer_atr_fraction * df["atr"]
    structural_distance = long_stop_distance.where(df["signal_long"], short_stop_distance)

    df["stop_distance_override"] = structural_distance.where(has_signal, float("nan"))
    df["target_distance_override"] = (structural_distance * config.target_rr_multiple).where(has_signal, float("nan"))

    # --- Spread, for the existing "is the spread normal?" check in risk_management.py ---
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    return df
