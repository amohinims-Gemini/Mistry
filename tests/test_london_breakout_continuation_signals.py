"""
test_london_breakout_continuation_signals.py
------------------------------------------------
Tests for signals_london_breakout_continuation_m15.py ("candidate 2")
and run_london_breakout_continuation_backtest.py's dataset-access
discipline. The Asian-range no-lookahead mechanism itself is REUSED
unchanged from signals_london_sweep_m15.py (imported, not copied) and
is already exhaustively covered by test_london_sweep_signals.py
(BST/GMT transitions, session boundaries, freeze-at-close behavior) -
this file confirms that reuse holds (a couple of direct checks) rather
than re-deriving all of that coverage from scratch, and focuses new
tests on what's actually NEW here: confirmed-breakout (continuation)
detection, one-trade-per-day, the opposite-boundary structural stop,
and validation/reserved-access prevention.

All synthetic, hand-constructed data - no real OANDA data or cached
files are touched by this file.
"""

import pandas as pd
import pytest

import dataset_split
from signals_london_breakout_continuation_m15 import (
    prepare_instrument_frame, LondonBreakoutContinuationConfig,
)
from tests.test_london_sweep_signals import make_m15_frame, set_bar, london_dt, WEEK_START, WEEK_END, base_week_frame


# =============================================================================
# 1. Asian-range infrastructure reuse (no-lookahead already proven in
#    test_london_sweep_signals.py - these two checks confirm the SAME
#    guarantee holds through this module's own prepare_instrument_frame)
# =============================================================================

def test_asian_range_columns_present_and_frozen_after_close():
    """Sanity check that this module's frame carries the same
    asian_high/asian_low columns, computed via the reused (not copied)
    _compute_asian_range - frozen for the rest of the day once set."""
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 3, 3, 0), High=1.1050, Low=1.0950)  # inside Asian session

    frame = prepare_instrument_frame(df)

    # After the Asian session closes (07:00 London local), the range must
    # reflect the 03:00 bar's extremes.
    london_idx = frame.index.tz_convert("Europe/London")
    post_close_mask = (london_idx.date == london_dt(2024, 6, 3, 7).date()) & (london_idx.hour == 7)
    post_close_rows = frame[post_close_mask]
    assert len(post_close_rows) > 0
    assert (post_close_rows["asian_high"] == 1.1050).all()
    assert (post_close_rows["asian_low"] == 1.0950).all()


def test_asian_range_not_yet_known_before_session_closes():
    """A bar strictly before 07:00 London local on the SAME day must not
    see that day's own (still-forming) Asian range yet - identical
    no-lookahead guarantee as V1, exercised through this module."""
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 3, 3, 0), High=1.1050, Low=1.0950)

    frame = prepare_instrument_frame(df)
    london_idx = frame.index.tz_convert("Europe/London")
    before_close_mask = (london_idx.date == london_dt(2024, 6, 3, 7).date()) & (london_idx.hour == 6)
    before_close_rows = frame[before_close_mask]
    assert len(before_close_rows) > 0
    assert before_close_rows["asian_high"].isna().all()
    assert before_close_rows["asian_low"].isna().all()


# =============================================================================
# 2. Confirmed-breakout (continuation) detection
# =============================================================================

def _build_asian_range(df, high=1.1030, low=1.0970):
    set_bar(df, london_dt(2024, 6, 5, 3, 0), High=high, Low=low)
    return high, low


def test_close_clearing_asian_high_by_buffer_fires_long_signal():
    """Note on the two-pass pattern used throughout this file: the first
    prepare_instrument_frame() call is only to read off a representative
    ATR value for sizing the test's price offsets - at that point the
    breakout bar hasn't been placed yet, so this pre-pass ATR can
    legitimately be 0 (the earlier Asian-range bar's own True Range has
    long since rolled out of the 14-bar window by the frame's last
    timestamp, which is days later). The REAL assertion is against a
    SECOND prepare_instrument_frame() call after the breakout bar is
    set, which correctly recomputes ATR including that bar's own large
    True Range - the same two-pass approach every test below uses."""
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]

    # Close well beyond high + buffer*ATR
    buffer = 0.1 * atr_now
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=high + buffer + 0.0010, High=high + buffer + 0.0010)
    frame = prepare_instrument_frame(df)

    assert frame.loc[ts, "signal_long"]
    assert not frame.loc[ts, "signal_short"]


def test_close_clearing_asian_low_by_buffer_fires_short_signal():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]

    buffer = 0.1 * atr_now
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=low - buffer - 0.0010, Low=low - buffer - 0.0010)
    frame = prepare_instrument_frame(df)

    assert frame.loc[ts, "signal_short"]
    assert not frame.loc[ts, "signal_long"]


def test_close_just_inside_buffer_does_not_fire():
    """A close that clears the raw Asian high but NOT by the full ATR
    buffer must not fire - the buffer is a required minimum, not a
    rounding margin. Uses a same-bar ATR read (a placeholder breakout
    bar first, to get a self-consistent ATR at that exact timestamp -
    ATR includes the current bar's own True Range, matching the exact
    convention _detect_day_breakouts itself uses) rather than a stale
    pre-pass value, so the "just inside" offset is precisely sized
    relative to the real buffer, not merely at the raw boundary."""
    df = base_week_frame()
    high, low = _build_asian_range(df)
    ts = london_dt(2024, 6, 5, 8, 0)

    # Placeholder move at the target bar to establish a realistic, non-zero
    # same-bar ATR reading (mirrors how the strategy itself computes ATR).
    set_bar(df, ts, Close=high + 0.0050, High=high + 0.0050)
    frame = prepare_instrument_frame(df)
    ts_key = set_bar(df, ts, Close=high + 0.0050, High=high + 0.0050)  # resolve to the exact index key
    atr_now = frame.loc[ts_key, "atr"]
    assert atr_now > 0  # the placeholder move must have produced a real ATR reading

    tiny_clear = 0.01 * atr_now  # far short of the 0.1xATR buffer, but still nonzero
    set_bar(df, ts, Close=high + tiny_clear, High=high + tiny_clear)
    frame = prepare_instrument_frame(df)

    assert not frame.loc[ts_key, "signal_long"]
    assert not frame.loc[ts_key, "signal_short"]
    assert not frame.loc[ts, "signal_short"]


def test_close_still_inside_asian_range_does_not_fire():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=1.1000, High=1.1005, Low=1.0995)
    frame = prepare_instrument_frame(df)

    assert not frame.loc[ts, "signal_long"]
    assert not frame.loc[ts, "signal_short"]


def test_intrabar_touch_beyond_range_without_close_does_not_fire():
    """Continuation entry requires a CONFIRMED Close beyond the range,
    not just an intrabar High/Low touch - same close-based confirmation
    discipline as V1 (which required Close back inside; here it's Close
    outside)."""
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]
    buffer = 0.1 * atr_now

    # High wicks well beyond the buffer, but Close falls back inside the range.
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=high + buffer + 0.0020, Close=1.1000)
    frame = prepare_instrument_frame(df)

    assert not frame.loc[ts, "signal_long"]
    assert not frame.loc[ts, "signal_short"]


# =============================================================================
# 3. One trade per instrument per day
# =============================================================================

def test_only_first_breakout_of_the_day_fires():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]
    buffer = 0.1 * atr_now

    first_ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=high + buffer + 0.0010, High=high + buffer + 0.0010)
    # A second, later breakout (even in the opposite direction) the same day must NOT fire.
    second_ts = set_bar(df, london_dt(2024, 6, 5, 9, 0), Close=low - buffer - 0.0010, Low=low - buffer - 0.0010)
    frame = prepare_instrument_frame(df)

    assert frame.loc[first_ts, "signal_long"]
    assert not frame.loc[second_ts, "signal_long"]
    assert not frame.loc[second_ts, "signal_short"]


def test_different_days_each_get_their_own_independent_signal():
    df = make_m15_frame(WEEK_START - pd.Timedelta(days=3), WEEK_END)
    high1, low1 = _build_asian_range(df, high=1.1030, low=1.0970)  # Wed
    set_bar(df, london_dt(2024, 6, 6, 3, 0), High=1.1040, Low=1.0960)  # Thu, different range

    atr_now = prepare_instrument_frame(df)["atr"].iloc[-1]
    buffer = 0.1 * atr_now
    ts_wed = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=1.1030 + buffer + 0.0010, High=1.1030 + buffer + 0.0010)
    ts_thu = set_bar(df, london_dt(2024, 6, 6, 8, 0), Close=1.1040 + buffer + 0.0010, High=1.1040 + buffer + 0.0010)
    frame = prepare_instrument_frame(df)

    assert frame.loc[ts_wed, "signal_long"]
    assert frame.loc[ts_thu, "signal_long"]


# =============================================================================
# 4. Structural stop/target (opposite-boundary stop, fixed R:R target)
# =============================================================================

def test_long_stop_distance_uses_opposite_asian_low_plus_buffer():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]
    buffer = 0.1 * atr_now

    close_price = high + buffer + 0.0010
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=close_price, High=close_price)
    frame = prepare_instrument_frame(df)
    atr_at_signal = frame.loc[ts, "atr"]

    expected_stop_distance = (close_price - low) + 0.1 * atr_at_signal
    assert frame.loc[ts, "stop_distance_override"] == pytest.approx(expected_stop_distance)
    assert frame.loc[ts, "target_distance_override"] == pytest.approx(expected_stop_distance)  # 1:1 R:R


def test_short_stop_distance_uses_opposite_asian_high_plus_buffer():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    frame = prepare_instrument_frame(df)
    atr_now = frame["atr"].iloc[-1]
    buffer = 0.1 * atr_now

    close_price = low - buffer - 0.0010
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=close_price, Low=close_price)
    frame = prepare_instrument_frame(df)
    atr_at_signal = frame.loc[ts, "atr"]

    expected_stop_distance = (high - close_price) + 0.1 * atr_at_signal
    assert frame.loc[ts, "stop_distance_override"] == pytest.approx(expected_stop_distance)
    assert frame.loc[ts, "target_distance_override"] == pytest.approx(expected_stop_distance)


def test_target_scales_with_configured_rr_multiple():
    df = base_week_frame()
    high, low = _build_asian_range(df)
    config = LondonBreakoutContinuationConfig(target_rr_multiple=2.0)
    frame = prepare_instrument_frame(df, config=config)
    atr_now = frame["atr"].iloc[-1]
    buffer = 0.1 * atr_now

    close_price = high + buffer + 0.0010
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Close=close_price, High=close_price)
    frame = prepare_instrument_frame(df, config=config)

    stop_dist = frame.loc[ts, "stop_distance_override"]
    target_dist = frame.loc[ts, "target_distance_override"]
    assert target_dist == pytest.approx(stop_dist * 2.0)


def test_stop_and_target_are_nan_on_non_signal_rows():
    df = base_week_frame()
    frame = prepare_instrument_frame(df)
    no_signal_rows = frame[~(frame["signal_long"] | frame["signal_short"])]
    assert no_signal_rows["stop_distance_override"].isna().all()
    assert no_signal_rows["target_distance_override"].isna().all()


# =============================================================================
# 5. Prevention of validation/final-reserved access
# =============================================================================

def test_run_london_breakout_continuation_backtest_only_ever_uses_split_for_iteration():
    """The entry-point script must import/call split_for_iteration (which
    cannot return the reserved period - see test_dataset_split.py) and
    must never import get_final_reserved_period at all."""
    import inspect
    import run_london_breakout_continuation_backtest as entry_point

    source = inspect.getsource(entry_point)
    assert "split_for_iteration" in source
    assert "get_final_reserved_period" not in source


def test_split_for_iteration_applied_to_a_full_range_frame_never_reaches_reserved_period():
    """Behavioral version of the same guarantee: feed a frame spanning
    all three windows through split_for_iteration (exactly as the entry
    point does) and confirm nothing returned reaches VALIDATION_END."""
    full_range = make_m15_frame(
        dataset_split.DEVELOPMENT_START - pd.Timedelta(days=1),
        dataset_split.FINAL_RESERVED_END + pd.Timedelta(days=1),
    )
    development, validation = dataset_split.split_for_iteration({"EUR_USD": full_range})

    assert development["EUR_USD"].index.max() < dataset_split.DEVELOPMENT_END
    assert validation["EUR_USD"].index.max() < dataset_split.VALIDATION_END
