"""
test_london_sweep_trend_aligned_signals.py
------------------------------------------------
Tests for signals_london_sweep_trend_aligned_m15.py (V2). V1's own
sweep+confirmation logic is already covered by
test_london_sweep_signals.py and is unchanged here (imported, not
copied) - these tests focus on what's NEW in V2: the daily trend
computation's no-lookahead behavior, and the trend-alignment gate
itself (kept when aligned, dropped when not, overrides cleaned up on
drop).

All synthetic, hand-constructed data - no real OANDA data or cached
files are touched by this file.
"""

import datetime as dt

import pandas as pd
import pytest

from signals_london_sweep_m15 import LondonSweepConfig
from signals_london_sweep_trend_aligned_m15 import prepare_instrument_frame, TrendAlignedLondonSweepConfig
from tests.test_london_sweep_signals import make_m15_frame, set_bar, london_dt, LONDON, WEEK_START, WEEK_END


def make_daily_frame(start_utc, end_utc, trend="up", base_price=1.1000, daily_step=0.0005):
    """A daily OHLC frame (OANDA convention: indexed by OPEN time, which
    is 21:00 UTC) with Close ramping steadily up or down - enough of a
    directional ramp that EMA50/200 cleanly agree on a single trend
    direction well before the end of the range, so tests can rely on a
    known, unambiguous trend classification."""
    index = pd.date_range(start=start_utc, end=end_utc, freq="1D", tz="UTC")
    n = len(index)
    step = daily_step if trend == "up" else -daily_step
    closes = [base_price + i * step for i in range(n)]
    df = pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
        "Volume": 100.0,
    }, index=index)
    return df


def base_week_daily_frame(trend="up"):
    # A long ramp well before WEEK_START so EMA200 has fully converged by
    # the time the M15 test window (WEEK_START..WEEK_END) is evaluated.
    return make_daily_frame(WEEK_START - pd.Timedelta(days=400), WEEK_END + pd.Timedelta(days=1), trend=trend)


def _build_v1_style_bearish_setup(df):
    """The exact EUR_USD-style bearish sweep+confirmation setup already
    proven in test_london_sweep_signals.py - reused here so these tests
    are only exercising the NEW trend-alignment layer, not re-deriving
    sweep detection from scratch."""
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    return set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)  # bearish (short) setup


def _build_v1_style_bullish_setup(df):
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    return set_bar(df, london_dt(2024, 6, 5, 8, 0), Low=1.0940, Close=1.0980)  # bullish (long) setup


# =============================================================================
# Trend-alignment gate
# =============================================================================

def test_short_setup_kept_when_daily_trend_is_down():
    m15 = base_week_frame_import_helper()
    ts = _build_v1_style_bearish_setup(m15)
    daily = base_week_daily_frame(trend="down")

    frame = prepare_instrument_frame(m15, daily)
    assert frame.loc[ts, "signal_short"]
    assert not pd.isna(frame.loc[ts, "stop_distance_override"])


def test_short_setup_dropped_when_daily_trend_is_up():
    m15 = base_week_frame_import_helper()
    ts = _build_v1_style_bearish_setup(m15)
    daily = base_week_daily_frame(trend="up")  # misaligned - short setup, up trend

    frame = prepare_instrument_frame(m15, daily)
    assert not frame.loc[ts, "signal_short"]
    assert not frame.loc[ts, "signal_long"]
    assert pd.isna(frame.loc[ts, "stop_distance_override"])
    assert pd.isna(frame.loc[ts, "target_distance_override"])


def test_long_setup_kept_when_daily_trend_is_up():
    m15 = base_week_frame_import_helper()
    ts = _build_v1_style_bullish_setup(m15)
    daily = base_week_daily_frame(trend="up")

    frame = prepare_instrument_frame(m15, daily)
    assert frame.loc[ts, "signal_long"]
    assert not pd.isna(frame.loc[ts, "stop_distance_override"])


def test_long_setup_dropped_when_daily_trend_is_down():
    m15 = base_week_frame_import_helper()
    ts = _build_v1_style_bullish_setup(m15)
    daily = base_week_daily_frame(trend="down")  # misaligned - long setup, down trend

    frame = prepare_instrument_frame(m15, daily)
    assert not frame.loc[ts, "signal_long"]
    assert not frame.loc[ts, "signal_short"]
    assert pd.isna(frame.loc[ts, "stop_distance_override"])


def test_v1_sweep_detection_itself_is_unchanged():
    """V2 must not alter WHERE V1 would have signaled - only whether that
    signal survives the trend gate. Confirm the underlying sweep
    detection (before gating) matches what test_london_sweep_signals.py
    already established for this exact setup."""
    from signals_london_sweep_m15 import prepare_instrument_frame as v1_prepare
    m15 = base_week_frame_import_helper()
    ts = _build_v1_style_bearish_setup(m15)

    v1_frame = v1_prepare(m15)
    assert v1_frame.loc[ts, "signal_short"]  # V1 alone (no trend gate) fires here, as already tested


# =============================================================================
# No lookahead in the daily trend merge
# =============================================================================

def test_daily_trend_not_visible_before_that_candles_own_close():
    """A daily candle indexed at OPEN time (21:00 UTC, day D) isn't
    knowable until it CLOSES (21:00 UTC, day D+1) - an M15 bar just
    before that close must still see the PRIOR daily trend reading, not
    the new one, even though the new candle's OWN close price is already
    fixed in the synthetic data by the time this function runs."""
    # A trend flip: down for the first stretch, sharply up starting at a
    # specific day - the flip day's own OPEN is 21:00 UTC on the flip
    # date, so its trend contribution isn't knowable until 21:00 UTC the
    # NEXT day.
    flip_date = pd.Timestamp("2024-06-01 21:00", tz="UTC")
    before = make_daily_frame(flip_date - pd.Timedelta(days=400), flip_date - pd.Timedelta(days=1), trend="down")
    after = make_daily_frame(flip_date, flip_date + pd.Timedelta(days=10), trend="up",
                              base_price=before["Close"].iloc[-1])
    daily = pd.concat([before, after])

    m15 = make_m15_frame(flip_date - pd.Timedelta(hours=1), flip_date + pd.Timedelta(hours=25))

    config = TrendAlignedLondonSweepConfig()
    from signals_london_sweep_trend_aligned_m15 import _prepare_daily_trend
    trend = _prepare_daily_trend(daily, config)

    # Just before the flip day's candle closes (21:00 UTC the next day) -
    # must still reflect the OLD (down) trend from the previous candle.
    just_before_close = pd.Timestamp("2024-06-02 20:45", tz="UTC")
    row = trend.asof(just_before_close)
    assert bool(row["daily_trend_down"]) or not bool(row["daily_trend_up"])

    # At/after that close - the new candle's contribution becomes visible.
    at_close = pd.Timestamp("2024-06-02 21:00", tz="UTC")
    row_after = trend.asof(at_close)
    assert row_after.name == at_close  # the shifted index lands exactly here, not one bar early or late


def test_daily_trend_index_shifts_by_exactly_24_hours():
    daily = base_week_daily_frame(trend="up")
    config = TrendAlignedLondonSweepConfig()
    from signals_london_sweep_trend_aligned_m15 import _prepare_daily_trend
    trend = _prepare_daily_trend(daily, config)

    original_first_ts = daily.index[0]
    assert trend.index[0] == original_first_ts + pd.Timedelta(hours=24)


# --- small helper to avoid re-deriving the base M15 frame construction ---
def base_week_frame_import_helper():
    from tests.test_london_sweep_signals import base_week_frame
    return base_week_frame()
