"""
test_london_sweep_signals.py
---------------------------------
Tests for signals_london_sweep_m15.py and run_london_sweep_backtest.py's
dataset-access discipline. Covers exactly the categories requested
before running any real backtest: BST/GMT transitions, Asian-range
boundaries, no lookahead, sweep detection, close-back-inside
confirmation, one-trade-per-day enforcement, stop/target calculations,
and prevention of validation/final-reserved access.

All synthetic, hand-constructed data - no real OANDA data or cached
files are touched by this file.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from signals_london_sweep_m15 import prepare_instrument_frame, LondonSweepConfig, LONDON_TZ
import dataset_split

LONDON = ZoneInfo("Europe/London")
SPREAD = 0.0002  # a flat, small, realistic-looking bid/ask spread for synthetic bars


def make_m15_frame(start_utc, end_utc, base_price=1.1000):
    """A flat M15 OHLC(+Bid/Ask) frame over [start_utc, end_utc) - every
    bar identical (Open=High=Low=Close=base_price) until the test
    overrides specific bars. UTC-indexed, matching real OANDA data."""
    index = pd.date_range(start=start_utc, end=end_utc, freq="15min", tz="UTC", inclusive="left")
    df = pd.DataFrame({
        "Open": base_price, "High": base_price, "Low": base_price, "Close": base_price,
        "Bid_Open": base_price - SPREAD / 2, "Bid_High": base_price - SPREAD / 2,
        "Bid_Low": base_price - SPREAD / 2, "Bid_Close": base_price - SPREAD / 2,
        "Ask_Open": base_price + SPREAD / 2, "Ask_High": base_price + SPREAD / 2,
        "Ask_Low": base_price + SPREAD / 2, "Ask_Close": base_price + SPREAD / 2,
        "Volume": 100.0,
    }, index=index)
    return df


def set_bar(df, local_dt, **kwargs):
    """Set OHLC values on the bar at the given Europe/London LOCAL
    datetime (a tz-aware datetime in the LONDON zone) - converts to the
    frame's UTC index internally, so tests can be written in the same
    terms as the strategy's own session logic."""
    ts_utc = local_dt.astimezone(dt.timezone.utc)
    ts_utc = df.index[df.index.get_indexer([ts_utc], method="nearest")[0]]
    for col, value in kwargs.items():
        df.loc[ts_utc, col] = value
        if col in ("High", "Low", "Close", "Open"):
            # keep Bid/Ask consistent with the overridden mid price
            df.loc[ts_utc, f"Bid_{col}"] = value - SPREAD / 2
            df.loc[ts_utc, f"Ask_{col}"] = value + SPREAD / 2
    return ts_utc


def london_dt(year, month, day, hour, minute=0):
    return dt.datetime(year, month, day, hour, minute, tzinfo=LONDON)


# A plain week with no DST transition nearby, well clear of the start of
# the frame so ATR (period 14) is always warmed up by the day under test.
WEEK_START = london_dt(2024, 6, 3, 0).astimezone(dt.timezone.utc)   # Monday
WEEK_END = london_dt(2024, 6, 8, 0).astimezone(dt.timezone.utc)     # following Saturday


def base_week_frame():
    return make_m15_frame(WEEK_START - pd.Timedelta(days=3), WEEK_END)  # extra days for ATR warmup


# =============================================================================
# 1. BST/GMT transitions
# =============================================================================

def test_bst_session_boundary_differs_from_naive_utc_boundary():
    """On a BST (summer time) day, 'Europe/London 07:00' is 06:00 UTC,
    NOT 07:00 UTC - a naive fixed-UTC-hour implementation would draw the
    London-window boundary in the wrong place. Directly demonstrate the
    tz-aware boundary lands where local time says it should, one hour
    off from what a naive UTC-hour check would have used."""
    # 2024-06-05 is deep in BST (clocks forward happened 2024-03-31).
    df = base_week_frame()
    config = LondonSweepConfig()

    frame = prepare_instrument_frame(df, config=config)

    london_local = frame.index.tz_convert(LONDON)
    is_london_window = (london_local.hour >= config.london_window_start_hour) & \
                        (london_local.hour < config.london_window_end_hour)

    # The bar at 07:00 BST local time is 06:00 UTC - it MUST be in the
    # London window (by local time), even though its UTC hour (6) is
    # outside what a naive [7,10) UTC check would have accepted.
    bar_0700_bst = pd.Timestamp("2024-06-05 06:00:00", tz="UTC")  # == 07:00 BST
    assert is_london_window[frame.index == bar_0700_bst].all()

    # And the bar at 07:00 UTC that same day is 08:00 BST local - still
    # inside the window by local time (window is [7,10) local), which a
    # naive UTC-based [7,10) check would ALSO have accepted, so pick a
    # bar that only a correct local-time check gets right: 10:00 UTC ==
    # 11:00 BST, outside the local window, while a naive UTC check
    # (if window were mistakenly [7,10) UTC) would have excluded 10:00
    # UTC too - so instead check 09:45 UTC == 10:45 BST, which a naive
    # UTC implementation (window [7,10) UTC) would WRONGLY include, but
    # correct local-time logic correctly excludes (10:45 local >= 10).
    bar_naive_would_wrongly_include = pd.Timestamp("2024-06-05 09:45:00", tz="UTC")
    assert not is_london_window[frame.index == bar_naive_would_wrongly_include].all()


def test_spring_forward_transition_2024():
    """2024-03-31 is the UK's spring-forward date (clocks go 00:59 GMT
    -> 02:00 BST, so 01:00-01:59 GMT never happens that day). Asian
    session/London window classification must still work across it
    without crashing or misclassifying, using ONLY real UTC timestamps
    (which are always well-defined, even on a transition day)."""
    start = pd.Timestamp("2024-03-30 00:00", tz="UTC") - pd.Timedelta(days=3)
    end = pd.Timestamp("2024-04-01 00:00", tz="UTC")
    df = make_m15_frame(start, end)
    config = LondonSweepConfig()

    frame = prepare_instrument_frame(df, config=config)  # must not raise

    london_local = frame.index.tz_convert(LONDON)
    transition_day = london_local.date == dt.date(2024, 3, 31)
    # The transition day still has classifiable bars in both the Asian
    # and London windows - the DST jump doesn't erase the day's data.
    assert transition_day.sum() > 0


def test_fall_back_transition_2024():
    """2024-10-27 is the UK's fall-back date (01:00-01:59 BST happens,
    then clocks go back to 01:00 GMT, so that local hour occurs twice
    that day, via two DISTINCT UTC timestamps). Both occurrences must be
    handled - each is a genuine, distinct M15 bar - without a crash or a
    silently dropped/duplicated bar."""
    start = pd.Timestamp("2024-10-26 00:00", tz="UTC") - pd.Timedelta(days=3)
    end = pd.Timestamp("2024-10-28 00:00", tz="UTC")
    df = make_m15_frame(start, end)
    config = LondonSweepConfig()

    frame = prepare_instrument_frame(df, config=config)  # must not raise
    assert len(frame) == len(df)  # every UTC bar survives - none dropped or merged


# =============================================================================
# 2. Asian-range boundaries
# =============================================================================

def test_asian_range_uses_only_bars_strictly_before_0700_local():
    df = base_week_frame()
    day = dt.date(2024, 6, 5)

    # An extreme high placed at exactly 07:00 local (first LONDON-window
    # bar) must NOT be counted in the Asian range - the boundary is
    # exclusive on the Asian side.
    set_bar(df, london_dt(2024, 6, 5, 7, 0), High=9.9999)
    # The genuine Asian-session extreme, one bar earlier (06:45 local).
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1050)

    frame = prepare_instrument_frame(df)
    london_local = frame.index.tz_convert(LONDON)
    row_0700 = frame[(london_local.date == day) & (london_local.hour == 7) & (london_local.minute == 0)]

    assert row_0700["asian_high"].iloc[0] == pytest.approx(1.1050)
    assert row_0700["asian_high"].iloc[0] != pytest.approx(9.9999)


def test_asian_range_excludes_bars_from_a_different_day():
    df = base_week_frame()
    # An extreme placed on the PREVIOUS day's Asian session must not
    # leak into today's range.
    set_bar(df, london_dt(2024, 6, 4, 3, 0), High=9.9999)
    set_bar(df, london_dt(2024, 6, 5, 3, 0), High=1.1030)

    frame = prepare_instrument_frame(df)
    london_local = frame.index.tz_convert(LONDON)
    row = frame[(london_local.date == dt.date(2024, 6, 5)) & (london_local.hour == 8)]

    assert row["asian_high"].iloc[0] == pytest.approx(1.1030)


# =============================================================================
# 3. No lookahead
# =============================================================================

def test_bar_before_asian_close_never_sees_that_days_range():
    """A bar at 03:00 local (mid-Asian-session) must have NaN
    asian_high/asian_low, even though the FULL day's Asian range is
    technically already computable in memory by the time this function
    runs - the whole point is that a bar can never see a same-day value
    before it would genuinely have been knowable."""
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1099)  # the eventual Asian high

    frame = prepare_instrument_frame(df)
    london_local = frame.index.tz_convert(LONDON)
    row_0300 = frame[(london_local.date == dt.date(2024, 6, 5)) & (london_local.hour == 3) & (london_local.minute == 0)]

    assert pd.isna(row_0300["asian_high"].iloc[0])
    assert pd.isna(row_0300["asian_low"].iloc[0])


def test_asian_range_is_frozen_after_0700_even_if_a_later_bar_is_more_extreme():
    """A new, more extreme High occurring DURING the London window must
    NOT retroactively change that day's already-finalized Asian range -
    the range describes the completed Asian session only, not a running
    high/low of the whole day."""
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1040)   # genuine Asian high
    set_bar(df, london_dt(2024, 6, 5, 9, 0), High=1.2000)     # a later, bigger high - during London window

    frame = prepare_instrument_frame(df)
    london_local = frame.index.tz_convert(LONDON)
    row_0915 = frame[(london_local.date == dt.date(2024, 6, 5)) & (london_local.hour == 9) & (london_local.minute == 15)]

    assert row_0915["asian_high"].iloc[0] == pytest.approx(1.1040)


# =============================================================================
# 4/5. Sweep detection + close-back-inside confirmation
# =============================================================================

def test_sweep_without_close_back_inside_produces_no_signal():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)  # Asian range
    # Sweeps above and STAYS above asian_high for the rest of the London
    # window that day (the background/default bars are 1.1000, BELOW
    # asian_high, so every remaining bar needs an explicit override too -
    # otherwise the very next flat bar would itself close back inside and
    # confirm, which is not what this test is checking).
    set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1050)
    for minute_offset in range(15, 120, 15):  # 08:15 through 09:45
        t = london_dt(2024, 6, 5, 8, 0) + dt.timedelta(minutes=minute_offset)
        set_bar(df, t, High=1.1055, Low=1.1045, Close=1.1050)

    frame = prepare_instrument_frame(df)
    assert not frame["signal_long"].any()
    assert not frame["signal_short"].any()


def test_sweep_and_close_back_inside_on_the_same_candle_fires_immediately():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)  # sweeps AND closes back inside

    frame = prepare_instrument_frame(df)
    assert frame.loc[ts, "signal_short"]
    assert not frame.loc[ts, "signal_long"]
    assert frame.loc[ts, "sweep_extreme_price"] == pytest.approx(1.1060)


def test_sweep_then_later_candle_confirms():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    sweep_ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1050)   # sweeps, doesn't confirm
    confirm_ts = set_bar(df, london_dt(2024, 6, 5, 8, 15), High=1.1040, Close=1.1010)  # confirms

    frame = prepare_instrument_frame(df)
    assert not frame.loc[sweep_ts, "signal_short"]
    assert frame.loc[confirm_ts, "signal_short"]
    # penetration uses the deepest sweep extreme (1.1060), not the confirmation candle's own high.
    assert frame.loc[confirm_ts, "sweep_extreme_price"] == pytest.approx(1.1060)


def test_bullish_mirror_sweep_and_confirmation():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), Low=1.0940, Close=1.0980)  # sweeps low, closes back inside

    frame = prepare_instrument_frame(df)
    assert frame.loc[ts, "signal_long"]
    assert not frame.loc[ts, "signal_short"]
    assert frame.loc[ts, "sweep_extreme_price"] == pytest.approx(1.0940)


def test_sweep_extreme_updates_to_deepest_penetration_while_pending():
    """If price sweeps, pulls back (not enough to confirm), sweeps EVEN
    further, then finally confirms, the recorded extreme/penetration
    must reflect the deepest point reached, not the first touch."""
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1040, Close=1.1035)   # first, shallower sweep
    set_bar(df, london_dt(2024, 6, 5, 8, 15), High=1.1080, Close=1.1045)  # deeper sweep, still no confirm
    confirm_ts = set_bar(df, london_dt(2024, 6, 5, 8, 30), High=1.1050, Close=1.1010)  # confirms

    frame = prepare_instrument_frame(df)
    assert frame.loc[confirm_ts, "signal_short"]
    assert frame.loc[confirm_ts, "sweep_extreme_price"] == pytest.approx(1.1080)


# =============================================================================
# 6. One trade per instrument per day
# =============================================================================

def test_only_one_signal_fires_per_day_even_with_two_valid_setups():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    first_ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)     # valid bearish setup
    second_ts = set_bar(df, london_dt(2024, 6, 5, 9, 0), Low=1.0900, Close=1.0960)      # ALSO a valid bullish setup, later

    frame = prepare_instrument_frame(df)
    assert frame.loc[first_ts, "signal_short"]
    assert not frame.loc[second_ts, "signal_long"]
    assert not frame.loc[second_ts, "signal_short"]

    day = dt.date(2024, 6, 5)
    london_local = frame.index.tz_convert(LONDON)
    day_mask = london_local.date == day
    assert (frame.loc[day_mask, "signal_long"] | frame.loc[day_mask, "signal_short"]).sum() == 1


def test_different_days_each_get_their_own_independent_signal():
    df = base_week_frame()
    for day in (4, 5, 6):
        set_bar(df, london_dt(2024, 6, day, 6, 45), High=1.1030, Low=1.0970)
        set_bar(df, london_dt(2024, 6, day, 8, 0), High=1.1060, Close=1.1020)

    frame = prepare_instrument_frame(df)
    assert frame["signal_short"].sum() == 3


# =============================================================================
# 7. Stop/target calculations
# =============================================================================

def test_stop_and_target_distance_match_structural_formula():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)

    config = LondonSweepConfig(stop_buffer_atr_fraction=0.1, target_rr_multiple=1.0)
    frame = prepare_instrument_frame(df, config=config)

    atr_at_signal = frame.loc[ts, "atr"]
    expected_stop = abs(1.1060 - 1.1020) + 0.1 * atr_at_signal  # |sweep_extreme - Close| + buffer
    assert frame.loc[ts, "stop_distance_override"] == pytest.approx(expected_stop)
    assert frame.loc[ts, "target_distance_override"] == pytest.approx(expected_stop)  # 1:1 R:R


def test_target_scales_with_configured_rr_multiple():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)

    config = LondonSweepConfig(target_rr_multiple=2.0)
    frame = prepare_instrument_frame(df, config=config)

    assert frame.loc[ts, "target_distance_override"] == pytest.approx(frame.loc[ts, "stop_distance_override"] * 2.0)


def test_stop_and_target_are_nan_on_non_signal_rows():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1060, Close=1.1020)

    frame = prepare_instrument_frame(df)
    non_signal_rows = frame[~(frame["signal_long"] | frame["signal_short"])]
    assert non_signal_rows["stop_distance_override"].isna().all()
    assert non_signal_rows["target_distance_override"].isna().all()


def test_sweep_penetration_recorded_in_price_and_atr_terms_but_not_used_as_a_filter():
    df = base_week_frame()
    set_bar(df, london_dt(2024, 6, 5, 6, 45), High=1.1030, Low=1.0970)
    ts = set_bar(df, london_dt(2024, 6, 5, 8, 0), High=1.1032, Close=1.1020)  # a TINY, 0.0002 sweep

    frame = prepare_instrument_frame(df)
    # A signal fires regardless of how small the sweep was - round 1 uses no penetration filter.
    assert frame.loc[ts, "signal_short"]
    assert frame.loc[ts, "sweep_penetration_price"] == pytest.approx(1.1032 - 1.1030)
    assert frame.loc[ts, "sweep_penetration_atr"] == pytest.approx(
        frame.loc[ts, "sweep_penetration_price"] / frame.loc[ts, "atr"]
    )


# =============================================================================
# 8. Prevention of validation/final-reserved access
# =============================================================================

def test_run_london_sweep_backtest_only_ever_uses_split_for_iteration():
    """The entry-point script must import/call split_for_iteration (which
    cannot return the reserved period - see test_dataset_split.py) and
    must never import get_final_reserved_period at all."""
    import inspect
    import run_london_sweep_backtest as entry_point

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
