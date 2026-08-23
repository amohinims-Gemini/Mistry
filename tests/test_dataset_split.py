"""
test_dataset_split.py
-------------------------
Locks in the guard rails dataset_split.py exists to provide - the whole
point of that module is to make leaking the final reserved period hard
to do by accident, so that property itself needs a test, not just a
docstring promising it.
"""

import pandas as pd
import pytest

from dataset_split import (
    DEVELOPMENT_START, DEVELOPMENT_END, VALIDATION_END, FINAL_RESERVED_END,
    split_for_iteration, get_final_reserved_period,
)


def _make_frame(start, end, freq="15min"):
    index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return pd.DataFrame({"Close": range(len(index))}, index=index)


@pytest.fixture
def full_range_frames():
    # Spans slightly before development start to slightly after the
    # final reserved end, so every boundary actually gets exercised.
    df = _make_frame(DEVELOPMENT_START - pd.Timedelta(days=1), FINAL_RESERVED_END + pd.Timedelta(days=1))
    return {"EUR_USD": df, "GBP_USD": df.copy()}


def test_windows_are_chronological_and_non_overlapping():
    assert DEVELOPMENT_START < DEVELOPMENT_END < VALIDATION_END < FINAL_RESERVED_END


def test_split_for_iteration_never_touches_or_returns_reserved_period(full_range_frames):
    development, validation = split_for_iteration(full_range_frames)

    for symbol in full_range_frames:
        assert development[symbol].index.max() < DEVELOPMENT_END
        assert development[symbol].index.min() >= DEVELOPMENT_START
        assert validation[symbol].index.max() < VALIDATION_END
        assert validation[symbol].index.min() >= DEVELOPMENT_END
        # Nothing from either returned frame reaches into the reserved window.
        assert development[symbol].index.max() < VALIDATION_END
        assert validation[symbol].index.max() < VALIDATION_END


def test_split_for_iteration_return_type_has_no_reserved_accessor(full_range_frames):
    # split_for_iteration returns exactly two things - there is no third
    # value or key anywhere in its return that could accidentally expose
    # the reserved period.
    result = split_for_iteration(full_range_frames)
    assert len(result) == 2


def test_get_final_reserved_period_refuses_without_explicit_flag(full_range_frames):
    with pytest.raises(RuntimeError):
        get_final_reserved_period(full_range_frames)

    with pytest.raises(RuntimeError):
        get_final_reserved_period(full_range_frames, i_am_freezing_the_strategy_and_running_the_final_check=False)


def test_get_final_reserved_period_returns_correct_slice_when_explicitly_requested(full_range_frames):
    reserved = get_final_reserved_period(
        full_range_frames, i_am_freezing_the_strategy_and_running_the_final_check=True
    )
    for symbol in full_range_frames:
        assert reserved[symbol].index.min() >= VALIDATION_END
        assert reserved[symbol].index.max() < FINAL_RESERVED_END
        assert len(reserved[symbol]) > 0  # the fixture's range actually covers this window


def test_development_and_validation_cover_the_intended_range_with_no_gap_or_overlap(full_range_frames):
    development, validation = split_for_iteration(full_range_frames)
    df = full_range_frames["EUR_USD"]

    # Every row strictly between DEVELOPMENT_START and VALIDATION_END
    # lands in exactly one of the two returned frames - no gap, no
    # double-count, at the DEVELOPMENT_END boundary.
    in_range = df[(df.index >= DEVELOPMENT_START) & (df.index < VALIDATION_END)]
    combined_count = len(development["EUR_USD"]) + len(validation["EUR_USD"])
    assert combined_count == len(in_range)
