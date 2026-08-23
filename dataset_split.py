"""
dataset_split.py
--------------------
Three-way chronological data split for NEW strategy development,
starting with the London Liquidity Sweep Reversal strategy. Kept
completely separate from run_backtest.py's own two-way TRAIN/TEST split
(`split_train_test`), which is untouched and still applies to every
strategy already tried (trend-following, mean-reversion, momentum,
confirmation filters - see README.md's "Systematic strategy search:
concluded"). This is a stricter discipline for the next round of work,
not a replacement of the old one.

Three chronological, non-overlapping, NEVER-shuffled windows:
  DEVELOPMENT     - everything: building, tuning, sweeping, the normal
                     train/test-split-then-stress-test iteration.
  VALIDATION      - checked only once a version is frozen. Seeing a bad
                     result here means going back to development and
                     iterating further - that's fine. Seeing a bad
                     result on the FINAL RESERVED period below is not
                     something to iterate away.
  FINAL RESERVED  - locked. Not inspected, not even descriptively,
                     until development AND validation are both done and
                     every parameter is frozen. Checked once. Whatever
                     it shows is the honest answer, not a suggestion to
                     tune further - see get_final_reserved_period()'s
                     docstring.

Dates were chosen from ACTUALLY CONFIRMED M15 data availability for
EUR_USD/GBP_USD (checked directly against the OANDA API, not assumed) -
history begins 2020-08-24 for M15/M5, eleven days later than the
H1/H4/D series' 2020-08-13 start. A sample-sufficiency check (a fixed,
non-tuned, generic Asian-range-sweep pattern, counted only on dates
before the reserved period) found ample raw candidate density in both
the development window (~750-765 distinct days/instrument with a
qualifying event) and the validation window (~175-190) - see the
project conversation history for the full methodology; that check
itself never touched the reserved period either.

    DEVELOPMENT:     2020-08-24  -> 2024-07-15   (~1,421 days, ~65%)
    VALIDATION:       2024-07-15  -> 2025-07-15   (~365 days, ~17%)
    FINAL RESERVED:  2025-07-15  -> 2026-08-21   (~402 days, ~18%)

FINAL_RESERVED_END is a FIXED date (the confirmed extent of available
data when this split was proposed and approved), not "today" computed
dynamically - a floating end date would mean the locked benchmark keeps
changing every time someone runs something, defeating the point of a
fixed final check.
"""

import pandas as pd

DEVELOPMENT_START = pd.Timestamp("2020-08-24", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2024-07-15", tz="UTC")      # exclusive upper bound
VALIDATION_END = pd.Timestamp("2025-07-15", tz="UTC")       # exclusive upper bound
FINAL_RESERVED_END = pd.Timestamp("2026-08-21", tz="UTC")   # fixed - confirmed available data as of the split proposal


def split_for_iteration(frames):
    """
    THE function every strategy-development script should import and
    use for day-to-day work. Returns (development_frames,
    validation_frames) - two {symbol: DataFrame} dicts, chronologically
    sliced, in that order.

    The FINAL RESERVED period is not reachable through this function AT
    ALL - by construction, not just convention, so accidentally
    including it requires a deliberate, separate, loud call (see
    get_final_reserved_period below), not a slip of a date argument
    somewhere in a sweep script.

    `frames`: {symbol: DataFrame} with a datetime index - either raw
    OANDA candle data or an already-prepared signal frame (anything
    prepare_instrument_frame()-shaped from any strategy module in this
    project); this function doesn't care which, it only slices on the
    index.
    """
    development, validation = {}, {}
    for symbol, df in frames.items():
        df = df.sort_index()
        development[symbol] = df[(df.index >= DEVELOPMENT_START) & (df.index < DEVELOPMENT_END)]
        validation[symbol] = df[(df.index >= DEVELOPMENT_END) & (df.index < VALIDATION_END)]
    return development, validation


def get_final_reserved_period(frames, i_am_freezing_the_strategy_and_running_the_final_check=False):
    """
    The FINAL RESERVED period (VALIDATION_END -> FINAL_RESERVED_END) -
    deliberately not importable via split_for_iteration() above.
    Requires an explicit, spelled-out confirmation flag, and prints a
    loud, unmissable banner every time it's actually used, so a run
    against the reserved period always leaves an obvious trace in
    whatever output/log captures it - never a silent, easy-to-miss call
    buried in a sweep loop.

    Call this ONCE, after every parameter is frozen following
    development and validation, and never tune anything further
    afterward based on what it shows - the whole point of reserving it
    is to get one honest, final read, not another round of feedback to
    optimize against.
    """
    if not i_am_freezing_the_strategy_and_running_the_final_check:
        raise RuntimeError(
            "Refusing to return the FINAL RESERVED period - pass "
            "i_am_freezing_the_strategy_and_running_the_final_check=True "
            "if you are DELIBERATELY running the one-time final check on a "
            "fully frozen strategy, not tuning anything further afterward."
        )

    print("=" * 78)
    print("*** ACCESSING THE FINAL RESERVED (LOCKED) OUT-OF-SAMPLE PERIOD ***")
    print(f"*** {VALIDATION_END.date()} -> {FINAL_RESERVED_END.date()} ***")
    print("*** This should happen ONCE per strategy, with everything already")
    print("*** frozen beforehand. Do not tune anything based on what this shows.")
    print("=" * 78)

    reserved = {}
    for symbol, df in frames.items():
        df = df.sort_index()
        reserved[symbol] = df[(df.index >= VALIDATION_END) & (df.index < FINAL_RESERVED_END)]
    return reserved


def describe_split():
    """Human-readable summary of the three windows, for a script to
    print at startup so every run states plainly which data it's
    allowed to be looking at."""
    return (
        f"DEVELOPMENT:    {DEVELOPMENT_START.date()} -> {DEVELOPMENT_END.date()}\n"
        f"VALIDATION:     {DEVELOPMENT_END.date()} -> {VALIDATION_END.date()}\n"
        f"FINAL RESERVED: {VALIDATION_END.date()} -> {FINAL_RESERVED_END.date()}  (LOCKED - do not inspect yet)"
    )
