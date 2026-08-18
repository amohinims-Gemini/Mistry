"""
econ_calendar.py
------------------
APPROXIMATION, NOT A REAL ECONOMIC CALENDAR.

This project has no access to a live or historical economic calendar
API. Rather than pretend to know exactly when FOMC meetings, NFP
releases, CPI prints, etc. happened over the last several years, this
uses a simple recurring heuristic: block new entries during the weekly
time window when high-impact US data most commonly drops.

WHAT THIS DOES catch: the recurring weekday ~8:30am ET slot, where many
NFP / CPI / retail sales / jobless claims releases land (including NFP
itself, which is always a Friday).

WHAT THIS WON'T catch:
  - FOMC rate decisions (2pm ET, ~8 times/year, on dates that move
    around from year to year and aren't hardcoded here)
  - ECB / BoE / BoJ rate decisions
  - One-off surprise announcements (e.g. surprise rate cuts, geopolitical
    shocks)
  - Non-recurring, date-specific events in general

This is a REAL limitation: it will let some trades through right before
genuine high-impact news, and will sometimes block trading around a
low-impact release for no good reason. It's a rough backstop, not a
substitute for a real calendar feed.

This logic is deliberately isolated behind one function
(`is_blackout_window`) so a real economic calendar (e.g. ForexFactory,
TradingEconomics, FRED release calendar) can be swapped in later by
rewriting just this file - nothing else in the project needs to change.
"""

# Each entry: (weekday, start_hour_utc, start_minute, end_hour_utc, end_minute)
# weekday: 0=Monday ... 6=Sunday
#
# US high-impact data commonly releases 8:30am ET. US Eastern time is
# UTC-5 in winter and UTC-4 in summer (daylight saving) - rather than
# track the DST calendar exactly, the window below is widened generously
# (12:15-13:15 UTC) to cover both without needing to know the exact date
# DST changed in any given year.
BLACKOUT_WINDOWS_UTC = [
    (0, 12, 0, 14, 0),  # Monday
    (1, 12, 0, 14, 0),  # Tuesday
    (2, 12, 0, 14, 0),  # Wednesday - doesn't cover the 2pm ET FOMC decision itself
    (3, 12, 0, 14, 0),  # Thursday
    (4, 12, 0, 14, 0),  # Friday - covers NFP (always a Friday, 8:30am ET)
]
# Widened to whole-hour boundaries (12:00-14:00 UTC, was 12:15-13:15) so
# this can actually catch candles on ANY timeframe used in this project -
# 4H candles land exactly on hours 0/4/8/12/16/20, so a window starting
# at :15 could never overlap one at all. A small, deliberate generosity
# increase on an already-approximate heuristic, not a 4H-specific hack -
# applies to every strategy that imports this module.


def is_blackout_window(timestamp_utc):
    """Return True if `timestamp_utc` (a UTC datetime/Timestamp) falls
    inside an approximate high-impact-news blackout window."""
    weekday = timestamp_utc.weekday()
    minute_of_day = timestamp_utc.hour * 60 + timestamp_utc.minute

    for wd, start_h, start_m, end_h, end_m in BLACKOUT_WINDOWS_UTC:
        if weekday != wd:
            continue
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if start <= minute_of_day <= end:
            return True
    return False
