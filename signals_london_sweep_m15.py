"""
signals_london_sweep_m15.py
--------------------------------
Round 1 of the London Liquidity Sweep Reversal strategy - a genuinely
different, structural idea from every entry-logic family in the
concluded systematic search (see README.md's "Systematic strategy
search: concluded"), not a variant of trend-following, mean-reversion,
or momentum. Single timeframe (M15), session-anchored, DST-aware.

Rule (short; long is the exact mirror):
  1. Build the Asian session's high/low (00:00-07:00 Europe/London
     LOCAL time - DST-aware, not a fixed UTC window) from COMPLETED
     bars only - see prepare_instrument_frame() for the exact
     no-lookahead mechanism.
  2. During the London entry window (07:00-10:00 Europe/London local
     time), watch for a SWEEP: a candle's High trades above the Asian
     high (bearish) or Low trades below the Asian low (bullish).
  3. The sweep itself is NOT an entry. Wait for the first subsequent
     candle (which may be the SAME candle) whose Close moves back
     INSIDE the Asian range - THAT candle is the signal.
  4. At most ONE trade per instrument per London-local day, whichever
     direction confirms first - once a signal fires, no further bars
     that day are evaluated at all.

Deliberately excludes RSI/MACD/EMA or any other indicator confirmation
- the close-back-inside-the-range structure IS the confirmation, not a
bolted-on filter. No minimum sweep-penetration threshold either -
round 1 takes every confirmed setup regardless of how small the sweep
was (see the penetration-recording note below).

Stop-loss is STRUCTURAL, not a flat ATR multiple like every prior
strategy in this project: placed beyond the sweep candle's own extreme
(the level whose violation would invalidate the reversal thesis), plus
a small ATR buffer (stop_buffer_atr_fraction) so the stop doesn't sit
exactly on the wick tip. Take-profit is a fixed 1:1 multiple of that
same distance for round 1. Both flow through risk_management.py's
UNCHANGED calculate_position_size() via stop_distance_override /
target_distance_override - the same mechanism combined_signals_4h.py
already established - so position sizing and every safety limit stay
exactly the infrastructure already validated elsewhere in this project.

Every confirmed setup also records its sweep penetration distance, in
raw price units AND ATR-normalized terms (sweep_penetration_price,
sweep_penetration_atr) - purely for LATER, SEPARATE analysis of whether
very small sweeps are distinguishable from spread/data noise. This is
explicitly NOT used as a filter or condition anywhere in this module -
round 1 takes every confirmed setup regardless of penetration size, and
nothing here reads these two columns to make any decision.

NO LOOKAHEAD: the Asian range for a given London-local calendar day is
computed from that day's 00:00-07:00 bars only, and is explicitly
blanked out (NaN) for any bar timestamped before that same session has
actually closed (07:00 local) - a bar can never see a same-day Asian
range value before that range has finished forming, even though the
underlying groupby technically has access to the whole day's data in
memory. Once set at 07:00, the value is frozen for the rest of the day
(never recomputed from later bars). See tests/test_london_sweep_signals.py
for tests that directly exercise this, including across a real BST/GMT
transition date.
"""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from indicators import atr

LONDON_TZ = ZoneInfo("Europe/London")


@dataclass
class LondonSweepConfig:
    asian_start_hour: int = 0
    asian_end_hour: int = 7            # exclusive - Asian session is [0, 7) London local time
    london_window_start_hour: int = 7  # inclusive - London entry window is [7, 10) London local time
    london_window_end_hour: int = 10   # exclusive
    atr_period: int = 14
    stop_buffer_atr_fraction: float = 0.1   # stop sits this many ATRs beyond the sweep extreme
    target_rr_multiple: float = 1.0         # fixed 1:1 reward:risk for round 1
    spread_avg_window: int = 100


DEFAULT_LONDON_SWEEP_CONFIG = LondonSweepConfig()


def _session_masks(df, config):
    """Boolean masks + a per-row calendar-day grouping key, all computed
    from Europe/London LOCAL wall-clock time (DST-aware) - never from a
    fixed UTC offset. `df.index` must already be tz-aware (UTC, as every
    OANDA fetch in this project produces)."""
    london_local = df.index.tz_convert(LONDON_TZ)
    hour = london_local.hour
    date_key = london_local.date

    is_asian = (hour >= config.asian_start_hour) & (hour < config.asian_end_hour)
    is_london_window = (hour >= config.london_window_start_hour) & (hour < config.london_window_end_hour)
    return is_asian, is_london_window, date_key, hour


def _compute_asian_range(df, is_asian, date_key, hour, config):
    """Per-London-local-day Asian high/low, computed from Asian-session
    bars only, then explicitly blanked out for any bar before that same
    day's Asian session has actually closed - see module docstring's
    NO LOOKAHEAD section. Days with no Asian-session bars in the frame
    (e.g. right at the start of the dataset) naturally produce NaN via
    the map() below, not an error."""
    asian_bars = df[is_asian]
    asian_by_day = asian_bars.groupby(date_key[is_asian]).agg(
        asian_high=("High", "max"), asian_low=("Low", "min")
    )

    day_series = pd.Series(date_key, index=df.index)
    asian_high = day_series.map(asian_by_day["asian_high"])
    asian_low = day_series.map(asian_by_day["asian_low"])

    not_yet_known = hour < config.asian_end_hour
    asian_high[not_yet_known] = float("nan")
    asian_low[not_yet_known] = float("nan")

    return asian_high, asian_low


def _detect_day_confirmations(day_df):
    """For ONE instrument's ONE London-local day's London-window bars,
    in chronological order: track at most one pending sweep per
    direction (updating to the MOST EXTREME point reached while a sweep
    is pending, not just the first touch), fire the FIRST close-back-
    inside confirmation in either direction, then stop entirely - no
    further bars this day are evaluated once a signal fires (one trade
    per instrument per day)."""
    n = len(day_df)
    signal_long = [False] * n
    signal_short = [False] * n
    penetration_price = [float("nan")] * n
    penetration_atr = [float("nan")] * n
    sweep_extreme_price = [float("nan")] * n

    sweep_up_extreme = None    # running most-extreme High while a bearish sweep is pending confirmation
    sweep_down_extreme = None  # mirror, Low

    rows = list(day_df.itertuples())
    for i, row in enumerate(rows):
        asian_high, asian_low, atr_now = row.asian_high, row.asian_low, row.atr
        if pd.isna(asian_high) or pd.isna(asian_low):
            continue  # Asian range not available - shouldn't happen inside the London window, but be safe

        # --- Bearish: sweep above asian_high, then close back below it ---
        if row.High > asian_high:
            sweep_up_extreme = max(sweep_up_extreme, row.High) if sweep_up_extreme is not None else row.High
        if sweep_up_extreme is not None and row.Close < asian_high:
            signal_short[i] = True
            pen = sweep_up_extreme - asian_high
            penetration_price[i] = pen
            penetration_atr[i] = pen / atr_now if atr_now and atr_now > 0 else float("nan")
            sweep_extreme_price[i] = sweep_up_extreme
            break  # one trade per instrument per day

        # --- Bullish: sweep below asian_low, then close back above it ---
        if row.Low < asian_low:
            sweep_down_extreme = min(sweep_down_extreme, row.Low) if sweep_down_extreme is not None else row.Low
        if sweep_down_extreme is not None and row.Close > asian_low:
            signal_long[i] = True
            pen = asian_low - sweep_down_extreme
            penetration_price[i] = pen
            penetration_atr[i] = pen / atr_now if atr_now and atr_now > 0 else float("nan")
            sweep_extreme_price[i] = sweep_down_extreme
            break  # one trade per instrument per day

    return pd.DataFrame({
        "signal_long": signal_long, "signal_short": signal_short,
        "sweep_penetration_price": penetration_price, "sweep_penetration_atr": penetration_atr,
        "sweep_extreme_price": sweep_extreme_price,
    }, index=day_df.index)


def prepare_instrument_frame(m15_df, config=DEFAULT_LONDON_SWEEP_CONFIG):
    """Build the M15-indexed signal frame for one instrument: ATR, the
    Asian range (no-lookahead, see module docstring), sweep+confirmation
    detection within the London entry window (one trade/day), the
    structural stop/target distances, spread, and the long/short entry
    signals - everything backtest_engine.py needs, using the generic
    stop_distance_override/target_distance_override mechanism
    combined_signals_4h.py already established."""
    df = m15_df.copy()

    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    is_asian, is_london_window, date_key, hour = _session_masks(df, config)
    df["asian_high"], df["asian_low"] = _compute_asian_range(df, is_asian, date_key, hour, config)

    df["signal_long"] = False
    df["signal_short"] = False
    df["sweep_penetration_price"] = float("nan")
    df["sweep_penetration_atr"] = float("nan")
    df["sweep_extreme_price"] = float("nan")

    london_window_df = df[is_london_window]
    london_window_dates = date_key[is_london_window]
    for _day, day_df in london_window_df.groupby(london_window_dates):
        result = _detect_day_confirmations(day_df)
        df.loc[result.index, "signal_long"] = result["signal_long"]
        df.loc[result.index, "signal_short"] = result["signal_short"]
        df.loc[result.index, "sweep_penetration_price"] = result["sweep_penetration_price"]
        df.loc[result.index, "sweep_penetration_atr"] = result["sweep_penetration_atr"]
        df.loc[result.index, "sweep_extreme_price"] = result["sweep_extreme_price"]

    # --- Structural stop/target distances - only meaningful on rows where a signal fired.
    # Reference price is this (confirmation) bar's own Close, the same
    # "estimate at signal time" convention used everywhere else in this
    # project (e.g. calculate_stop_and_target) - the engine fills at the
    # NEXT bar's open and re-derives absolute stop/target levels from the
    # real fill price, preserving this DISTANCE, not this estimated level.
    has_signal = df["signal_long"] | df["signal_short"]
    structural_distance = (df["sweep_extreme_price"] - df["Close"]).abs() + config.stop_buffer_atr_fraction * df["atr"]
    df["stop_distance_override"] = structural_distance.where(has_signal, float("nan"))
    df["target_distance_override"] = (structural_distance * config.target_rr_multiple).where(has_signal, float("nan"))

    # --- Spread, for the existing "is the spread normal?" check in risk_management.py ---
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    return df
