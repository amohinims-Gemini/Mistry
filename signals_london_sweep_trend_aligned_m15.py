"""
signals_london_sweep_trend_aligned_m15.py
-----------------------------------------------
V2 of the London Liquidity Sweep Reversal hypothesis - REUSES V1's
sweep+confirmation signal logic completely unchanged (imported, not
copied - see signals_london_sweep_m15.py, which remains untouched and
independently re-runnable as the concluded, REJECTED V1 experiment),
and adds ONE new gate: the reversal direction must align with the
higher-timeframe (daily) trend. Treated as a genuinely new hypothesis,
not a V1 parameter tweak - see README.md's "London Liquidity Sweep
Reversal V2" section for the full design rationale.

Research question: does a London liquidity sweep have positive
expectancy when the reversal direction is aligned with the established
higher-timeframe market trend?

Trend definition: daily EMA(50) vs EMA(200) - the same measure and
periods this project already uses everywhere else (signals.py,
signals_4h.py, signals_daily.py), computed on daily closes rather than
4H, since M15 needs a genuinely higher timeframe than the entry series
to count as "higher-timeframe" at all. ONE fixed trend definition, not
searched or swept across multiple period combinations.

Rule: V1's sweep+confirmation logic fires exactly as before, unchanged.
A LONG reversal signal is kept ONLY if the daily trend is UP
(EMA50>EMA200) as of that point; a SHORT reversal signal is kept ONLY
if the daily trend is DOWN. A signal that fires against the trend is
dropped entirely - never flipped into a countertrend trade.

Economic rationale: a sweep WITH the higher-timeframe trend is more
consistent with a genuine stop-hunt/liquidity-grab that resumes the
dominant order flow (a "spring"/"shakeout" pattern); a sweep AGAINST it
has to argue price reverses against the larger prevailing flow on local,
session-scale evidence alone. This is also a direct, falsifiable attempt
to explain V1's own strongest diagnostic finding - the severe long/short
asymmetry (long PF 0.977, short PF 0.455, consistent across both
instruments) - rather than a new, unmotivated guess.

NO LOOKAHEAD: daily EMA50/200 is computed on daily closes, then the
index is shifted forward by exactly one full day (24 hours) - a daily
candle in this project is indexed by its OPEN time (21:00 UTC) but isn't
actually knowable until it CLOSES, 24 hours later - before
merge_asof(direction="backward") onto the M15 timeline. This is the
EXACT "shift by this candle's own bar duration" mechanism signals.py's
prepare_4h_trend() already established (there: +4 hours for a 4H
candle; here: +24 hours for a daily one) - reused, not reinvented.
"""

from dataclasses import dataclass, field

import pandas as pd

from indicators import ema
from signals_london_sweep_m15 import prepare_instrument_frame as _v1_prepare_instrument_frame, LondonSweepConfig


@dataclass
class TrendAlignedLondonSweepConfig:
    sweep_config: LondonSweepConfig = field(default_factory=LondonSweepConfig)  # V1, unchanged
    trend_ema_fast_period: int = 50
    trend_ema_slow_period: int = 200


DEFAULT_TREND_ALIGNED_CONFIG = TrendAlignedLondonSweepConfig()


def _prepare_daily_trend(daily_df, config):
    """Daily EMA50/200 trend, index-shifted forward by one full day (to
    that candle's own close) so merge_asof can never see a trend reading
    before the daily candle it's based on has actually finished - see
    module docstring's NO LOOKAHEAD section."""
    out = pd.DataFrame(index=daily_df.index)
    ema_fast = ema(daily_df["Close"], config.trend_ema_fast_period)
    ema_slow = ema(daily_df["Close"], config.trend_ema_slow_period)
    out["daily_trend_up"] = ema_fast > ema_slow
    out["daily_trend_down"] = ema_fast < ema_slow
    out.index = out.index + pd.Timedelta(hours=24)
    return out


def prepare_instrument_frame(m15_df, daily_df, config=DEFAULT_TREND_ALIGNED_CONFIG):
    """Build the V2 signal frame: V1's sweep+confirmation frame,
    unchanged, with signal_long/signal_short additionally gated by daily
    trend alignment. Everything else (ATR, stop/target distances, sweep
    penetration, spread) is exactly what V1 already computed - only the
    two signal columns (and their associated stop/target overrides, on
    rows a signal gets dropped from) are narrowed."""
    df = _v1_prepare_instrument_frame(m15_df, config=config.sweep_config)

    trend = _prepare_daily_trend(daily_df, config)
    df = pd.merge_asof(
        df.sort_index(), trend.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    # merge_asof leaves NaN before any daily trend data exists yet (EMA200's
    # own warmup) - treat "trend not yet known" as "not trending", same
    # convention signals.py's own 4H-trend merge already uses.
    df["daily_trend_up"] = df["daily_trend_up"].astype("boolean").fillna(False).astype(bool)
    df["daily_trend_down"] = df["daily_trend_down"].astype("boolean").fillna(False).astype(bool)

    aligned_long = df["signal_long"] & df["daily_trend_up"]
    aligned_short = df["signal_short"] & df["daily_trend_down"]

    # A signal dropped by the trend gate shouldn't leave a dangling
    # stop/target override behind.
    dropped = (df["signal_long"] & ~aligned_long) | (df["signal_short"] & ~aligned_short)
    df.loc[dropped, "stop_distance_override"] = float("nan")
    df.loc[dropped, "target_distance_override"] = float("nan")

    df["signal_long"] = aligned_long
    df["signal_short"] = aligned_short

    return df
