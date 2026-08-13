"""
signals.py
-----------
Pure signal-generation logic: given prepared 4H and 1H OANDA candle data
for one instrument, produce a per-1H-bar "long entry available" / "short
entry available" signal based on the strategy's technical rules:

  1. Trend filter (4H): fast EMA above slow EMA = uptrend, longs allowed
     (mirror: fast EMA below slow EMA = downtrend, shorts allowed)
  2. Entry trigger (1H): this bar's close breaks above the highest High
     of the previous N COMPLETED 1H candles (mirror: breaks below the
     lowest Low of the previous N)
  3. (Optional, off by default) Volatility filter: skip signals when ATR
     is below its own trailing average - avoids taking breakouts during
     unusually quiet, chop-prone conditions.
  4. (Optional, off by default) Breakout buffer: require the close to
     clear the breakout level by a small ATR-scaled margin, rather than
     by any amount at all - filters out marginal/ambiguous breakouts.
  5. (Optional, off by default) Efficiency Ratio filter: skip signals
     unless the market has been moving DIRECTIONALLY efficiently lately,
     not just cleared a price level - see below, this is the one filter
     backed by an actual diagnosed pattern rather than a hypothesis.

All of this is configurable via SignalConfig so run_backtest.py can try
several parameter/filter combinations without editing this file.

*** channel_period default is 20, the original spec value. *** An
evidence-based deviation to 95 was tried and adopted for a while, but a
later bug-fix round (correcting a leverage-sizing bug and excluding an
unrealistic financing assumption from scoring) invalidated the evidence
behind it - re-run under corrected math, 95 no longer outperformed, and
NEITHER value reliably cleared the spec's profit-factor bar across an
11-scenario robustness stress test. Channel length is no longer being
treated as a productive tuning lever; see README.md for the full history.

*** efficiency_ratio filter (#5) exists because of an actual diagnosed
weakness, not a hypothesis tried on spec. *** A trade-level diagnostic
(comparing winning vs losing trades on an INDEPENDENT trend-strength
measure, not the strategy's own EMA filter - every trade already passes
that by construction, so it can't explain win/loss variation on its own)
found a strong, consistent pattern across two different channel-length
configurations: entries taken during choppy/inefficient price action
lose badly (win rate as low as 8%), entries taken during genuinely
efficient directional moves do much better (37-39%, near or above the
2:1 R:R's ~33% breakeven). The EMA50/200 filter is evidently a weak
proxy for "is this actually trending" - it's often satisfied during
sideways chop that sits just above/below the long-term average. This is
DIFFERENT from the (already-tried-and-rejected) volatility filter: that
measured raw ATR level; this measures directional EFFICIENCY, which is
what the diagnostic actually separated on. Off by default until swept
and stress-tested with the same rigor as everything else in this file.

This file only decides "is there a technically valid setup right now" -
it doesn't know about spreads, account risk, or safety limits. That's
risk_management.py's job. Keeping the two separate means the technical
strategy logic can be reasoned about and tested on its own.

NO LOOKAHEAD: every indicator here is computed so a value "as of" a given
1H bar only ever uses information that was actually available by that
bar's close:
  - the 4H trend is merged onto the 1H timeline using only 4H candles
    that had already CLOSED (via merge_asof, direction="backward", on an
    index shifted forward by one 4H period - see prepare_4h_trend)
  - the breakout channel and the volatility filter's own trailing
    average both explicitly exclude the current bar (via .shift(1))
  - the efficiency ratio is NOT shifted - it deliberately includes the
    current bar's own move (net_move and path_length both run up to and
    including bar T), because the question it answers is "has price
    action up to and including right now been efficient", which needs
    the current bar's contribution. This is not lookahead: everything it
    uses is known by bar T's close, same as ATR (also unshifted).
"""

from dataclasses import dataclass

import pandas as pd
from indicators import ema, atr, rolling_high, rolling_low


@dataclass
class SignalConfig:
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    channel_period: int = 20  # the original spec value - see module docstring for the history
    atr_period: int = 14
    spread_avg_window: int = 100

    # Entry-filter additions - all OFF by default (matches the original spec)
    use_volatility_filter: bool = False
    volatility_filter_window: int = 100      # ATR's own trailing-average window
    breakout_buffer_atr_fraction: float = 0.0  # e.g. 0.1 = must clear the level by 0.1x ATR

    # Efficiency Ratio filter - backed by an actual diagnosed pattern (see
    # module docstring), not a hypothesis. Window is FIXED at 20 to match
    # exactly what the diagnostic validated - only the threshold should be
    # swept, deliberately keeping this a single-parameter search rather
    # than reopening a second free dimension (see the channel-length
    # saga in README.md for why that's a real risk, not paranoia).
    use_efficiency_filter: bool = False
    efficiency_ratio_window: int = 20
    efficiency_ratio_min_threshold: float = 0.3  # starting point - to be swept


DEFAULT_SIGNAL_CONFIG = SignalConfig()


def prepare_4h_trend(h4_df, config=DEFAULT_SIGNAL_CONFIG):
    """Compute the 4H trend filter. Returns a DataFrame whose index has
    been shifted forward by 4 hours - from "this candle's OPEN time" to
    "this candle's CLOSE time" - so that merge_asof in
    prepare_instrument_frame() naturally lines each 1H bar up with the
    most recently *closed* 4H candle. A 4H candle's trend reading isn't
    "known" until it actually finishes."""
    out = pd.DataFrame(index=h4_df.index)
    out["ema_fast"] = ema(h4_df["Close"], config.ema_fast_period)
    out["ema_slow"] = ema(h4_df["Close"], config.ema_slow_period)
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]
    out["trend_down"] = out["ema_fast"] < out["ema_slow"]

    out.index = out.index + pd.Timedelta(hours=4)
    return out


def prepare_instrument_frame(h1_df, h4_df, config=DEFAULT_SIGNAL_CONFIG):
    """
    Combine 1H and 4H data for one instrument into a single 1H-indexed
    DataFrame with everything the backtest engine needs: mid/bid/ask
    OHLC, ATR, the breakout channel, the 4H trend filter, spread, and the
    long/short technical entry signals.
    """
    df = h1_df.copy()

    # --- Volatility, for stop-loss/take-profit sizing AND (optionally) the
    # volatility filter below.
    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    # --- Entry trigger: breakout of the last N COMPLETED candles.
    # rolling(...).max() over the last channel_period bars includes the
    # CURRENT bar by default - .shift(1) drops it, so "highest of the
    # previous N" really means previous, not "these N including now".
    df["breakout_high"] = rolling_high(df["High"], config.channel_period).shift(1)
    df["breakout_low"] = rolling_low(df["Low"], config.channel_period).shift(1)

    # --- Spread, for the "is the spread normal?" check in risk_management.py.
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    # shift(1): the trailing average must not include the current bar's own spread
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    # --- 4H trend filter, aligned onto the 1H index without lookahead.
    trend = prepare_4h_trend(h4_df, config)
    df = pd.merge_asof(
        df.sort_index(), trend.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    # merge_asof leaves NaN for 1H bars before any 4H trend data existed yet
    # (the first ema_slow_period 4H candles' worth of warm-up), which turns
    # these columns into object dtype - explicitly convert back to a clean
    # bool (treating "trend not yet known" as "not trending") rather than
    # relying on plain .fillna(), which pandas now warns is ambiguous here.
    df["trend_up"] = df["trend_up"].astype("boolean").fillna(False).astype(bool)
    df["trend_down"] = df["trend_down"].astype("boolean").fillna(False).astype(bool)

    # --- Optional volatility filter: only signal when ATR is above its own
    # trailing average (skip unusually quiet, chop-prone stretches). shift(1)
    # so the trailing average never includes the current bar's own ATR.
    if config.use_volatility_filter:
        atr_trailing_avg = df["atr"].rolling(config.volatility_filter_window).mean().shift(1)
        volatility_ok = df["atr"] > atr_trailing_avg
    else:
        volatility_ok = pd.Series(True, index=df.index)

    # --- Optional breakout buffer: require the close to clear the breakout
    # level by a small ATR-scaled margin rather than by any amount. Uses
    # THIS bar's own ATR, which is fine - ATR at bar T is built from True
    # Range values up to and including bar T, all known by bar T's close.
    long_breakout_level = df["breakout_high"] + config.breakout_buffer_atr_fraction * df["atr"]
    short_breakout_level = df["breakout_low"] - config.breakout_buffer_atr_fraction * df["atr"]

    # --- Optional Efficiency Ratio filter: only signal when price has been
    # moving DIRECTIONALLY, not just churning. Kaufman's Efficiency Ratio:
    #   ER = |net move over the window| / |sum of bar-to-bar moves over the window|
    # ER near 1 = price moved efficiently in one direction (a real trend).
    # ER near 0 = lots of back-and-forth with little net progress (chop).
    # Deliberately NOT shifted - see the module docstring's NO LOOKAHEAD
    # section for why that's still lookahead-safe.
    if config.use_efficiency_filter:
        window = config.efficiency_ratio_window
        net_move = (df["Close"] - df["Close"].shift(window)).abs()
        path_length = df["Close"].diff().abs().rolling(window).sum()
        efficiency_ratio = net_move / path_length
        df["efficiency_ratio"] = efficiency_ratio
        efficiency_ok = efficiency_ratio > config.efficiency_ratio_min_threshold
    else:
        efficiency_ok = pd.Series(True, index=df.index)

    # --- The two mirrored technical entry rules, plus whichever optional
    # filters are enabled. (Rule 3 - spread - is checked in
    # risk_management.py, since it's a risk/execution concern rather than
    # "is there a technical setup".)
    df["signal_long"] = df["trend_up"] & (df["Close"] > long_breakout_level) & volatility_ok & efficiency_ok
    df["signal_short"] = df["trend_down"] & (df["Close"] < short_breakout_level) & volatility_ok & efficiency_ok

    return df
