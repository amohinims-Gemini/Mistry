"""
mean_reversion_signals.py
----------------------------
Pure signal-generation logic for the MEAN-REVERSION strategy - a
deliberately separate module from signals.py (the trend-following
strategy), not a variant of it. Different indicators (Bollinger Bands +
RSI vs EMA + breakout), and a genuinely different expected statistical
shape: this strategy should show a HIGH win rate with small average
wins, the opposite profile from the trend-following strategy's ~30% win
rate / big-winner shape. A win rate near 33% here would be a red flag,
not the normal baseline it was for the other strategy.

Given prepared 4H and 1H OANDA candle data for one instrument, produces
a per-1H-bar "long entry available" / "short entry available" signal
based on:

  1. Distance trigger (1H): close is below the lower Bollinger Band
     (SMA(20) - 2 standard deviations) - price is "unusually far" below
     its recent average. (Mirror for shorts: close above the upper band.)
  2. Momentum confirmation (1H): RSI(14) is below 30 - the move shows
     genuine exhaustion, not just a single noisy tick outside the bands.
     (Mirror: RSI above 70.)
  3. Trend-avoidance filter (4H): the 4H trend-efficiency ratio (the
     SAME measure used in signals.py for the trend-following strategy,
     here with the comparison FLIPPED) is BELOW a threshold - confirms
     the broader market is genuinely range-bound, not trending strongly,
     before betting on a reversion. Fading a strong trend is the classic
     mean-reversion mistake this exists to avoid.
  4. Spread within a normal range for the instrument (checked in
     risk_management.py, same mechanism as the trend-following strategy).

STOP-LOSS / TAKE-PROFIT: static ATR-based levels set once at entry, same
mechanism as the trend-following strategy (via RiskConfig's
stop_loss_atr_multiple/take_profit_atr_multiple - NOT duplicated here,
see below) - deliberately NOT a dynamic "exit when price reverts to the
moving average" target. That's the more textbook-authentic mean-
reversion exit, but it needs new per-bar engine logic the current
backtest_engine.py doesn't have; starting with the simpler, fully-
reusable static-target design to validate the core idea first.

CONFIG SPLIT, same pattern as signals.py/SignalConfig +
risk_management.py/RiskConfig: MeanReversionConfig here holds only
entry-signal parameters (Bollinger, RSI, trend-avoidance). Stop-loss/
take-profit ATR multiples, position sizing, and every safety limit stay
in the EXISTING RiskConfig - pass a RiskConfig with different ATR
multiples alongside this config, don't duplicate them here.

NO LOOKAHEAD, same discipline as signals.py:
  - Bollinger/RSI/ATR at bar T use only data up to and including bar T's
    close - not shifted, because they're deliberately meant to include
    the current bar, and everything they use is already known by then.
  - The 4H trend-efficiency ratio is merged onto the 1H timeline using
    only 4H candles that had already CLOSED (via merge_asof, direction=
    "backward", on an index shifted forward by one 4H period) - the
    identical technique signals.py's prepare_4h_trend uses, duplicated
    here (not imported) so this module stays fully independent.
"""

from dataclasses import dataclass

import pandas as pd
from indicators import atr, sma, rolling_std, rsi as rsi_indicator


@dataclass
class MeanReversionConfig:
    bollinger_period: int = 20
    bollinger_std_multiple: float = 2.0

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Trend-avoidance filter - reuses the Efficiency Ratio measure built
    # for the trend-following strategy's redesign, comparison FLIPPED:
    # there, high efficiency was good (a real trend). Here, low
    # efficiency is good (genuinely range-bound, safe to fade).
    trend_efficiency_window: int = 20              # in 4H bars
    trend_efficiency_max_threshold: float = 0.3     # REJECT entries above this

    # ATR series period only - the STOP/TARGET MULTIPLES applied to it
    # live in RiskConfig (stop_loss_atr_multiple/take_profit_atr_multiple),
    # not here, matching the existing SignalConfig/RiskConfig split.
    atr_period: int = 14

    spread_avg_window: int = 100


DEFAULT_MEAN_REVERSION_CONFIG = MeanReversionConfig()


def _prepare_4h_trend_efficiency(h4_df, config):
    """4H trend-efficiency ratio, with its index shifted forward by 4
    hours (open time -> close time) so merge_asof in
    prepare_instrument_frame() lines each 1H bar up with the most
    recently *closed* 4H candle - the same technique as signals.py's
    prepare_4h_trend, duplicated rather than imported."""
    window = config.trend_efficiency_window
    net_move = (h4_df["Close"] - h4_df["Close"].shift(window)).abs()
    path_length = h4_df["Close"].diff().abs().rolling(window).sum()

    out = pd.DataFrame(index=h4_df.index)
    out["trend_efficiency_ratio"] = net_move / path_length
    out.index = out.index + pd.Timedelta(hours=4)
    return out


def prepare_instrument_frame(h1_df, h4_df, config=DEFAULT_MEAN_REVERSION_CONFIG):
    """
    Combine 1H and 4H data for one instrument into a single 1H-indexed
    DataFrame with everything backtest_engine.py needs: mid/bid/ask OHLC
    (already in h1_df), ATR, Bollinger Bands, RSI, the 4H trend-avoidance
    filter, spread, and the long/short mean-reversion entry signals.

    Produces the same required columns as signals.prepare_instrument_frame
    (atr, spread_close, avg_spread_100, signal_long, signal_short) so
    backtest_engine.py works completely unchanged regardless of which
    strategy's frame it's given.
    """
    df = h1_df.copy()

    # --- Volatility, for stop-loss/take-profit sizing (multiples come from RiskConfig) ---
    df["atr"] = atr(df["High"], df["Low"], df["Close"], config.atr_period)

    # --- Bollinger Bands: "how far from the recent average, in std devs" ---
    df["sma"] = sma(df["Close"], config.bollinger_period)
    df["rolling_std"] = rolling_std(df["Close"], config.bollinger_period)
    df["lower_band"] = df["sma"] - config.bollinger_std_multiple * df["rolling_std"]
    df["upper_band"] = df["sma"] + config.bollinger_std_multiple * df["rolling_std"]

    # --- RSI: momentum confirmation ---
    df["rsi"] = rsi_indicator(df["Close"], config.rsi_period)

    # --- Spread, for the "is the spread normal?" check in risk_management.py ---
    df["spread_close"] = df["Ask_Close"] - df["Bid_Close"]
    df["avg_spread_100"] = df["spread_close"].rolling(config.spread_avg_window).mean().shift(1)

    # --- 4H trend-avoidance filter, aligned onto the 1H index without lookahead ---
    trend_eff = _prepare_4h_trend_efficiency(h4_df, config)
    df = pd.merge_asof(
        df.sort_index(), trend_eff.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    # NaN (not enough 4H history yet) compares as False via `<` - be
    # conservative, treat "don't know if it's trending" as "don't trade".
    range_bound = df["trend_efficiency_ratio"] < config.trend_efficiency_max_threshold

    # --- The two mirrored mean-reversion entry rules ---
    df["signal_long"] = (df["Close"] < df["lower_band"]) & (df["rsi"] < config.rsi_oversold) & range_bound
    df["signal_short"] = (df["Close"] > df["upper_band"]) & (df["rsi"] > config.rsi_overbought) & range_bound

    return df
