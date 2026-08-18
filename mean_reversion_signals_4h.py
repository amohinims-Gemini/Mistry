"""
mean_reversion_signals_4h.py
--------------------------------
Pure signal-generation logic for the SINGLE-TIMEFRAME 4H mean-reversion
strategy - Bollinger/RSI entry AND the trend-avoidance filter all
computed from the SAME 4H series, unlike mean_reversion_signals.py
(1H Bollinger/RSI entry, 4H trend-avoidance filter - two different
clocks). Mirrors exactly how signals_4h.py relates to signals.py for the
trend-following strategy: built to test whether the stability found by
moving trend-following to a single 4H timeframe (much tighter, more
consistent drawdown control across the stress test, though no real edge
found yet) transfers to mean-reversion too.

Rules (long/oversold; short is the exact mirror):
  1. Distance trigger: 4H close is below the lower Bollinger Band
     (SMA(20) - 2 standard deviations)
  2. Momentum confirmation: RSI(14), computed on the same 4H series, is
     below 30
  3. Trend-avoidance filter: the SAME 4H series' own trend-efficiency
     ratio is below a threshold - confirms the market has genuinely been
     range-bound over the recent 4H window, not trending strongly,
     before betting on a reversion
  4. Spread within a normal range for the instrument (checked in
     risk_management.py, same mechanism as every strategy in this project)

Starting R:R (1.5x stop / 1.0x target ATR) is the literal original
proposal's numbers, NOT the 1H-tuned values (2.5x/1.0x) found for
mean_reversion_signals.py - reusing a value tuned for a different
timeframe's noise characteristics would be exactly the mistake avoided
for the trend-following strategy's re-derivation.

NO LOOKAHEAD: everything here uses only data up to and including the
current 4H bar's close. No cross-timeframe merge is needed at all.
"""

from dataclasses import dataclass

from indicators import atr, sma, rolling_std, rsi as rsi_indicator


@dataclass
class MeanReversion4HConfig:
    bollinger_period: int = 20
    bollinger_std_multiple: float = 2.0

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    trend_efficiency_window: int = 20               # in 4H bars
    trend_efficiency_max_threshold: float = 0.3      # REJECT entries above this

    # ATR series period only - stop/target MULTIPLES live in RiskConfig,
    # same split as every other strategy config in this project.
    atr_period: int = 14

    spread_avg_window: int = 100


DEFAULT_MEAN_REVERSION_4H_CONFIG = MeanReversion4HConfig()


def prepare_instrument_frame(h4_df, config=DEFAULT_MEAN_REVERSION_4H_CONFIG):
    """
    Build a single 4H-indexed DataFrame with everything backtest_engine.py
    needs: mid/bid/ask OHLC (already in h4_df), ATR, Bollinger Bands, RSI,
    the trend-avoidance filter, spread, and the long/short entry signals -
    all computed from ONE timeframe, no merge_asof required.
    """
    df = h4_df.copy()

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

    # --- Trend-avoidance filter, computed on the SAME 4H series as entry ---
    window = config.trend_efficiency_window
    net_move = (df["Close"] - df["Close"].shift(window)).abs()
    path_length = df["Close"].diff().abs().rolling(window).sum()
    df["trend_efficiency_ratio"] = net_move / path_length
    range_bound = df["trend_efficiency_ratio"] < config.trend_efficiency_max_threshold

    # --- The two mirrored mean-reversion entry rules ---
    df["signal_long"] = (df["Close"] < df["lower_band"]) & (df["rsi"] < config.rsi_oversold) & range_bound
    df["signal_short"] = (df["Close"] > df["upper_band"]) & (df["rsi"] > config.rsi_overbought) & range_bound

    return df
