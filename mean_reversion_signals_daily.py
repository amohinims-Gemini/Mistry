"""
mean_reversion_signals_daily.py
------------------------------------
Pure signal-generation logic for the SINGLE-TIMEFRAME DAILY mean-
reversion strategy - the daily-timeframe fallback, parallel to
mean_reversion_signals_4h.py. Bollinger/RSI entry AND the trend-
avoidance filter are all computed from the same daily series.

Rules (long/oversold; short is the exact mirror):
  1. Distance trigger: daily close is below the lower Bollinger Band
     (SMA(20) - 2 standard deviations)
  2. Momentum confirmation: RSI(14), on the same daily series, below 30
  3. Trend-avoidance filter: the same daily series' own trend-efficiency
     ratio is below a threshold - confirms the market has genuinely
     been range-bound, not trending, before betting on a reversion
  4. Spread within a normal range (checked in risk_management.py)

Parameters carried forward literally from mean_reversion_signals_4h.py
(Bollinger 20/2std, RSI 14, efficiency window 20 / threshold 0.3) -
round-1 baseline, expected to be re-tuned via sweep like every other
config in this project, not a claim these are already correct here.
A pre-build signal-frequency check found daily mean-reversion produces
noticeably fewer raw signal bars than trend-following (93 vs 504 across
all 4 instruments, ~6 years) - worth watching once this is backtested,
since it may struggle to clear the 150-trade minimum on its own.

ATR-based stop-loss/take-profit MULTIPLES reset to the literal spec
numbers (1.5x/1.0x) in RiskConfig - not the 4H-tuned values.

NO LOOKAHEAD: everything here uses only data up to and including the
current daily bar's close.
"""

from dataclasses import dataclass

from indicators import atr, sma, rolling_std, rsi as rsi_indicator


@dataclass
class MeanReversionDailyConfig:
    bollinger_period: int = 20
    bollinger_std_multiple: float = 2.0

    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    trend_efficiency_window: int = 20               # in daily bars
    trend_efficiency_max_threshold: float = 0.3      # REJECT entries above this

    # ATR series period only - stop/target MULTIPLES live in RiskConfig,
    # same split as every other strategy config in this project.
    atr_period: int = 14

    spread_avg_window: int = 100


DEFAULT_MEAN_REVERSION_DAILY_CONFIG = MeanReversionDailyConfig()


def prepare_instrument_frame(daily_df, config=DEFAULT_MEAN_REVERSION_DAILY_CONFIG):
    """Build a single daily-indexed DataFrame with everything
    backtest_engine.py needs: mid/bid/ask OHLC (already in daily_df),
    ATR, Bollinger Bands, RSI, the trend-avoidance filter, spread, and
    the long/short entry signals - all computed from ONE timeframe."""
    df = daily_df.copy()

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

    # --- Trend-avoidance filter, computed on the SAME daily series as entry ---
    window = config.trend_efficiency_window
    net_move = (df["Close"] - df["Close"].shift(window)).abs()
    path_length = df["Close"].diff().abs().rolling(window).sum()
    df["trend_efficiency_ratio"] = net_move / path_length
    range_bound = df["trend_efficiency_ratio"] < config.trend_efficiency_max_threshold

    # --- The two mirrored mean-reversion entry rules ---
    df["signal_long"] = (df["Close"] < df["lower_band"]) & (df["rsi"] < config.rsi_oversold) & range_bound
    df["signal_short"] = (df["Close"] > df["upper_band"]) & (df["rsi"] > config.rsi_overbought) & range_bound

    return df
