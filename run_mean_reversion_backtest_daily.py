"""
run_mean_reversion_backtest_daily.py
-----------------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the SINGLE-TIMEFRAME DAILY mean-reversion strategy
(mean_reversion_signals_daily.py) - the daily-timeframe fallback,
parallel to run_mean_reversion_backtest_4h.py.

A pre-build signal-frequency check found daily mean-reversion produces
noticeably fewer raw signal bars than trend-following at this
granularity (93 vs 504 across all 4 instruments over ~6 years) -
flagged before building as a real risk that this may fall short of the
150-trade minimum on its own; watch n_completed_trades below.

RiskConfig/engine adjustments required for correctness at daily
granularity: same two as run_backtest_daily.py (stale_data_max_gap_hours
widened to 30, bar_duration_hours=24). See that file's docstring for
the econ-calendar no-op limitation, which applies here too.

Usage:
    source venv/bin/activate
    python run_mean_reversion_backtest_daily.py
"""

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from mean_reversion_signals_daily import prepare_instrument_frame, MeanReversionDailyConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

BAR_DURATION_HOURS = 24
MR_CONFIG = MeanReversionDailyConfig()
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=30)  # spec-literal 1.5x/1.0x ATR stop/target


def build_all_frames(raw_data, config=MR_CONFIG):
    frames = {}
    for symbol in INSTRUMENTS:
        d1 = raw_data[symbol]["D"]
        frames[symbol] = prepare_instrument_frame(d1, config=config)
    return frames


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("SINGLE-TIMEFRAME DAILY MEAN-REVERSION STRATEGY - Bollinger/RSI entry and")
    print("trend-avoidance filter all computed from the same daily series.")
    print("NOTE: the econ-calendar blackout filter is a documented no-op at this")
    print("granularity - see run_backtest_daily.py's docstring.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()), granularities=("D",))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric below is MEANINGLESS as a "
            "strategy evaluation on this data. ***\n"
        )

    frames = build_all_frames(raw_data)
    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)
    print(f"Training period: {common_start} to {split_point}")
    print(f"Testing period:  {split_point} to {common_end}")

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                 bar_duration_hours=BAR_DURATION_HOURS)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                bar_duration_hours=BAR_DURATION_HOURS)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)

    print_period_report("TRAINING", train_result, train_metrics)
    print_period_report("TESTING (unseen data)", test_result, test_metrics)
    print_final_verdict(checks)


if __name__ == "__main__":
    main()
