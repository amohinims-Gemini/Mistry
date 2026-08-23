"""
run_london_sweep_trend_aligned_backtest.py
------------------------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for London Liquidity Sweep Reversal V2 - Higher-Timeframe
Trend-Aligned (signals_london_sweep_trend_aligned_m15.py). Same M15,
EUR_USD/GBP_USD scope as V1, plus daily candle data for the trend gate.
Reuses V1's sweep+confirmation logic completely unchanged (imported,
not copied) - see that module's docstring. Same backtest engine, risk
management, safety limits, and OANDA data pipeline as every other
strategy in this project.

DEVELOPMENT DATA ONLY, per explicit instruction: uses
dataset_split.split_for_iteration() exclusively - the validation and
final reserved periods are not reachable through this script at all.

*** THIS SCRIPT HAS NOT BEEN RUN YET. *** Built and tested (see
tests/test_london_sweep_trend_aligned_signals.py), matching the same
build-then-test-then-wait-for-approval discipline used for V1. Do not
run it without explicit approval.

Usage (once approved):
    source venv/bin/activate
    python run_london_sweep_trend_aligned_backtest.py
"""

from instruments import INSTRUMENTS
from data_fetch import get_instrument_data
from signals_london_sweep_trend_aligned_m15 import prepare_instrument_frame, TrendAlignedLondonSweepConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from dataset_split import split_for_iteration, describe_split
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

LONDON_SWEEP_SYMBOLS = ["EUR_USD", "GBP_USD"]  # same scope as V1, deliberately not PORTFOLIO_SYMBOLS

BAR_DURATION_HOURS = 0.25  # M15 - the tradeable timeframe; the daily trend context doesn't change this
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=0.5)  # identical to V1 - unchanged risk management

SIGNAL_CONFIG = TrendAlignedLondonSweepConfig()


def fetch_raw_data():
    raw = {}
    for symbol in LONDON_SWEEP_SYMBOLS:
        oanda_symbol = INSTRUMENTS[symbol].oanda_symbol
        raw[symbol] = {
            "M15": get_instrument_data(oanda_symbol, "M15"),
            "D": get_instrument_data(oanda_symbol, "D"),
        }
    return raw


def build_all_frames(raw_data, config=SIGNAL_CONFIG):
    frames = {}
    for symbol in LONDON_SWEEP_SYMBOLS:
        frames[symbol] = prepare_instrument_frame(raw_data[symbol]["M15"], raw_data[symbol]["D"], config=config)
    return frames


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("LONDON LIQUIDITY SWEEP REVERSAL V2 - HIGHER-TIMEFRAME TREND-ALIGNED")
    print("M15, EUR_USD/GBP_USD only. Daily EMA50/200 trend gate on V1's")
    print("unchanged sweep+confirmation signal. See")
    print("signals_london_sweep_trend_aligned_m15.py for the full design.")
    print("=" * 78)
    print()
    print(describe_split())
    print()
    print("DEVELOPMENT DATA ONLY - validation and final reserved periods are not")
    print("reachable through this script.")
    print("=" * 78)

    raw_data = fetch_raw_data()
    frames = build_all_frames(raw_data)

    development_frames, _validation_frames_not_used_this_round = split_for_iteration(frames)
    # _validation_frames_not_used_this_round is intentionally never touched.

    train_frames, test_frames, common_start, split_point, common_end = split_train_test(development_frames)
    print(f"\nDevelopment-internal training period: {common_start} to {split_point}")
    print(f"Development-internal testing period:  {split_point} to {common_end}")

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                 bar_duration_hours=BAR_DURATION_HOURS)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                bar_duration_hours=BAR_DURATION_HOURS)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)

    print_period_report("DEVELOPMENT-TRAIN", train_result, train_metrics)
    print_period_report("DEVELOPMENT-TEST", test_result, test_metrics)
    print_final_verdict(checks)


if __name__ == "__main__":
    main()
