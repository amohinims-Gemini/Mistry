"""
run_mean_reversion_backtest_4h.py
-------------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the SINGLE-TIMEFRAME 4H mean-reversion strategy
(mean_reversion_signals_4h.py) - parallel to run_backtest_4h.py (trend-
following) and run_mean_reversion_backtest.py (1H mean-reversion),
reusing the same backtest engine, risk management, safety limits, data
pipeline, and metrics/reporting helpers.

Only 4H data is fetched - no 1H candles needed for this strategy.

Same two RiskConfig/engine adjustments required for correctness at 4H as
run_backtest_4h.py: stale_data_max_gap_hours widened to 6, and
bar_duration_hours=4 passed to run_backtest() so the overnight financing
rollover check works correctly (see backtest_engine.py's docs).

Usage:
    source venv/bin/activate
    python run_mean_reversion_backtest_4h.py

Expected statistical shape: HIGH win rate with small average wins, same
as the 1H mean-reversion strategy - a win rate near 33% here is a red
flag, not a normal baseline.
"""

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from mean_reversion_signals_4h import prepare_instrument_frame, MeanReversion4HConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict, print_comparison_table,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

# =============================================================================
# Experiments - each is (label, MeanReversion4HConfig, RiskConfig).
#
# ROUND 1 (stop=1.5x/target=1.0x, untuned) result: the closest-to-
# breakeven, most stable result found anywhere in this project without
# any tuning at all. Win rate 61.2%/57.9% straddled the ~60% breakeven
# this R:R requires within 2-3 points either side; drawdown never
# exceeded 2.38% in either period (far tighter than any 1H config).
# Still failed OOS-positive and PF>1.2, but by the smallest margins seen.
#
# ROUND 2 (this sweep): target FIXED at 1.0x ATR, sweeping ONLY the stop
# multiple - one free parameter, same discipline, mirroring exactly the
# 1H mean-reversion sweep's structure for direct comparability. No
# assumption that the 1H finding (wider stop -> higher win rate) repeats
# here - testing both directions empirically.
# =============================================================================

BAR_DURATION_HOURS = 4
TARGET_ATR_MULTIPLE = 1.0
STOP_MULTIPLES_TO_TEST = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

EXPERIMENTS = [
    (f"stop {s}x / target {TARGET_ATR_MULTIPLE}x ATR" + (" (round 1 baseline)" if s == 1.5 else ""),
     MeanReversion4HConfig(),
     RiskConfig(stale_data_max_gap_hours=6, stop_loss_atr_multiple=s, take_profit_atr_multiple=TARGET_ATR_MULTIPLE))
    for s in STOP_MULTIPLES_TO_TEST
]


def build_all_frames(raw_data, mr_config=MeanReversion4HConfig()):
    frames = {}
    for symbol in INSTRUMENTS:
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h4, config=mr_config)
    return frames


def run_experiment(label, mr_config, risk_config, raw_data):
    frames = build_all_frames(raw_data, mr_config)
    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=risk_config,
                                 bar_duration_hours=BAR_DURATION_HOURS)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=risk_config,
                                bar_duration_hours=BAR_DURATION_HOURS)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)

    return {
        "label": label, "mr_config": mr_config, "risk_config": risk_config,
        "train_result": train_result, "test_result": test_result,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "checks": checks,
        "split_info": (common_start, split_point, common_end),
    }


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("SINGLE-TIMEFRAME 4H MEAN-REVERSION STRATEGY - Bollinger/RSI entry and")
    print("trend-avoidance filter all computed from the same 4H series.")
    print("Expect a HIGH win rate with small average wins - a win rate near 33%")
    print("here is a red flag, not a normal baseline.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()), granularities=("H4",))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric below is MEANINGLESS as a "
            "strategy evaluation on this data. ***\n"
        )

    experiment_results = []
    for i, (label, mr_config, risk_config) in enumerate(EXPERIMENTS):
        print(f"\n\n{'#' * 78}")
        print(f"# EXPERIMENT: {label}")
        print(f"{'#' * 78}")

        result = run_experiment(label, mr_config, risk_config, raw_data)
        experiment_results.append(result)

        common_start, split_point, common_end = result["split_info"]
        print(f"Training period: {common_start} to {split_point}")
        print(f"Testing period:  {split_point} to {common_end}")

        if i == 0:
            print_period_report("TRAINING", result["train_result"], result["train_metrics"])
            print_period_report("TESTING (unseen data)", result["test_result"], result["test_metrics"])
            print_final_verdict(result["checks"])
        else:
            tm, sm, c = result["train_metrics"], result["test_metrics"], result["checks"]
            print(f"Train: return {tm['total_return_pct']:.2f}% (trading only), drawdown {tm['max_drawdown_pct']:.2f}%, "
                  f"{tm['n_completed_trades']} trades, win rate {tm['win_rate_pct']:.1f}%, PF {tm['profit_factor']:.3f}")
            print(f"Test:  return {sm['total_return_pct']:.2f}% (trading only), drawdown {sm['max_drawdown_pct']:.2f}%, "
                  f"{sm['n_completed_trades']} trades, win rate {sm['win_rate_pct']:.1f}%, PF {sm['profit_factor']:.3f}")
            print(f"Meets all 4 requirements: {'YES' if c['all_pass'] else 'no'}")

    print_comparison_table(experiment_results)


if __name__ == "__main__":
    main()
