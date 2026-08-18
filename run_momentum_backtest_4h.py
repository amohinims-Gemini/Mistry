"""
run_momentum_backtest_4h.py
-------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the SINGLE-TIMEFRAME 4H TIME-SERIES MOMENTUM strategy
(signals_momentum_4h.py) - a genuinely different signal class from
everything else in this project (see that module's docstring), reusing
the same backtest engine, risk management, safety limits, data
pipeline, and metrics/reporting helpers as every other run_*.py script.

Only 4H data is fetched - this strategy has no use for 1H candles.

Same RiskConfig/engine adjustments required for correctness at 4H
granularity as every other 4H entry point: stale_data_max_gap_hours
widened to 6, bar_duration_hours=4 passed to run_backtest().

Expected trading character, going in: because momentum's sign is
defined on almost every bar (see signals_momentum_4h.py), this strategy
wants to be in a position almost continuously - a much higher trade
frequency than channel-breakout's or mean-reversion's sparser signals.

=============================================================================
Experiments - each is (label, MomentumConfig, RiskConfig).

ROUND 1 (lookback=20, spec-literal 1.5x/3.0x ATR stop/target, matching
signals_4h.py's channel window for comparability) result: a clean FAIL,
not a near-miss - 433 combined trades (comfortably clears the minimum),
but test PF 0.771 and test return -4.57%. Win rate hugged ~28-33% in
both periods, close to the ~33% breakeven this 2:1 R:R needs - a shape
statistically similar to signals_4h.py's own result (avg win rate 33.9%
across its stress test), despite a mechanically different signal
construction (raw trailing-return sign vs. channel breakout). Not
stress-tested - failed the single split outright.

ROUND 2 (this sweep): lookback is the one free parameter, target/stop
ATR multiples held at the spec-literal 1.5x/3.0x throughout - same
single-parameter discipline as every other sweep in this project.
=============================================================================
"""

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from signals_momentum_4h import prepare_instrument_frame, MomentumConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict, print_comparison_table,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

BAR_DURATION_HOURS = 4
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=6)  # spec-literal 1.5x/3.0x ATR stop/target throughout

LOOKBACKS_TO_TEST = [5, 10, 15, 20, 30, 50, 75, 100]

EXPERIMENTS = [
    (f"lookback {lb} bars" + (" (round 1 baseline)" if lb == 20 else ""), MomentumConfig(lookback_period=lb))
    for lb in LOOKBACKS_TO_TEST
]


def build_all_frames(raw_data, config=MomentumConfig()):
    frames = {}
    for symbol in INSTRUMENTS:
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h4, config=config)
    return frames


def run_experiment(label, signal_config, raw_data):
    frames = build_all_frames(raw_data, signal_config)
    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                 bar_duration_hours=BAR_DURATION_HOURS)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                bar_duration_hours=BAR_DURATION_HOURS)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)

    return {
        "label": label, "signal_config": signal_config, "risk_config": RISK_CONFIG,
        "train_result": train_result, "test_result": test_result,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "checks": checks,
        "split_info": (common_start, split_point, common_end),
    }


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("SINGLE-TIMEFRAME 4H TIME-SERIES MOMENTUM STRATEGY - a genuinely")
    print("different signal class (trailing-return sign), not a trend-breakout")
    print("or mean-reversion variant. See signals_momentum_4h.py.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()), granularities=("H4",))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric below is MEANINGLESS as a "
            "strategy evaluation on this data. ***\n"
        )

    experiment_results = []
    for i, (label, signal_config) in enumerate(EXPERIMENTS):
        print(f"\n\n{'#' * 78}")
        print(f"# EXPERIMENT: {label}")
        print(f"{'#' * 78}")

        result = run_experiment(label, signal_config, raw_data)
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
