"""
run_mean_reversion_backtest.py
---------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the MEAN-REVERSION strategy (mean_reversion_signals.py) -
parallel to run_backtest.py (the trend-following strategy), reusing the
same backtest engine, risk management, safety limits, data pipeline, and
metrics/reporting helpers. Only signal generation differs.

Usage:
    source venv/bin/activate
    python run_mean_reversion_backtest.py

Expected statistical shape is DIFFERENT from the trend-following
strategy: mean-reversion should show a HIGH win rate with small average
wins, not the ~30% win rate / big-winner shape the other strategy has.
A win rate near 33% here is a red flag, not a normal baseline.
"""

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from data_fetch import fetch_all
from mean_reversion_signals import prepare_instrument_frame, MeanReversionConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict, print_comparison_table,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

# =============================================================================
# Experiments - each is (label, MeanReversionConfig, RiskConfig).
#
# ROUND 2 (stop sweep, target fixed at 1.0x): tightening (0.6x-1.2x) made
# things worse - win rate climbs monotonically as the stop WIDENS (28% at
# 0.6x -> 56% at 1.5x test), opposite of the original hypothesis. Widening
# further (1.75x-2.5x) kept improving it: stop=2.5x reached avg test win
# rate 67% and was the first config to ever pass all 4 bars on a single
# split - BUT an 11-scenario stress test showed it's still net-losing on
# average (avg test PF 0.852, avg return -1.75%), just less badly than
# 1.5x. Real, validated improvement (win rate higher in every scenario,
# not just the one it was found on) - not yet sufficient on its own.
#
# ROUND 3 (target sweep, stop fixed at 2.5x): found NO new candidate -
# 1.0x target (already in use) turned out to be the exact local peak of a
# clean, unimodal PF curve (0.75x->1.102, 1.0x->1.202 peak, 1.25x->0.990).
# Fixed-number R:R tuning is now exhausted on both axes without finding
# anything beyond what round 2's stress test already showed insufficient.
#
# ROUND 4 (this sweep): entry-side tuning, R:R FIXED at stop=2.5x/
# target=1.0x (the best found so far). Bollinger band width first - the
# PRIMARY "how far is unusually far" trigger, more central to the core
# idea than the RSI/trend-avoidance confirmations. One free parameter
# (bollinger_std_multiple), RSI and trend-efficiency held at their
# original defaults. No strong directional prior - testing both looser
# and stricter than the current 2.0 standard-deviation default.
# =============================================================================

STOP_ATR_MULTIPLE = 2.5
TARGET_ATR_MULTIPLE = 1.0
BOLLINGER_STD_MULTIPLES_TO_TEST = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

EXPERIMENTS = [
    (f"bollinger {b} std" + (" (default)" if b == 2.0 else ""),
     MeanReversionConfig(bollinger_std_multiple=b),
     RiskConfig(stop_loss_atr_multiple=STOP_ATR_MULTIPLE, take_profit_atr_multiple=TARGET_ATR_MULTIPLE))
    for b in BOLLINGER_STD_MULTIPLES_TO_TEST
]


def build_all_frames(raw_data, mr_config=MeanReversionConfig()):
    frames = {}
    for symbol in PORTFOLIO_SYMBOLS:
        h1 = raw_data[symbol]["H1"]
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h1, h4, config=mr_config)
    return frames


def run_experiment(label, mr_config, risk_config, raw_data):
    frames = build_all_frames(raw_data, mr_config)
    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=risk_config)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=risk_config)
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
    print("MEAN-REVERSION STRATEGY - expect a HIGH win rate with small average")
    print("wins. A win rate near 33% here is a red flag, not a normal baseline")
    print("(that was the trend-following strategy's expected shape, not this one's).")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS)
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
