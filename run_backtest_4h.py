"""
run_backtest_4h.py
---------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the SINGLE-TIMEFRAME 4H trend-following strategy
(signals_4h.py) - parallel to run_backtest.py (the original 4H-trend/1H-
entry strategy) and run_mean_reversion_backtest.py, reusing the same
backtest engine, risk management, safety limits, data pipeline, and
metrics/reporting helpers. Only signal generation and timeframe differ.

Only 4H data is fetched here - this strategy has no use for 1H candles
at all, unlike signals.py.

Two RiskConfig adjustments are required for correctness at 4H
granularity (not just parameter re-tuning - see signals_4h.py and
backtest_engine.py's bar_duration_hours docs for why):
  - stale_data_max_gap_hours widened from the 1H default of 3 to 6 - the
    NORMAL gap between 4H bars is 4 hours, which already exceeds the old
    3-hour "stale" threshold and would incorrectly flag every bar.
  - bar_duration_hours=4 passed to run_backtest() so the overnight
    financing rollover check (hour 21) correctly detects which 4H bar
    covers it, instead of never firing (4H bars land on hours 0/4/8/12/
    16/20, never exactly on 21).

Usage:
    source venv/bin/activate
    python run_backtest_4h.py
"""

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from data_fetch import fetch_all
from signals_4h import prepare_instrument_frame, Signal4HConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict, print_comparison_table,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

# =============================================================================
# Experiments - each is (label, Signal4HConfig, RiskConfig).
#
# ROUND 1 (spec defaults: 1.5x/3x ATR stop/target) result: 0/11 stress-
# test scenarios passed, but with a genuinely different, more informative
# shape than any prior config - avg test PF 0.997 (closer to breakeven
# than anything else tried), max test drawdown never exceeded 8.02% in
# any scenario (vs 20-28% for the 1H strategies), and avg win rate 33.9%
# sat almost exactly on the ~33.3% breakeven this 2:1 R:R requires. Read:
# the structural fix removed the wild regime-dependent swings, but there
# isn't yet a real edge on top of that stability.
#
# ROUND 2 (stop sweep, target fixed at 3.0x) result: a clear directional
# signal - TIGHTER stops do better here (1.0x -> PF 1.273, 1.25x -> PF
# 1.149, 1.5x baseline -> PF 1.004, worse beyond that). Opposite of the
# mean-reversion strategy's finding, which makes sense: a wrong breakout
# should be cut fast, not given room to "come back" the way a reversion
# trade needs room to revert. BUT a real tension: 1.0x has the best PF
# but only 116 combined trades (fails the 150 minimum); 1.25x clears
# trade count (306) but PF is only 1.149, just short of 1.2.
#
# ROUND 3 (fine stop sweep, 1.0-1.3x, target fixed at 3.0x) result: noisy,
# non-monotonic (1.0x->1.273, 1.05x->1.091, 1.1x->0.937, 1.15x->0.938,
# 1.2x->1.184, 1.25x->1.149, 1.3x->1.071) - no clean winner, nothing
# clearing both the trade-count and profit-factor bars at once. Not
# pursued further at the time.
#
# ROUND 4 (this sweep): a different free parameter - reward:risk itself,
# via the TARGET multiple, stop held at the spec-literal 1.5x (not the
# round-2/3 tuned stop values, which were never themselves validated
# beyond an unstressed single look - starting the new free parameter from
# the last VALIDATED baseline, not an unvalidated one). Motivated by every
# variant of this strategy landing win rates around 26-39%, while a 2:1
# R:R needs ~33%+ to break even - testing whether a less greedy target
# lets the win rate actually being achieved clear the profit-factor bar,
# rather than assuming a higher win rate would follow from a closer
# target (it may not - the 4H mean-reversion sweep found exactly that:
# win rate rose with a wider stop, but profit factor didn't follow
# proportionally). R:R swept from the 2:1 baseline down to 1:1.
# =============================================================================

BAR_DURATION_HOURS = 4

STOP_ATR_MULTIPLE = 1.5  # spec-literal, fixed - target/R:R is this round's one free parameter
RR_RATIOS_TO_TEST = [2.0, 1.75, 1.5, 1.25, 1.0]

EXPERIMENTS = [
    (f"R:R {rr}:1 (stop {STOP_ATR_MULTIPLE}x / target {STOP_ATR_MULTIPLE * rr:.3f}x ATR)"
     + (" (round 1 baseline)" if rr == 2.0 else ""),
     Signal4HConfig(),
     RiskConfig(stale_data_max_gap_hours=6, stop_loss_atr_multiple=STOP_ATR_MULTIPLE,
                take_profit_atr_multiple=STOP_ATR_MULTIPLE * rr))
    for rr in RR_RATIOS_TO_TEST
]


def build_all_frames(raw_data, config=Signal4HConfig()):
    frames = {}
    for symbol in PORTFOLIO_SYMBOLS:
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h4, config=config)
    return frames


def run_experiment(label, signal_config, risk_config, raw_data):
    frames = build_all_frames(raw_data, signal_config)
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
        "label": label, "signal_config": signal_config, "risk_config": risk_config,
        "train_result": train_result, "test_result": test_result,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "checks": checks,
        "split_info": (common_start, split_point, common_end),
    }


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("SINGLE-TIMEFRAME 4H STRATEGY - trend filter and entry trigger both")
    print("computed from the same 4H series (see signals_4h.py).")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS, granularities=("H4",))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric below is MEANINGLESS as a "
            "strategy evaluation on this data. ***\n"
        )

    experiment_results = []
    for i, (label, signal_config, risk_config) in enumerate(EXPERIMENTS):
        print(f"\n\n{'#' * 78}")
        print(f"# EXPERIMENT: {label}")
        print(f"{'#' * 78}")

        result = run_experiment(label, signal_config, risk_config, raw_data)
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
