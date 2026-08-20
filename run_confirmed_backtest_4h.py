"""
run_confirmed_backtest_4h.py
--------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the DUAL-CONFIRMATION 4H strategy (signals_confirmed_4h.py)
- trend-following (signals_4h.py) AND time-series momentum
(signals_momentum_4h.py) must both agree on direction before entering.
Reuses the same backtest engine, risk management, safety limits, data
pipeline, and metrics/reporting helpers as every other run_*.py script -
this combination needed no new engine work at all (unlike
combined_signals_4h.py's portfolio-sharing case), since it's just two
existing boolean signal columns ANDed together on the same instrument.

Same RiskConfig/engine adjustments as every other 4H entry point:
stale_data_max_gap_hours widened to 6, bar_duration_hours=4.

=============================================================================
Experiments - each is (label, MomentumConfig) - trend side (EMA 50/200,
20-bar channel) and RiskConfig (1.5x/3.0x ATR) held fixed throughout;
momentum's lookback is this round's one free parameter.

ROUND 1 (momentum lookback=20, matching the breakout channel's own
window) result: BYTE-FOR-BYTE IDENTICAL to the unfiltered trend-
following baseline - same trade count, same win rate, same P&L to the
cent. Root cause, proven algebraically not just observed: with matching
windows, Close[t] > max(High over the prior 20 bars) MATHEMATICALLY
GUARANTEES Close[t] > Close[t-20] (every price in that window, including
the one 20 bars back, is bounded by that same max), so momentum
agreement is a tautology given the breakout condition - it filters out
zero trades. This holds for ANY momentum lookback <= the channel period,
not just 20 - the same proof applies verbatim. Only a momentum lookback
STRICTLY GREATER than the channel period (20) asks a genuinely
independent question ("does the bigger-picture trend agree", not
implied by a recent-20-bar breakout).

ROUND 2 (this sweep): momentum lookback swept both as a control (10 -
expected to reproduce round 1's identical numbers again, confirming the
tautology proof empirically) and across values that break the
tautology (25 and up).
=============================================================================

Usage:
    source venv/bin/activate
    python run_confirmed_backtest_4h.py
"""

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from data_fetch import fetch_all
from signals_confirmed_4h import prepare_instrument_frame
from signals_4h import Signal4HConfig
from signals_momentum_4h import MomentumConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict, print_comparison_table,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

BAR_DURATION_HOURS = 4
TREND_CONFIG = Signal4HConfig()  # EMA 50/200, 20-bar channel - fixed throughout
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=6)  # spec-literal 1.5x/3.0x ATR stop/target -
                                                        # this is still fundamentally a trend-following
                                                        # entry, just filtered, so it keeps that
                                                        # strategy's own R:R rather than momentum's

MOMENTUM_LOOKBACKS_TO_TEST = [10, 25, 30, 40, 50, 75, 100, 150]

EXPERIMENTS = [
    (f"momentum lookback {lb} bars" + (" (control - expect identical to unfiltered baseline)" if lb <= 20 else ""),
     MomentumConfig(lookback_period=lb))
    for lb in MOMENTUM_LOOKBACKS_TO_TEST
]


def build_all_frames(raw_data, momentum_config):
    frames = {}
    for symbol in PORTFOLIO_SYMBOLS:
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h4, trend_config=TREND_CONFIG, momentum_config=momentum_config)
    return frames


def run_experiment(label, momentum_config, raw_data):
    frames = build_all_frames(raw_data, momentum_config)
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
        "label": label, "signal_config": momentum_config, "risk_config": RISK_CONFIG,
        "train_result": train_result, "test_result": test_result,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "checks": checks,
        "split_info": (common_start, split_point, common_end),
    }


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("DUAL-CONFIRMATION 4H STRATEGY - trend-following AND time-series")
    print("momentum must both agree before entering. See signals_confirmed_4h.py.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS, granularities=("H4",))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric below is MEANINGLESS as a "
            "strategy evaluation on this data. ***\n"
        )

    experiment_results = []
    for i, (label, momentum_config) in enumerate(EXPERIMENTS):
        print(f"\n\n{'#' * 78}")
        print(f"# EXPERIMENT: {label}")
        print(f"{'#' * 78}")

        result = run_experiment(label, momentum_config, raw_data)
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
