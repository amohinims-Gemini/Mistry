"""
run_combined_backtest_4h.py
-------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the COMBINED 4H portfolio: trend-following (signals_4h.py)
and mean-reversion (mean_reversion_signals_4h.py) trading out of ONE
shared PortfolioAccount, sharing every existing risk/safety limit (max
open positions, total open risk cap, correlation groups, daily/weekly/
drawdown circuit breakers) - see combined_signals_4h.py for the exact
combination rule.

Motivation: neither 4H strategy alone clears the 4 validation bars (see
run_backtest_4h.py / run_mean_reversion_backtest_4h.py) - both sit near
breakeven - but both show far tighter, more consistent drawdown control
than any 1H config tried, and a direct check of their historical trades
found the two are essentially uncorrelated (-0.02 daily P&L correlation,
~2.7% same-instrument-same-day trade overlap) - genuinely different
edges, not duplicates of the same one. Worth testing whether the
combination clears the bars in aggregate even though neither does alone.

R:R for each side of the combination uses each strategy's own literal
ROUND-1 baseline (trend 1.5x/3.0x, mean-reversion 1.5x/1.0x), not any
sweep result - neither strategy's sweep found a configuration that
itself passed even the single train/test split, so building on top of
an unvalidated sweep result here would just be compounding an unproven
number. This can be revisited if the combination itself looks promising
enough to be worth its own R:R sweep later.

Usage:
    source venv/bin/activate
    python run_combined_backtest_4h.py
"""

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from combined_signals_4h import build_combined_frame
from signals_4h import Signal4HConfig
from mean_reversion_signals_4h import MeanReversion4HConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

BAR_DURATION_HOURS = 4
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=6)  # required for correctness at 4H, see run_backtest_4h.py

TREND_STOP_ATR_MULTIPLE = 1.5
TREND_TARGET_ATR_MULTIPLE = 3.0
MR_STOP_ATR_MULTIPLE = 1.5
MR_TARGET_ATR_MULTIPLE = 1.0


def build_all_frames(raw_data):
    frames = {}
    for symbol in INSTRUMENTS:
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = build_combined_frame(
            h4,
            trend_config=Signal4HConfig(),
            trend_stop_atr_multiple=TREND_STOP_ATR_MULTIPLE,
            trend_target_atr_multiple=TREND_TARGET_ATR_MULTIPLE,
            mr_config=MeanReversion4HConfig(),
            mr_stop_atr_multiple=MR_STOP_ATR_MULTIPLE,
            mr_target_atr_multiple=MR_TARGET_ATR_MULTIPLE,
        )
    return frames


def run_once(train_frames, test_frames):
    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                 bar_duration_hours=BAR_DURATION_HOURS)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=RISK_CONFIG,
                                bar_duration_hours=BAR_DURATION_HOURS)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)
    return train_result, test_result, train_metrics, test_metrics, checks


def print_source_breakdown(label, result):
    """Break trade count/win-rate/P&L down by which strategy generated
    the entry - lets us see whether one side of the combination is
    carrying the other, and roughly how often the rare 'both agreed'
    case actually happened."""
    trades = result["trades"]
    if len(trades) == 0 or "signal_source" not in trades.columns:
        return
    real = trades[trades["exit_reason"].isin(["stop_loss", "take_profit"])]
    if len(real) == 0:
        return
    print(f"\n{label} - trade attribution by signal source:")
    for source, group in real.groupby("signal_source"):
        n = len(group)
        wins = (group["pnl"] > 0).sum()
        wr = 100 * wins / n if n else float("nan")
        print(f"  {source:15s}: {n:4d} trades, win rate {wr:5.1f}%, total P&L ${group['pnl'].sum():10.2f}")


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("COMBINED 4H PORTFOLIO - trend-following + mean-reversion trading out of")
    print("ONE shared account, sharing every existing risk/safety limit.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()), granularities=("H4",))
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

    train_result, test_result, train_metrics, test_metrics, checks = run_once(train_frames, test_frames)

    print_period_report("TRAINING", train_result, train_metrics)
    print_source_breakdown("TRAINING", train_result)
    print_period_report("TESTING (unseen data)", test_result, test_metrics)
    print_source_breakdown("TESTING (unseen data)", test_result)
    print_final_verdict(checks)


if __name__ == "__main__":
    main()
