"""
run_backtest_daily.py
-------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the SINGLE-TIMEFRAME DAILY trend-following strategy
(signals_daily.py) - the daily-timeframe fallback, parallel to
run_backtest_4h.py. Only 4H/1H results failed to clear the validation
bars (individually and combined) - see signals_daily.py's docstring for
why daily parameters were chosen the way they were.

RiskConfig adjustments required for correctness at daily granularity:
  - stale_data_max_gap_hours widened to 30 (normal daily gap is 24h).
  - bar_duration_hours=24 passed to run_backtest() - daily candles land
    exactly on the 21:00 UTC rollover hour, so financing applies once
    per bar as expected.

KNOWN LIMITATION carried over from econ_calendar.py: the recurring
12:00-14:00 UTC blackout-window heuristic becomes a NO-OP at daily
granularity - daily candle timestamps are fixed at 21:00 UTC, which
never falls inside that window. Making it meaningfully apply at daily
resolution would need a different design (blocking whole announcement
DAYS, not a timestamp match) - not implemented; documented here rather
than silently left unexplained.

Usage:
    source venv/bin/activate
    python run_backtest_daily.py
"""

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from data_fetch import fetch_all
from signals_daily import prepare_instrument_frame, SignalDailyConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

BAR_DURATION_HOURS = 24
SIGNAL_CONFIG = SignalDailyConfig()
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=30)  # spec-literal 1.5x/3.0x ATR stop/target


def build_all_frames(raw_data, config=SIGNAL_CONFIG):
    frames = {}
    for symbol in PORTFOLIO_SYMBOLS:
        d1 = raw_data[symbol]["D"]
        frames[symbol] = prepare_instrument_frame(d1, config=config)
    return frames


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("SINGLE-TIMEFRAME DAILY STRATEGY - trend filter and entry trigger both")
    print("computed from the same daily series (see signals_daily.py).")
    print("NOTE: the econ-calendar blackout filter is a documented no-op at this")
    print("granularity - see this file's docstring.")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS, granularities=("D",))
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
