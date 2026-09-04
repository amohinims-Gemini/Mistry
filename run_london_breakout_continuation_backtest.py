"""
run_london_breakout_continuation_backtest.py
--------------------------------------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
Entry point for the Asian/London Range Breakout - Continuation strategy
(signals_london_breakout_continuation_m15.py, "candidate 2" of the
post-Hypothesis-4 structural search) - M15, EUR_USD/GBP_USD only,
Europe/London session timing. Reuses the same backtest engine, risk
management, safety limits, and OANDA data pipeline as every other
strategy in this project - see the signal module's docstring for what's
genuinely new here (continuation entry logic, opposite-boundary
structural stop) vs. what's unchanged (Asian-range infrastructure,
position sizing, safety limits, GBP-account handling, the engine
itself).

Unlike Hypothesis 4's breakout-trigger simulation, this strategy's
signal-on-Close/fill-at-next-Open timing matches backtest_engine.py's
standard, unmodified convention exactly - so it runs through the SAME
run_backtest() every other strategy in this project uses, not a
standalone simulator.

DEVELOPMENT DATA ONLY, per standing project discipline: uses
dataset_split.split_for_iteration() exclusively - the validation and
final reserved periods are not reachable through this script at all.
Within development, the usual (unchanged) train/test split from
run_backtest.py is applied for this round's own iterative work.

*** THIS SCRIPT HAS NOT BEEN RUN YET, PER STANDING PROJECT DISCIPLINE. ***
Built and tested (see tests/test_london_breakout_continuation_signals.py),
but the actual backtest has not been executed against real data. Do not
run it without explicit approval.

Usage (once approved):
    source venv/bin/activate
    python run_london_breakout_continuation_backtest.py
"""

from instruments import INSTRUMENTS
from data_fetch import get_instrument_data
from signals_london_breakout_continuation_m15 import (
    prepare_instrument_frame, LondonBreakoutContinuationConfig,
)
from backtest_engine import run_backtest
from risk_management import RiskConfig
from dataset_split import split_for_iteration, describe_split
from run_backtest import (
    split_train_test, compute_metrics, evaluate_requirements,
    print_period_report, print_final_verdict,
    STARTING_CASH, COMMISSION_PER_TRADE,
)

# Deliberately NOT added to the shared PORTFOLIO_SYMBOLS - same reasoning
# as V1/V2's own standalone symbol lists: a new instrument/strategy scope
# shouldn't silently widen what every other multi-instrument script trades.
BREAKOUT_CONTINUATION_SYMBOLS = ["EUR_USD", "GBP_USD"]

BAR_DURATION_HOURS = 0.25  # M15 - so the overnight financing rollover-hour check
                           # correctly identifies which 15-minute bar covers hour 21 UTC.
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=0.5)  # 30 min - same convention as V1/V2.
                                                          # stop_loss_atr_multiple/take_profit_atr_multiple
                                                          # are UNUSED by this strategy - every trade's
                                                          # stop/target comes from the structural
                                                          # stop_distance_override/target_distance_override
                                                          # computed in signals_london_breakout_continuation_m15.py.

SIGNAL_CONFIG = LondonBreakoutContinuationConfig()


def fetch_raw_data():
    raw = {}
    for symbol in BREAKOUT_CONTINUATION_SYMBOLS:
        oanda_symbol = INSTRUMENTS[symbol].oanda_symbol
        raw[symbol] = {"M15": get_instrument_data(oanda_symbol, "M15")}
    return raw


def build_all_frames(raw_data, config=SIGNAL_CONFIG):
    frames = {}
    for symbol in BREAKOUT_CONTINUATION_SYMBOLS:
        frames[symbol] = prepare_instrument_frame(raw_data[symbol]["M15"], config=config)
    return frames


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("ASIAN/LONDON RANGE BREAKOUT - CONTINUATION - M15, EUR_USD/GBP_USD only.")
    print("Europe/London session timing (DST-aware). See")
    print("signals_london_breakout_continuation_m15.py.")
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
    # _validation_frames_not_used_this_round is intentionally never
    # touched - this round only looks at development data, per standing
    # project discipline. Named this way (not `_`) so that's obvious on
    # read, not just in a comment.

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
