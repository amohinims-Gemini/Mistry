"""
run_backtest.py
------------------
*** BACKTEST ONLY - THIS SCRIPT DOES NOT PLACE REAL TRADES. ***
There is no live broker connection and no order-placement, trade, or
account-modification endpoint is ever called anywhere in this project.
The only OANDA API usage is a read-only historical-candles fetch (see
data_fetch.py). This script only replays historical data through
backtest_engine.py's simulator to see how the strategy *would have*
performed - nothing here buys or sells anything for real.

What this script does:
  1. Fetches (or loads from cache) several years of 1H + 4H candle data
     for EUR/USD, GBP/USD, USD/JPY, and Gold from OANDA.
  2. Builds each instrument's signal frame (signals.py): the 4H trend
     filter, the 1H breakout trigger, ATR, and spread.
  3. Splits the data CHRONOLOGICALLY at one shared cutoff date (not per
     instrument) into a 70% TRAINING period and a 30% TESTING period the
     strategy has never "seen" - so the whole portfolio is evaluated
     out-of-sample together, the way it would actually be traded.
  4. Runs the full multi-instrument backtest engine on each period
     separately, each starting from a fresh $10,000 account.
  5. Prints a full metrics report for both periods - including a
     breakdown of every reason a signal was rejected (spread, news
     blackout, safety limits, etc.) for transparency - and states
     PLAINLY whether the strategy clears every bar from the spec:
     >=150 completed trades, positive out-of-sample return, <10% max
     drawdown, profit factor >1.2. No bar is loosened to force a pass.

Usage:
    source venv/bin/activate
    python run_backtest.py
"""

import pandas as pd

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from signals import prepare_instrument_frame
from backtest_engine import run_backtest

STARTING_CASH = 10_000
COMMISSION_PER_TRADE = 0.0   # OANDA's standard retail accounts are spread-only, no separate
                             # commission - the real transaction cost here is the bid/ask
                             # spread itself, already modeled via realistic bid/ask fills.
TRAIN_FRACTION = 0.70

MIN_REQUIRED_TRADES = 150
MAX_ALLOWED_DRAWDOWN_PCT = 10.0
MIN_REQUIRED_PROFIT_FACTOR = 1.2


def build_all_frames():
    """Fetch/cache data for every instrument and build its signal frame."""
    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()))

    frames = {}
    for symbol, spec in INSTRUMENTS.items():
        h1 = raw_data[symbol]["H1"]
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h1, h4)

    return frames, is_synthetic


def split_train_test(frames, train_fraction=TRAIN_FRACTION):
    """Split every instrument's frame at ONE SHARED chronological cutoff
    (not a per-instrument fraction), so the whole portfolio is tested
    out-of-sample together - the way it would actually be traded."""
    common_start = max(df.index.min() for df in frames.values())
    common_end = min(df.index.max() for df in frames.values())
    split_point = common_start + (common_end - common_start) * train_fraction

    train_frames, test_frames = {}, {}
    for symbol, df in frames.items():
        df = df[(df.index >= common_start) & (df.index <= common_end)]
        train_frames[symbol] = df[df.index < split_point]
        test_frames[symbol] = df[df.index >= split_point]

    return train_frames, test_frames, common_start, split_point, common_end


def compute_metrics(result, starting_cash):
    """Performance metrics for one backtest run. 'Completed trades' means
    trades that hit their stop-loss or take-profit naturally - forced
    closes at the end of the data (the backtest simply running out of
    history) are tracked separately and NOT counted toward the 150-trade
    requirement or win-rate/profit-factor, since they're an artifact of
    where the data ends, not a real strategy outcome."""
    trades = result["trades"]
    if len(trades) == 0:
        real_trades = trades
    else:
        real_trades = trades[trades["exit_reason"].isin(["stop_loss", "take_profit"])]

    n_completed = len(real_trades)
    n_forced = len(trades) - n_completed

    wins = real_trades[real_trades["pnl"] > 0] if n_completed else real_trades
    losses = real_trades[real_trades["pnl"] <= 0] if n_completed else real_trades
    win_rate_pct = (len(wins) / n_completed * 100) if n_completed else float("nan")

    gross_profit = wins["pnl"].sum() if len(wins) else 0.0
    gross_loss = -losses["pnl"].sum() if len(losses) else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else float("nan")

    final_balance = result["account"].balance
    total_return_pct = (final_balance - starting_cash) / starting_cash * 100

    equity = result["equity_curve"]["equity"] if len(result["equity_curve"]) else pd.Series(dtype=float)
    if len(equity):
        running_peak = equity.cummax()
        drawdown_pct = ((running_peak - equity) / running_peak * 100)
        max_drawdown_pct = drawdown_pct.max()
    else:
        max_drawdown_pct = 0.0

    return {
        "n_completed_trades": n_completed,
        "n_forced_closes": n_forced,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_return_pct": total_return_pct,
        "final_balance": final_balance,
        "max_drawdown_pct": max_drawdown_pct,
        "total_financing_paid": result["account"].total_financing_paid,
    }


def print_period_report(label, result, metrics):
    print(f"\n===== {label} =====")
    print(f"Completed trades (SL/TP only): {metrics['n_completed_trades']}  "
          f"(+ {metrics['n_forced_closes']} forced closes at period end, excluded from stats)")
    print(f"Win rate:            {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.3f}  "
          f"(gross profit ${metrics['gross_profit']:.2f} / gross loss ${metrics['gross_loss']:.2f})")
    print(f"Total return:        {metrics['total_return_pct']:.2f}%")
    print(f"Final balance:       ${metrics['final_balance']:.2f}")
    print(f"Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Total financing paid (swap): ${metrics['total_financing_paid']:.2f}")

    trades = result["trades"]
    if len(trades):
        print("\nPer-instrument breakdown:")
        real = trades[trades["exit_reason"].isin(["stop_loss", "take_profit"])]
        for symbol, group in real.groupby("symbol"):
            wins = (group["pnl"] > 0).sum()
            print(f"  {symbol}: {len(group)} trades, {wins}/{len(group)} won "
                  f"({wins/len(group)*100:.1f}%), P&L ${group['pnl'].sum():.2f}")

    rejections = result["rejections"]
    if len(rejections):
        print("\nSignals rejected (safety limits / filters doing their job):")
        for reason, count in rejections["reason"].value_counts().items():
            print(f"  {reason}: {count}")


def print_final_verdict(train_metrics, test_metrics):
    total_completed = train_metrics["n_completed_trades"] + test_metrics["n_completed_trades"]

    checks = [
        (
            f"At least {MIN_REQUIRED_TRADES} completed trades (train + test combined)",
            total_completed >= MIN_REQUIRED_TRADES,
            f"{total_completed} completed",
        ),
        (
            "Positive result on unseen out-of-sample (test) data",
            test_metrics["total_return_pct"] > 0,
            f"test return {test_metrics['total_return_pct']:.2f}%",
        ),
        (
            f"Max drawdown stays below {MAX_ALLOWED_DRAWDOWN_PCT:.0f}% (both periods)",
            train_metrics["max_drawdown_pct"] < MAX_ALLOWED_DRAWDOWN_PCT
            and test_metrics["max_drawdown_pct"] < MAX_ALLOWED_DRAWDOWN_PCT,
            f"train {train_metrics['max_drawdown_pct']:.2f}%, test {test_metrics['max_drawdown_pct']:.2f}%",
        ),
        (
            f"Profit factor above {MIN_REQUIRED_PROFIT_FACTOR} on out-of-sample (test) data",
            test_metrics["profit_factor"] > MIN_REQUIRED_PROFIT_FACTOR,
            f"test profit factor {test_metrics['profit_factor']:.3f}",
        ),
    ]

    print("\n" + "=" * 78)
    print("FINAL VERDICT - does this strategy meet every bar from the spec?")
    print("=" * 78)
    all_passed = True
    for description, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {description}  ->  {detail}")

    print("-" * 78)
    if all_passed:
        print("Overall: PASSES every requirement in the spec on this dataset.")
        print("That is NOT the same as a guarantee of future performance - it means")
        print("the strategy cleared the bars you set, on the history available.")
    else:
        print("Overall: DOES NOT pass every requirement. Reporting this honestly")
        print("rather than loosening any rule to force a pass, as agreed.")
    print("=" * 78)


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("The only OANDA API call anywhere in this project is a read-only")
    print("historical-candles fetch (data_fetch.py).")
    print("=" * 78)

    frames, is_synthetic = build_all_frames()
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric and PASS/FAIL check below is "
            "MEANINGLESS as a strategy evaluation on this data - it only proves the code "
            "runs end to end. Fix your OANDA credentials in .env and re-run for a real "
            "evaluation. ***\n"
        )

    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)
    print(f"\nFull available data range: {common_start} to {common_end}")
    print(f"Training period: {common_start} to {split_point}")
    print(f"Testing period:  {split_point} to {common_end}")

    print("\nRunning backtest on TRAINING period...")
    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH, commission_per_trade=COMMISSION_PER_TRADE)
    train_metrics = compute_metrics(train_result, STARTING_CASH)
    print_period_report("TRAINING", train_result, train_metrics)

    print("\nRunning backtest on TESTING period (unseen data)...")
    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH, commission_per_trade=COMMISSION_PER_TRADE)
    test_metrics = compute_metrics(test_result, STARTING_CASH)
    print_period_report("TESTING (unseen data)", test_result, test_metrics)

    print_final_verdict(train_metrics, test_metrics)

    print("\nKnown approximations in this backtest (see file docstrings for detail):")
    print("  - Economic calendar: a recurring weekday time-window heuristic, not a real")
    print("    calendar - misses FOMC/ECB/BoE decisions and one-off events (econ_calendar.py)")
    print("  - Swap/financing rates: static annual approximations, not real historical")
    print("    OANDA rates, which move with central bank policy (instruments.py)")
    print("  - Rollover hour fixed at 21:00 UTC year-round (doesn't track US DST exactly)")


if __name__ == "__main__":
    main()
