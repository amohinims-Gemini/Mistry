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
  2. For each experiment in EXPERIMENTS below, builds each instrument's
     signal frame (signals.py) using that experiment's SignalConfig -
     entry logic itself can vary between experiments, not just risk
     parameters - then splits CHRONOLOGICALLY at one shared cutoff date
     into a 70% TRAINING period and a 30% TESTING period.
  3. Runs the full multi-instrument backtest engine with that
     experiment's RiskConfig, each period starting from a fresh $10,000
     account.
  4. Prints a detailed report for the baseline, a short summary for each
     variation, and a comparison table across all of them at the end -
     so tradeoffs are visible side by side rather than picking a winner
     silently.

Usage:
    source venv/bin/activate
    python run_backtest.py
"""

import pandas as pd

from instruments import INSTRUMENTS
from data_fetch import fetch_all
from signals import prepare_instrument_frame, SignalConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig

STARTING_CASH = 10_000
COMMISSION_PER_TRADE = 0.0   # OANDA's standard retail accounts are spread-only, no separate
                             # commission - the real transaction cost here is the bid/ask
                             # spread itself, already modeled via realistic bid/ask fills.
TRAIN_FRACTION = 0.70

MIN_REQUIRED_TRADES = 150
MAX_ALLOWED_DRAWDOWN_PCT = 10.0
MIN_REQUIRED_PROFIT_FACTOR = 1.2


# =============================================================================
# Experiments - each is (label, SignalConfig, RiskConfig).
#
# STATUS: the structural redesign is PAUSED - see README.md for the full
# history. Two of three ideas tried (channel-length retuning, stop-to-
# breakeven trade management) failed full 11-scenario stress-testing after
# looking promising on a single split or one diagnosed episode. One
# (an efficiency-ratio entry filter) showed a genuine, if partial,
# improvement (pass rate 18%->36% across the stress test) but was never
# adopted as a default pending further work. Current defaults below are
# the plain, original spec entry/risk logic - no optional filter active.
#
# To resume exploring, add more (label, SignalConfig, RiskConfig) tuples
# here - the comparison-table machinery handles any number of them.
# =============================================================================

EXPERIMENTS = [
    ("current defaults (spec-original entry logic)", SignalConfig(), RiskConfig()),
]


def build_all_frames(raw_data, signal_config=SignalConfig()):
    """Build each instrument's signal frame from already-fetched raw
    OANDA data, using the given SignalConfig."""
    frames = {}
    for symbol in INSTRUMENTS:
        h1 = raw_data[symbol]["H1"]
        h4 = raw_data[symbol]["H4"]
        frames[symbol] = prepare_instrument_frame(h1, h4, config=signal_config)
    return frames


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
    """
    Performance metrics for one backtest run. 'Completed trades' means
    trades that hit their stop-loss or take-profit naturally - forced
    closes at the end of the data are tracked separately and NOT counted
    toward the 150-trade requirement or win-rate/profit-factor.

    FINANCING IS EXCLUDED from total_return_pct and max_drawdown_pct (the
    two metrics checked against the spec's pass/fail bars) - confirmed in
    testing that once leverage was fixed to allow realistic position
    sizes, financing under this project's static/approximate swap-rate
    model could reach ~30x the strategy's actual trading P&L. Scoring the
    strategy on numbers that are mostly a financing assumption rather
    than trading performance would be evaluating the wrong thing.
    profit_factor/win_rate were ALREADY financing-free (trade pnl never
    included it) and are unaffected by this. The financing-inclusive
    numbers are still fully computed and returned below, just not used
    for pass/fail - see print_period_report, which shows both.
    """
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

    account = result["account"]
    final_balance = account.balance
    final_trading_only_balance = account.trading_only_balance
    total_return_pct = (final_trading_only_balance - starting_cash) / starting_cash * 100
    total_return_pct_incl_financing = (final_balance - starting_cash) / starting_cash * 100

    def _max_drawdown_pct(series):
        if not len(series):
            return 0.0
        running_peak = series.cummax()
        return ((running_peak - series) / running_peak * 100).max()

    ec = result["equity_curve"]
    trading_only_equity = ec["trading_only_equity"] if len(ec) else pd.Series(dtype=float)
    real_equity = ec["equity"] if len(ec) else pd.Series(dtype=float)
    max_drawdown_pct = _max_drawdown_pct(trading_only_equity)
    max_drawdown_pct_incl_financing = _max_drawdown_pct(real_equity)

    return {
        "n_completed_trades": n_completed,
        "n_forced_closes": n_forced,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        # --- Scored metrics (financing excluded) ---
        "total_return_pct": total_return_pct,
        "final_balance": final_trading_only_balance,
        "max_drawdown_pct": max_drawdown_pct,
        # --- Reported separately, NOT used for pass/fail ---
        "total_return_pct_incl_financing": total_return_pct_incl_financing,
        "final_balance_incl_financing": final_balance,
        "max_drawdown_pct_incl_financing": max_drawdown_pct_incl_financing,
        "total_financing_paid": account.total_financing_paid,
    }


def evaluate_requirements(train_metrics, test_metrics):
    total_completed = train_metrics["n_completed_trades"] + test_metrics["n_completed_trades"]
    checks = {
        "trades_pass": total_completed >= MIN_REQUIRED_TRADES,
        "oos_positive_pass": test_metrics["total_return_pct"] > 0,
        "drawdown_pass": (
            train_metrics["max_drawdown_pct"] < MAX_ALLOWED_DRAWDOWN_PCT
            and test_metrics["max_drawdown_pct"] < MAX_ALLOWED_DRAWDOWN_PCT
        ),
        "profit_factor_pass": test_metrics["profit_factor"] > MIN_REQUIRED_PROFIT_FACTOR,
    }
    checks["total_completed"] = total_completed
    checks["all_pass"] = all(
        v for k, v in checks.items() if k.endswith("_pass")
    )
    return checks


def print_period_report(label, result, metrics):
    print(f"\n===== {label} =====")
    print(f"Completed trades (SL/TP only): {metrics['n_completed_trades']}  "
          f"(+ {metrics['n_forced_closes']} forced closes at period end, excluded from stats)")
    print(f"Win rate:            {metrics['win_rate_pct']:.2f}%")
    print(f"Profit factor:       {metrics['profit_factor']:.3f}  "
          f"(gross profit ${metrics['gross_profit']:.2f} / gross loss ${metrics['gross_loss']:.2f})")
    print(f"Total return (trading only, SCORED):   {metrics['total_return_pct']:.2f}%")
    print(f"Final balance (trading only, SCORED):  ${metrics['final_balance']:.2f}")
    print(f"Max drawdown (trading only, SCORED):   {metrics['max_drawdown_pct']:.2f}%")
    print(f"--- financing, reported separately, NOT scored (static/approximate rate) ---")
    print(f"Total financing paid (swap):            ${metrics['total_financing_paid']:.2f}")
    print(f"Total return INCLUDING financing:       {metrics['total_return_pct_incl_financing']:.2f}%")
    print(f"Final balance INCLUDING financing:      ${metrics['final_balance_incl_financing']:.2f}")
    print(f"Max drawdown INCLUDING financing:        {metrics['max_drawdown_pct_incl_financing']:.2f}%")

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


def print_final_verdict(checks):
    print("\n" + "=" * 78)
    print("Does this configuration meet every bar from the spec?")
    print("=" * 78)
    labels = {
        "trades_pass": f"At least {MIN_REQUIRED_TRADES} completed trades (train + test combined)",
        "oos_positive_pass": "Positive result on unseen out-of-sample (test) data",
        "drawdown_pass": f"Max drawdown stays below {MAX_ALLOWED_DRAWDOWN_PCT:.0f}% (both periods)",
        "profit_factor_pass": f"Profit factor above {MIN_REQUIRED_PROFIT_FACTOR} on out-of-sample (test) data",
    }
    for key, description in labels.items():
        status = "PASS" if checks[key] else "FAIL"
        print(f"[{status}] {description}")
    print("-" * 78)
    if checks["all_pass"]:
        print("Overall: PASSES every requirement in the spec on this dataset.")
    else:
        print("Overall: DOES NOT pass every requirement.")
    print("=" * 78)


def run_experiment(label, signal_config, risk_config, raw_data):
    """Build this experiment's signal frames (entry logic can differ per
    experiment), split train/test, and run the backtest with this
    experiment's risk config."""
    frames = build_all_frames(raw_data, signal_config)
    train_frames, test_frames, common_start, split_point, common_end = split_train_test(frames)

    train_result = run_backtest(train_frames, starting_cash=STARTING_CASH,
                                 commission_per_trade=COMMISSION_PER_TRADE, config=risk_config)
    train_metrics = compute_metrics(train_result, STARTING_CASH)

    test_result = run_backtest(test_frames, starting_cash=STARTING_CASH,
                                commission_per_trade=COMMISSION_PER_TRADE, config=risk_config)
    test_metrics = compute_metrics(test_result, STARTING_CASH)

    checks = evaluate_requirements(train_metrics, test_metrics)

    return {
        "label": label, "signal_config": signal_config, "risk_config": risk_config,
        "train_result": train_result, "test_result": test_result,
        "train_metrics": train_metrics, "test_metrics": test_metrics,
        "checks": checks,
        "split_info": (common_start, split_point, common_end),
    }


def print_comparison_table(experiment_results):
    print("\n" + "=" * 100)
    print("COMPARISON ACROSS ALL VARIATIONS")
    print("=" * 100)

    header = (
        f"{'Config':<45} {'TrainDD%':>8} {'TestDD%':>8} {'TestPF':>7} "
        f"{'Trades':>7} {'TestRet%':>9} {'TestFin$':>9} {'Trd':>4} {'OOS+':>5} {'DD':>4} {'PF':>4} {'ALL':>5}"
    )
    print(header)
    print("-" * len(header))
    print("(TestRet% and TestDD% are trading-only/scored - financing excluded; TestFin$ shows it separately)")

    for exp in experiment_results:
        tm, sm, c = exp["train_metrics"], exp["test_metrics"], exp["checks"]
        row = (
            f"{exp['label']:<45} "
            f"{tm['max_drawdown_pct']:>8.2f} "
            f"{sm['max_drawdown_pct']:>8.2f} "
            f"{sm['profit_factor']:>7.3f} "
            f"{c['total_completed']:>7} "
            f"{sm['total_return_pct']:>9.2f} "
            f"{sm['total_financing_paid']:>9.2f} "
            f"{'OK' if c['trades_pass'] else 'X':>4} "
            f"{'OK' if c['oos_positive_pass'] else 'X':>5} "
            f"{'OK' if c['drawdown_pass'] else 'X':>4} "
            f"{'OK' if c['profit_factor_pass'] else 'X':>4} "
            f"{'PASS' if c['all_pass'] else 'fail':>5}"
        )
        print(row)

    print("=" * 100)
    print("TrainDD%/TestDD% = max drawdown per period (fails if either >= 10%). TestPF = profit")
    print("factor on out-of-sample data. Trades = completed trades, train+test combined (need >=150).")
    print("TestRet% = out-of-sample total return. Trd/OOS+/DD/PF = pass(OK)/fail(X) on each individual")
    print("requirement; ALL = every requirement passing at once.")


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("The only OANDA API call anywhere in this project is a read-only")
    print("historical-candles fetch (data_fetch.py).")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(list(INSTRUMENTS.keys()))
    if is_synthetic:
        print(
            "\n*** NOTE: Using SYNTHETIC sample data (not real OANDA history), because "
            "live data could not be fetched. Every metric and comparison below is "
            "MEANINGLESS as a strategy evaluation on this data - it only proves the code "
            "runs end to end. Fix your OANDA credentials in .env and re-run for a real "
            "evaluation. ***\n"
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
            # Full detail for the baseline, as a reference point.
            print_period_report("TRAINING", result["train_result"], result["train_metrics"])
            print_period_report("TESTING (unseen data)", result["test_result"], result["test_metrics"])
            print_final_verdict(result["checks"])
        else:
            # Shorter summary for each variation - the full comparison table at
            # the end is where these are meant to be read side by side.
            tm, sm, c = result["train_metrics"], result["test_metrics"], result["checks"]
            print(f"Train: return {tm['total_return_pct']:.2f}% (trading only), drawdown {tm['max_drawdown_pct']:.2f}%, "
                  f"{tm['n_completed_trades']} trades, PF {tm['profit_factor']:.3f}, "
                  f"financing ${tm['total_financing_paid']:.2f}")
            print(f"Test:  return {sm['total_return_pct']:.2f}% (trading only), drawdown {sm['max_drawdown_pct']:.2f}%, "
                  f"{sm['n_completed_trades']} trades, PF {sm['profit_factor']:.3f}, "
                  f"financing ${sm['total_financing_paid']:.2f}")
            print(f"Meets all 4 requirements: {'YES' if c['all_pass'] else 'no'}")

    print_comparison_table(experiment_results)

    print("\nKnown approximations in this backtest (see file docstrings for detail):")
    print("  - Economic calendar: a recurring weekday time-window heuristic, not a real")
    print("    calendar - misses FOMC/ECB/BoE decisions and one-off events (econ_calendar.py)")
    print("  - Swap/financing rates: static annual approximations, not real historical")
    print("    OANDA rates, which move with central bank policy (instruments.py)")
    print("  - Rollover hour fixed at 21:00 UTC year-round (doesn't track US DST exactly)")


if __name__ == "__main__":
    main()
