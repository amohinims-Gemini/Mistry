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

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS
from data_fetch import fetch_all
from signals import prepare_instrument_frame, SignalConfig
from backtest_engine import run_backtest
from risk_management import RiskConfig

STARTING_CASH = 10_000
COMMISSION_PER_TRADE = 0.0   # OANDA's standard retail accounts are spread-only, no separate
                             # commission - the real transaction cost here is the bid/ask
                             # spread itself, already modeled via realistic bid/ask fills.
TRAIN_FRACTION = 0.70

# --- Standing validation bar, STRENGTHENED after Candidate 2's rejection ---
# Raised from the original spec's 150 trades / PF > 1.2 to this stricter bar
# for every strategy round going forward (Candidate 2's own result, PF 0.912
# train / 0.546 test, already failed the OLD bar decisively - this change
# doesn't retroactively alter that verdict, it only raises what "passing"
# means for whatever is tested next). 200 is the enforced floor of the
# requested 200-300 trade range; 300 is treated as a "comfortable" level
# worth noting in reports, not a hard additional gate.
MIN_REQUIRED_TRADES = 200
MAX_ALLOWED_DRAWDOWN_PCT = 10.0
MIN_REQUIRED_PROFIT_FACTOR = 1.3

# --- Extended standing criteria (new - see evaluate_extended_requirements()
# below). These need trade-level data the simple train/test metrics dicts
# don't carry, so they're a separate, additive check - not folded into
# evaluate_requirements() so every already-written entry-point script that
# calls evaluate_requirements(train_metrics, test_metrics) unchanged keeps
# working exactly as before. Any FUTURE strategy's entry point should call
# BOTH evaluate_requirements() and evaluate_extended_requirements().
#
# Every threshold below is a first-pass, explicitly disclosed convention,
# not something precisely specified - flagged here rather than presented as
# more authoritative than it is:
IS_OOS_MAX_RELATIVE_PF_DROP = 0.35      # test PF must not fall more than this
                                          # fraction below train PF
MAX_SINGLE_TRADE_PROFIT_SHARE = 0.20    # no single trade may account for more
                                          # than this fraction of total gross profit
MAX_SINGLE_MONTH_PROFIT_SHARE = 0.40    # no single calendar month may account for
                                          # more than this fraction of total net profit
MIN_POSITIVE_REGIME_YEARS_FRACTION = 0.60  # at least this fraction of calendar
                                          # years (used as a pragmatic regime
                                          # proxy - matches the per-year
                                          # breakdown already used throughout
                                          # this project's reports, e.g. H4,
                                          # Candidate 2 - not a sophisticated
                                          # volatility/trend regime classifier)
                                          # must be individually net-profitable


# =============================================================================
# Experiments - each is (label, SignalConfig, RiskConfig).
#
# REDESIGN RESUMED, item #2 - strengthen the trend filter itself. A deep
# dive into the worst stress-test episode found the 1H efficiency filter
# (item from the prior round) does NOT fix a 4H trend that's technically
# "up" but not actually pushing with conviction - measuring efficiency on
# the entry timeframe couldn't distinguish that from real chop. This
# sweep tests the same Efficiency Ratio measure applied to the 4H series
# instead (signals.py's use_trend_efficiency_filter). Deliberately ONE
# free parameter (the threshold) at spec-default entry logic (channel=20,
# no other filters) - not combining with anything else yet, to isolate
# its effect first.
# =============================================================================

TREND_EFFICIENCY_THRESHOLDS_TO_TEST = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

EXPERIMENTS = [("no trend-efficiency filter (baseline)", SignalConfig(), RiskConfig())] + [
    (f"trend efficiency > {t}", SignalConfig(use_trend_efficiency_filter=True, trend_efficiency_min_threshold=t), RiskConfig())
    for t in TREND_EFFICIENCY_THRESHOLDS_TO_TEST
]


def build_all_frames(raw_data, signal_config=SignalConfig()):
    """Build each instrument's signal frame from already-fetched raw
    OANDA data, using the given SignalConfig."""
    frames = {}
    for symbol in PORTFOLIO_SYMBOLS:
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
    toward the MIN_REQUIRED_TRADES requirement or win-rate/profit-factor.

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


def evaluate_extended_requirements(train_result, test_result, train_metrics, test_metrics):
    """The NEW standing checks (see the constants above), additive to
    evaluate_requirements() - not folded into it so every already-written
    entry-point script keeps working unchanged. Any FUTURE strategy's
    entry point should call this ALONGSIDE evaluate_requirements().

    Needs trade-level data (per-trade P&L, entry_time) that the simple
    train/test metrics dicts don't carry, so it takes the full backtest
    result dicts (train_result/test_result, each with a "trades"
    DataFrame) rather than just the summary metrics.
    """
    train_trades = train_result["trades"]
    test_trades = test_result["trades"]
    train_completed = train_trades[train_trades["exit_reason"].isin(["stop_loss", "take_profit"])]
    test_completed = test_trades[test_trades["exit_reason"].isin(["stop_loss", "take_profit"])]
    combined = pd.concat([train_completed, test_completed], ignore_index=False)

    # --- 1. In-sample vs out-of-sample similarity: both periods must
    # independently clear the PF bar, AND test PF must not have collapsed
    # relative to train PF (the Candidate 2 failure mode - train PF 0.912,
    # test PF 0.546, a >40% relative drop - this check exists specifically
    # to catch that pattern even if some future case's test PF alone were
    # to clear MIN_REQUIRED_PROFIT_FACTOR on a small/lucky sample). ---
    train_pf, test_pf = train_metrics["profit_factor"], test_metrics["profit_factor"]
    if train_pf > 0 and not (isinstance(train_pf, float) and train_pf == float("inf")):
        pf_ratio = test_pf / train_pf
    else:
        pf_ratio = float("nan")
    is_oos_similarity_pass = (
        train_pf > MIN_REQUIRED_PROFIT_FACTOR
        and test_pf > MIN_REQUIRED_PROFIT_FACTOR
        and not pd.isna(pf_ratio)
        and pf_ratio >= (1 - IS_OOS_MAX_RELATIVE_PF_DROP)
    )

    # --- 2. No single trade dominates the profit ---
    if len(combined):
        gross_profit = combined.loc[combined["pnl"] > 0, "pnl"].sum()
        max_single_trade_pnl = combined["pnl"].max() if len(combined) else 0.0
        single_trade_share = (max_single_trade_pnl / gross_profit) if gross_profit > 0 else float("nan")
    else:
        single_trade_share = float("nan")
    no_single_trade_dominance_pass = (
        not pd.isna(single_trade_share) and single_trade_share <= MAX_SINGLE_TRADE_PROFIT_SHARE
    )

    # --- 3. No single calendar month dominates the profit ---
    if len(combined):
        combined_by_month = combined.copy()
        # tz_localize(None) first: only the calendar month bucket matters here,
        # and to_period() would otherwise warn about dropping tz info anyway.
        combined_by_month["month"] = pd.to_datetime(
            combined_by_month["entry_time"]).dt.tz_localize(None).dt.to_period("M")
        monthly_pnl = combined_by_month.groupby("month")["pnl"].sum()
        total_net_pnl = monthly_pnl.sum()
        max_month_pnl = monthly_pnl.max() if len(monthly_pnl) else 0.0
        month_share = (max_month_pnl / total_net_pnl) if total_net_pnl > 0 else float("nan")
    else:
        month_share = float("nan")
    no_single_month_dominance_pass = (
        not pd.isna(month_share) and 0 <= month_share <= MAX_SINGLE_MONTH_PROFIT_SHARE
    )

    # --- 4. Positive results across more than one market regime, using
    # calendar year as a pragmatic proxy (same breakdown already used
    # throughout this project's reports - not a volatility/trend regime
    # classifier, a documented simplification) ---
    if len(combined):
        combined_by_year = combined.copy()
        combined_by_year["year"] = pd.to_datetime(combined_by_year["entry_time"]).dt.year
        yearly_pnl = combined_by_year.groupby("year")["pnl"].sum()
        n_years = len(yearly_pnl)
        n_positive_years = (yearly_pnl > 0).sum()
        positive_year_fraction = n_positive_years / n_years if n_years else 0.0
    else:
        n_years, n_positive_years, positive_year_fraction = 0, 0, 0.0
    multi_regime_pass = n_years >= 2 and positive_year_fraction >= MIN_POSITIVE_REGIME_YEARS_FRACTION

    checks = {
        "is_oos_similarity_pass": is_oos_similarity_pass,
        "no_single_trade_dominance_pass": no_single_trade_dominance_pass,
        "no_single_month_dominance_pass": no_single_month_dominance_pass,
        "multi_regime_pass": multi_regime_pass,
        "train_pf": train_pf, "test_pf": test_pf, "pf_ratio": pf_ratio,
        "max_single_trade_profit_share": single_trade_share,
        "max_single_month_profit_share": month_share,
        "n_years": n_years, "n_positive_years": n_positive_years,
        "positive_year_fraction": positive_year_fraction,
    }
    checks["all_extended_pass"] = all(
        v for k, v in checks.items() if k.endswith("_pass")
    )
    return checks


def print_extended_verdict(checks):
    print("\n" + "=" * 78)
    print("EXTENDED standing criteria (adopted after Candidate 2's rejection)")
    print("=" * 78)
    print(f"[{'PASS' if checks['is_oos_similarity_pass'] else 'FAIL'}] "
          f"Train PF ({checks['train_pf']:.3f}) and test PF ({checks['test_pf']:.3f}) both above "
          f"{MIN_REQUIRED_PROFIT_FACTOR}, test not more than {IS_OOS_MAX_RELATIVE_PF_DROP*100:.0f}% "
          f"below train (ratio {checks['pf_ratio']:.3f})")
    share = checks["max_single_trade_profit_share"]
    print(f"[{'PASS' if checks['no_single_trade_dominance_pass'] else 'FAIL'}] "
          f"No single trade exceeds {MAX_SINGLE_TRADE_PROFIT_SHARE*100:.0f}% of gross profit "
          f"({'n/a' if pd.isna(share) else f'{share*100:.1f}%'} observed)")
    mshare = checks["max_single_month_profit_share"]
    print(f"[{'PASS' if checks['no_single_month_dominance_pass'] else 'FAIL'}] "
          f"No single calendar month exceeds {MAX_SINGLE_MONTH_PROFIT_SHARE*100:.0f}% of net profit "
          f"({'n/a' if pd.isna(mshare) else f'{mshare*100:.1f}%'} observed)")
    print(f"[{'PASS' if checks['multi_regime_pass'] else 'FAIL'}] "
          f"At least {MIN_POSITIVE_REGIME_YEARS_FRACTION*100:.0f}% of calendar years "
          f"individually net-profitable ({checks['n_positive_years']}/{checks['n_years']} years, "
          f"regime proxy)")
    print("-" * 78)
    if checks["all_extended_pass"]:
        print("Overall: PASSES every extended requirement too.")
    else:
        print("Overall: DOES NOT pass every extended requirement.")
    print("=" * 78)


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
    print(f"TrainDD%/TestDD% = max drawdown per period (fails if either >= {MAX_ALLOWED_DRAWDOWN_PCT:.0f}%). "
          f"TestPF = profit")
    print(f"factor on out-of-sample data. Trades = completed trades, train+test combined "
          f"(need >={MIN_REQUIRED_TRADES}).")
    print("TestRet% = out-of-sample total return. Trd/OOS+/DD/PF = pass(OK)/fail(X) on each individual")
    print("requirement; ALL = every requirement passing at once.")


def main():
    print("=" * 78)
    print("REMINDER: This is a BACKTEST only - a historical simulation.")
    print("No real trades are placed and no broker/account connection is used.")
    print("The only OANDA API call anywhere in this project is a read-only")
    print("historical-candles fetch (data_fetch.py).")
    print("=" * 78)

    raw_data, is_synthetic = fetch_all(PORTFOLIO_SYMBOLS)
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
