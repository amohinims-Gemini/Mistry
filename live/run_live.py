"""
run_live.py
--------------
*** PLACES REAL ORDERS ON THE OANDA PRACTICE (DEMO) ACCOUNT. ***
Demo money only - verify_practice_account() (account_safety.py) runs
first, every time, with no bypass, and refuses to start against
anything that isn't a genuine OANDA practice account.

*** THE STRATEGY PLUGGED IN HERE (signals_4h.py) IS NOT VALIDATED. ***
It failed this project's own validation bars (see README.md's
"Systematic strategy search: on hold"). This script exists ONLY to
prove the execution plumbing works end-to-end - fetch price, check
signal, size position, place order, log, stop mechanism - so that
plumbing doesn't have to be built from scratch once a validated
strategy exists. Swapping one in later means changing the SIGNAL_CONFIG/
prepare_instrument_frame import below and nothing else in this loop.

Position MANAGEMENT is much simpler live than in the backtest: OANDA's
own broker-held stop-loss/take-profit (attached atomically at entry)
execute automatically on their side. This loop never decides exits - it
only needs to notice, each cycle, that a position closed (via
live_account_sync), to keep the consecutive-loss count and logs current.

Usage:
    source venv/bin/activate
    python live/run_live.py

Stop mechanisms (all three work):
  - Ctrl+C: finishes the current cycle cleanly, then exits.
  - Create a file at live_data/STOP (any content, or empty) - checked
    every cycle and every second while sleeping between cycles.
  - MAX_RUNTIME_HOURS below (default 24) - stops automatically so a
    "just proving it works" run can't be left running unattended.
"""

import os
import sys
import time
import signal
import datetime as dt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from account_safety import verify_practice_account
from oanda_live_client import get_instrument_trading_specs
from live_state import LiveRiskState
from live_account_sync import get_account_balances, get_live_open_positions, get_fresh_consecutive_losses, LivePosition
from order_execution import execute_signal
import live_logging as log

from instruments import INSTRUMENTS, PORTFOLIO_SYMBOLS, value_per_price_unit
from data_fetch import get_instrument_data
from signals_4h import prepare_instrument_frame, Signal4HConfig
from risk_management import (
    check_trade_gates, spread_is_acceptable, near_economic_announcement, maybe_start_cooldown, RiskConfig,
)

POLL_INTERVAL_SECONDS = 5 * 60
MAX_RUNTIME_HOURS = 24
BAR_DURATION_HOURS = 4               # signals_4h.py - matches run_backtest_4h.py
STALE_CANDLE_BUFFER_HOURS = 2        # a fresh 4H candle can legitimately be up to BAR_DURATION_HOURS
                                       # old the moment it closes; add a buffer for poll cadence/API lag
                                       # before treating an old "latest candle" as a real data problem
STOP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_data", "STOP")

SIGNAL_CONFIG = Signal4HConfig()
RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=6)  # same 4H correctness adjustment as run_backtest_4h.py

_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    print("\nCtrl+C received - finishing this cycle, then stopping.")
    _stop_requested = True


def _stop_file_present():
    return os.path.exists(STOP_FILE)


def _candle_is_stale(latest_ts):
    age_hours = (dt.datetime.now(dt.timezone.utc) - latest_ts).total_seconds() / 3600
    return age_hours > BAR_DURATION_HOURS + STALE_CANDLE_BUFFER_HOURS


def run_cycle(state, last_seen_ts, real_specs, account_currency):
    balances = get_account_balances()
    log.log_event("account_state", **balances)

    # --- Pull the latest closed candle for every instrument first - both
    # for signal-checking below AND because current_prices is needed for
    # cross-currency risk_pct math on a non-USD account.
    frames = {}
    current_prices = {}
    for symbol in PORTFOLIO_SYMBOLS:
        oanda_symbol = INSTRUMENTS[symbol].oanda_symbol
        df = get_instrument_data(oanda_symbol, "H4")
        frame = prepare_instrument_frame(df, config=SIGNAL_CONFIG)
        frames[symbol] = frame
        if len(frame):
            current_prices[symbol] = float(frame["Close"].iloc[-1])

    open_positions, warnings = get_live_open_positions(account_currency, current_prices=current_prices)
    for w in warnings:
        log.log_event("warning", message=w)
    correlation_groups_open = {p.correlation_group for p in open_positions.values()}

    state.consecutive_losses = get_fresh_consecutive_losses()
    maybe_start_cooldown(state, dt.datetime.now(dt.timezone.utc).date(), config=RISK_CONFIG)
    state.update(dt.datetime.now(dt.timezone.utc), balances["nav"], config=RISK_CONFIG)

    for symbol, frame in frames.items():
        if len(frame) == 0:
            continue
        latest_ts = frame.index[-1]
        if last_seen_ts.get(symbol) == latest_ts:
            continue  # no new closed candle since last cycle for this instrument
        last_seen_ts[symbol] = latest_ts
        row = frame.iloc[-1]

        signal_long = bool(row.get("signal_long", False))
        signal_short = bool(row.get("signal_short", False))
        if not (signal_long or signal_short):
            log.log_event("signal_checked", symbol=symbol, candle_time=latest_ts, signal=None)
            continue
        direction = "long" if signal_long else "short"

        reject_reason = None
        if pd.isna(row.get("atr")):
            reject_reason = "atr_not_ready"
        elif _candle_is_stale(latest_ts):
            reject_reason = "stale_data"
        elif near_economic_announcement(latest_ts):
            reject_reason = "econ_calendar_blackout"
        elif not spread_is_acceptable(row["spread_close"], row["avg_spread_100"],
                                       INSTRUMENTS[symbol].hard_spread_cap, config=RISK_CONFIG):
            reject_reason = "spread_abnormal"
        else:
            ok, gate_reason = check_trade_gates(symbol, state, open_positions, correlation_groups_open,
                                                 config=RISK_CONFIG)
            if not ok:
                reject_reason = gate_reason

        if reject_reason is not None:
            log.log_event("signal_checked", symbol=symbol, candle_time=latest_ts,
                           signal=direction, rejected=reject_reason)
            continue

        result = execute_signal(
            symbol, direction, row, real_specs,
            account_balance=balances["balance"], account_currency=account_currency,
            current_prices=current_prices, config=RISK_CONFIG,
        )
        log.log_event("order_result", symbol=symbol, candle_time=latest_ts, **result)

        if result.get("action") == "opened":
            # CRITICAL: update the LOCAL open_positions/correlation_groups_open
            # immediately, in this same cycle - not just next cycle's fresh
            # get_live_open_positions() sync. Found live: without this, every
            # instrument in a cycle's loop was gated against the SAME stale
            # snapshot taken at the top of the cycle, so multiple instruments
            # in the same correlation group could all open in one cycle before
            # any of them showed up in that snapshot - the exact concentrated-
            # correlated-risk scenario correlation grouping exists to prevent.
            # Confirmed happening for real: EUR_USD, GBP_USD, and USD_JPY (all
            # "usd_fx") all opened in the same cycle before this fix.
            value_per_unit = value_per_price_unit(
                symbol, result["fill_price"], account_currency=account_currency, current_prices=current_prices
            )
            risk_dollars = result["stop_distance"] * abs(result["units"]) * value_per_unit
            risk_pct = risk_dollars / balances["nav"] if balances["nav"] > 0 else 0.0
            correlation_group = INSTRUMENTS[symbol].correlation_group
            open_positions[symbol] = LivePosition(
                symbol=symbol, correlation_group=correlation_group, risk_pct=risk_pct,
                trade_id=result["trade_id"], direction=direction, units=result["units"],
                entry_price=result["fill_price"],
            )
            correlation_groups_open.add(correlation_group)

    state.save()


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    print("=" * 78)
    print("*** LIVE (DEMO) EXECUTION - PLACES REAL ORDERS ON THE OANDA PRACTICE ACCOUNT ***")
    print("*** THE STRATEGY PLUGGED IN HERE (signals_4h.py) IS NOT VALIDATED ***")
    print("*** This run exists to prove the execution plumbing, not to trade profitably ***")
    print("=" * 78)

    account_info = verify_practice_account()
    account_currency = account_info.get("currency", "USD")

    real_specs = get_instrument_trading_specs([INSTRUMENTS[s].oanda_symbol for s in PORTFOLIO_SYMBOLS])

    state = LiveRiskState.load_or_create(starting_equity=float(account_info.get("balance", 0.0)))
    last_seen_ts = {}

    start_time = dt.datetime.now(dt.timezone.utc)
    cycle = 0
    log.log_event("run_started", account_currency=account_currency, max_runtime_hours=MAX_RUNTIME_HOURS,
                   poll_interval_seconds=POLL_INTERVAL_SECONDS)

    while not _stop_requested:
        if _stop_file_present():
            print(f"Stop file found at {STOP_FILE} - stopping.")
            log.log_event("stop_requested", reason="stop_file")
            break
        elapsed_hours = (dt.datetime.now(dt.timezone.utc) - start_time).total_seconds() / 3600
        if elapsed_hours >= MAX_RUNTIME_HOURS:
            print(f"Max runtime ({MAX_RUNTIME_HOURS}h) reached - stopping.")
            log.log_event("stop_requested", reason="max_runtime")
            break

        cycle += 1
        print(f"\n--- cycle {cycle} @ {dt.datetime.now(dt.timezone.utc).isoformat()} ---")
        try:
            run_cycle(state, last_seen_ts, real_specs, account_currency)
        except Exception as e:
            print(f"ERROR during cycle: {e}")
            log.log_event("error", message=str(e))

        for _ in range(POLL_INTERVAL_SECONDS):
            if _stop_requested or _stop_file_present():
                break
            time.sleep(1)

    log.log_event("run_stopped", cycles_completed=cycle)
    print("Stopped.")


if __name__ == "__main__":
    main()
