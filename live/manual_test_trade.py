"""
manual_test_trade.py
------------------------
*** PLACES ONE REAL TEST ORDER ON THE OANDA PRACTICE ACCOUNT, THEN CLOSES IT. ***

A manual, one-off diagnostic - NOT part of run_live.py's loop, and NOT
driven by any strategy signal. Exists purely to verify the order-
placement path (order_execution.execute_signal -> real sizing -> real
stop-loss distance -> place_market_order_with_stop -> stop-attachment
verification) actually works end-to-end against OANDA's real practice
server, on demand, rather than waiting for signals_4h.py's placeholder
strategy to happen to fire on its own. Calls the EXACT SAME
execute_signal() function run_live.py's loop calls - this is a genuine
test of the real path, not a separate reimplementation of it.

Run this ONLY while run_live.py is NOT running - both scripts
independently manage orders on the same account and neither knows about
the other's in-flight state. (This project's own convention: stop
run_live.py first via live_data/STOP or Ctrl+C.)

Usage:
    source venv/bin/activate
    python live/manual_test_trade.py [SYMBOL] [long|short]
    # defaults to EUR_USD long if no arguments given
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from account_safety import verify_practice_account
from oanda_live_client import get_instrument_trading_specs, get_open_trades, close_trade
from order_execution import execute_signal
from live_account_sync import get_account_balances

from instruments import INSTRUMENTS
from data_fetch import get_instrument_data
from signals_4h import prepare_instrument_frame, Signal4HConfig
from risk_management import RiskConfig

RISK_CONFIG = RiskConfig(stale_data_max_gap_hours=6)  # same 4H correctness adjustment as run_live.py
SIGNAL_CONFIG = Signal4HConfig()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"
    direction = sys.argv[2] if len(sys.argv) > 2 else "long"
    if symbol not in INSTRUMENTS:
        print(f"Unknown symbol {symbol!r} - choose from {list(INSTRUMENTS)}")
        return
    if direction not in ("long", "short"):
        print(f"direction must be 'long' or 'short', got {direction!r}")
        return

    print("=" * 78)
    print("*** MANUAL TEST TRADE - places ONE real order on the OANDA PRACTICE ***")
    print("*** account, verifies it, then closes it again. NOT a strategy signal. ***")
    print(f"*** {symbol} {direction} ***")
    print("=" * 78)

    account_info = verify_practice_account()
    account_currency = account_info.get("currency", "USD")

    real_specs = get_instrument_trading_specs([spec.oanda_symbol for spec in INSTRUMENTS.values()])
    balances = get_account_balances()
    print(f"\nAccount balance: {balances['balance']} {balances['currency']}, NAV: {balances['nav']}")

    current_prices = {}
    test_row = None
    for sym in INSTRUMENTS:
        df = get_instrument_data(INSTRUMENTS[sym].oanda_symbol, "H4")
        frame = prepare_instrument_frame(df, config=SIGNAL_CONFIG)
        if len(frame):
            current_prices[sym] = float(frame["Close"].iloc[-1])
        if sym == symbol:
            test_row = frame.iloc[-1]

    print(f"Using latest {symbol} 4H candle: close={test_row['Close']}, atr={test_row['atr']:.5f}")
    print(f"\nCalling order_execution.execute_signal({symbol}, {direction!r}, ...) - the exact "
          f"function run_live.py's loop calls on a real signal...\n")

    result = execute_signal(
        symbol, direction, test_row, real_specs,
        account_balance=balances["balance"], account_currency=account_currency,
        current_prices=current_prices, config=RISK_CONFIG,
    )
    print("execute_signal() result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result.get("action") != "opened":
        print(f"\nTrade did not open cleanly (action={result.get('action')}) - nothing to close. See above.")
        return

    trade_id = result["trade_id"]

    # --- Confirm what OANDA itself thinks is on the books, independent of
    # execute_signal()'s own internal verification step ---
    open_trades = get_open_trades()
    matching = next((t for t in open_trades if t.get("id") == trade_id), None)
    if matching:
        print(f"\nIndependently confirmed via get_open_trades(): trade {trade_id} is open on OANDA.")
        print(f"  instrument={matching.get('instrument')} units={matching.get('currentUnits')} "
              f"price={matching.get('price')}")
        print(f"  stopLossOrder={matching.get('stopLossOrder')}")
        print(f"  takeProfitOrder={matching.get('takeProfitOrder')}")
    else:
        print(f"\nWARNING: trade {trade_id} not found in get_open_trades() - it may have already closed "
              f"(e.g. a very tight stop/target hit almost immediately).")

    # --- Close it back out - this was a plumbing test, not a real position ---
    print(f"\nClosing test trade {trade_id}...")
    close_response = close_trade(trade_id)
    closed_txn = close_response.get("orderFillTransaction", {})
    print(f"Closed. Realized P&L: {closed_txn.get('pl', '?')} {account_currency} "
          f"(expected: roughly -spread, since this closes almost immediately after opening)")


if __name__ == "__main__":
    main()
