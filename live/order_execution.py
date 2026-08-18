"""
order_execution.py
---------------------
Turns an accepted signal into a real (demo-account) OANDA order: takes
real numbers only (current account balance, real per-instrument margin
rate, the signal candle's ATR/spread), sizes the trade via the SAME
position-sizing/stop-target math used in the backtest (risk_management.py
- no reimplementation), places the order, and verifies the stop-loss
actually attached - force-closing immediately if not.

*** Places REAL orders on the OANDA PRACTICE account. Demo money only,
but treat every call here as consequential - see oanda_live_client.py's
module docstring. ***
"""

import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruments import INSTRUMENTS, value_per_price_unit, notional_value_per_unit
from risk_management import calculate_stop_and_target, calculate_position_size, RiskConfig

from oanda_live_client import place_market_order_with_stop, get_open_trades, close_trade


def _leverage_cap_for_instrument(symbol, real_specs, config):
    """The STRICTER of our own policy cap (RiskConfig.max_leverage) and
    OANDA's real margin-implied leverage for this instrument
    (1 / margin_rate) - Gold's real margin rate (5%, i.e. 20:1) was found
    to be stricter than this project's blanket 30:1 policy cap while
    building the live connector (see oanda_live_client.py's original
    docstring); using our own cap alone there would size trades OANDA
    itself would reject on margin grounds."""
    oanda_symbol = INSTRUMENTS[symbol].oanda_symbol
    spec = real_specs.get(oanda_symbol)
    if spec is None or not spec.get("margin_rate"):
        return config.max_leverage  # fall back to policy cap if OANDA didn't return this instrument
    real_leverage_cap = 1.0 / spec["margin_rate"]
    return min(config.max_leverage, real_leverage_cap)


def execute_signal(symbol, direction, latest_row, real_specs, account_balance, account_currency,
                    current_prices, config=RiskConfig()):
    """
    `latest_row`: the most recent CLOSED candle's row (dict-like, e.g. a
    pandas Series) for this instrument, from the same signal frame
    prepare_instrument_frame() produces - gives Close/atr, the same
    reference-price inputs the backtest uses to size a trade at signal
    time.
    `real_specs`: output of oanda_live_client.get_instrument_trading_specs().
    `account_balance`: real balance fetched once per cycle by the
    caller (run_live.py) - not re-fetched here, so multiple orders
    placed in the same cycle size consistently against the same number.
    `current_prices`: {symbol: last_close} for ALL instruments this
    cycle - needed for true cross-currency triangulation on a non-USD
    account (see instruments.py).

    Returns a dict describing what happened (for live_logging.py) - never
    raises for an expected rejection (bad spread already filtered by the
    caller, size too small, order not filled, stop didn't attach) - only
    lets a genuine API/connectivity exception propagate, which run_live.py's
    per-cycle try/except catches and logs.
    """
    spec = INSTRUMENTS[symbol]
    oanda_symbol = spec.oanda_symbol
    real_spec = real_specs.get(oanda_symbol)
    if real_spec is None:
        return {"action": "rejected", "reason": "no_real_specs_returned"}

    price_precision = real_spec["display_precision"]

    _, take_profit_price_est, stop_distance = calculate_stop_and_target(
        direction, latest_row["Close"], latest_row["atr"], config=config
    )

    value_per_unit = value_per_price_unit(
        symbol, latest_row["Close"], account_currency=account_currency, current_prices=current_prices
    )
    notional_per_unit = notional_value_per_unit(
        symbol, latest_row["Close"], account_currency=account_currency, current_prices=current_prices
    )

    leverage_cap = _leverage_cap_for_instrument(symbol, real_specs, config)
    effective_config = config if leverage_cap == config.max_leverage else replace(config, max_leverage=leverage_cap)

    size = calculate_position_size(
        account_balance, stop_distance, value_per_unit, spec.min_units,
        notional_per_unit,
        # reference_balance_for_leverage = the current real balance, not a
        # fixed backtest starting_cash: the backtest needed a FIXED
        # reference to break a runaway leverage/financing feedback loop
        # from its own approximate, self-compounding financing model (see
        # risk_management.py's docstring) - that failure mode doesn't
        # exist live, since OANDA charges/credits real financing directly
        # against the real account, not something this project models or
        # compounds itself.
        account_balance,
        config=effective_config,
    )

    if size == 0:
        return {"action": "rejected", "reason": "min_size_exceeds_risk_budget"}

    units = size if direction == "long" else -size

    response = place_market_order_with_stop(
        oanda_symbol, units, stop_distance, take_profit_price_est, price_precision
    )

    fill_txn = response.get("orderFillTransaction")
    if fill_txn is None:
        # Order didn't fill (e.g. rejected by OANDA) - nothing to verify or unwind.
        return {"action": "order_not_filled", "response": response}

    trade_id = fill_txn.get("tradeOpened", {}).get("tradeID")
    fill_price = float(fill_txn.get("price", 0.0))

    # --- Verify the stop actually attached before trusting this trade is protected ---
    time.sleep(1.0)  # give OANDA a moment to finish linking the SL/TP orders
    open_trades = get_open_trades()
    matching = next((t for t in open_trades if t.get("id") == trade_id), None)

    if matching is None:
        # Not open anymore by the time we checked - almost certainly
        # already closed by its own stop/target in a fast market (a very
        # tight ATR-based distance can do this). Nothing to verify or
        # force-close - calling close_trade() on an already-closed trade
        # would raise, not a real failure here.
        return {"action": "opened_and_closed_before_verification", "trade_id": trade_id, "fill_price": fill_price}

    stop_attached = bool(matching.get("stopLossOrder"))

    if not stop_attached:
        close_trade(trade_id)
        return {"action": "force_closed_no_stop", "trade_id": trade_id, "fill_price": fill_price}

    return {
        "action": "opened", "trade_id": trade_id, "direction": direction, "units": units,
        "fill_price": fill_price, "stop_distance": stop_distance,
        "take_profit_price": take_profit_price_est, "balance_at_entry": account_balance,
    }
