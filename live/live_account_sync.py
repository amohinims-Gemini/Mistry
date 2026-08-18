"""
live_account_sync.py
------------------------
Reconciles local expectations against OANDA's real account state each
cycle - real balance/NAV, real open trades (filtered to this bot's own
CLIENT_TAG), and a fresh consecutive-loss count derived from real closed-
trade history. Local state should never silently drift from what the
broker actually reports - a mismatch between what we expect and what
OANDA shows is returned as a loud warning to be logged, not silently
"corrected" without a trace.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruments import INSTRUMENTS, value_per_price_unit

from oanda_live_client import get_account_summary, get_open_trades, get_recent_closed_trades, CLIENT_TAG


class LivePosition:
    """Just enough about one open position to feed risk_management.py's
    shared check_trade_gates() (risk_pct, correlation_group) - not a
    full mirror of the backtest's Position dataclass, since live doesn't
    need to independently recompute P&L (OANDA's NAV already includes
    it)."""

    def __init__(self, symbol, correlation_group, risk_pct, trade_id, direction, units, entry_price):
        self.symbol = symbol
        self.correlation_group = correlation_group
        self.risk_pct = risk_pct
        self.trade_id = trade_id
        self.direction = direction
        self.units = units
        self.entry_price = entry_price


def get_account_balances():
    account = get_account_summary()
    return {
        "balance": float(account["balance"]),
        "nav": float(account["NAV"]),
        "currency": account["currency"],
        "margin_used": float(account.get("marginUsed", 0.0)),
    }


def get_live_open_positions(account_currency, current_prices=None):
    """Returns ({symbol: LivePosition}, [warning strings]) for every open
    trade tagged CLIENT_TAG, grouped by our own instrument symbol (not
    OANDA's). Never raises on a missing/unexpected stop-loss field on a
    trade - falls back to a deliberately conservative risk_pct estimate
    and returns a warning, so a field-name surprise on first live use
    degrades gracefully (blocks new trades via the risk cap rather than
    silently under-counting real exposure) instead of crashing the loop."""
    by_oanda_symbol = {spec.oanda_symbol: sym for sym, spec in INSTRUMENTS.items()}
    account = get_account_summary()
    nav = float(account["NAV"])

    positions = {}
    warnings = []
    for trade in get_open_trades():
        if trade.get("clientExtensions", {}).get("tag") != CLIENT_TAG:
            continue  # not one of ours - some other activity on this shared demo account

        oanda_symbol = trade["instrument"]
        symbol = by_oanda_symbol.get(oanda_symbol)
        if symbol is None:
            warnings.append(f"Open trade on untracked instrument {oanda_symbol} (trade {trade['id']})")
            continue

        entry_price = float(trade["price"])
        units = float(trade["currentUnits"])
        direction = "long" if units > 0 else "short"

        stop_order = trade.get("stopLossOrder")
        if stop_order and "price" in stop_order:
            stop_distance = abs(entry_price - float(stop_order["price"]))
            value_per_unit = value_per_price_unit(
                symbol, entry_price, account_currency=account_currency, current_prices=current_prices
            )
            risk_dollars = stop_distance * abs(units) * value_per_unit
            risk_pct = risk_dollars / nav if nav > 0 else 0.0
        else:
            warnings.append(
                f"Trade {trade['id']} ({symbol}) has no visible stop-loss order in the API "
                f"response - using a deliberately conservative risk_pct estimate"
            )
            risk_pct = 1.0  # huge on purpose: blocks new trades via total_open_risk_cap
                              # rather than silently under-counting real exposure

        positions[symbol] = LivePosition(
            symbol=symbol, correlation_group=INSTRUMENTS[symbol].correlation_group,
            risk_pct=risk_pct, trade_id=trade["id"], direction=direction,
            units=units, entry_price=entry_price,
        )

    return positions, warnings


def get_fresh_consecutive_losses(count=20):
    """Derive the consecutive-loss count fresh from real closed-trade
    history each cycle (newest first), rather than tracking it as local
    state that has to survive restarts perfectly - matches the pattern
    already documented as the intent in oanda_live_client.py."""
    closed = get_recent_closed_trades(count=count)
    ours = [t for t in closed if t.get("clientExtensions", {}).get("tag") == CLIENT_TAG]
    streak = 0
    for trade in ours:
        pnl = float(trade.get("realizedPL", 0.0))
        if pnl < 0:
            streak += 1
        else:
            break
    return streak
