"""
oanda_live_client.py
----------------------
The ONLY place in this project that talks to OANDA endpoints beyond the
read-only historical candles used everywhere else (data_fetch.py).
Everything here is scoped to the PRACTICE (demo) environment - see
account_safety.py, which verifies this before run_live.py does anything
else, every single time, with no bypass.

This wraps:
  - account state (balance, NAV/equity)              - accounts_api
  - real per-instrument precision/margin requirements - accounts_api
  - open trades                                        - trades_api
  - recent trade history (for consecutive-loss check)  - trades_api
  - order placement WITH stop-loss/take-profit attached - orders_api
  - force-closing a trade (safety fallback)             - trades_api

*** This is the first code in the project that places real orders. ***
Even on a demo account, treat every function here as consequential.
"""

import os
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.accounts as accounts_api
import oandapyV20.endpoints.trades as trades_api
import oandapyV20.endpoints.orders as orders_api
from oandapyV20.contrib.requests import MarketOrderRequest, StopLossDetails, TakeProfitDetails

load_dotenv()

# Hardcoded, not configurable - see account_safety.py Layer 1. Nothing in
# this project should ever be able to silently point this at "live".
OANDA_ENVIRONMENT = "practice"

OANDA_API_TOKEN = os.environ.get("OANDA_API_TOKEN")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")

# Tags every order this bot places, so its own trades are unambiguously
# identifiable even if other activity ever touches the same demo account.
CLIENT_TAG = "mistry-live-v1"


def get_live_client():
    if not OANDA_API_TOKEN:
        raise RuntimeError("OANDA_API_TOKEN not set - see .env.example.")
    if not OANDA_ACCOUNT_ID:
        raise RuntimeError("OANDA_ACCOUNT_ID not set - see .env.example.")
    return oandapyV20.API(access_token=OANDA_API_TOKEN, environment=OANDA_ENVIRONMENT)


def get_account_summary():
    """Real current balance, NAV (equity - balance plus all unrealized
    P&L, computed by OANDA itself), margin used."""
    client = get_live_client()
    request = accounts_api.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    client.request(request)
    return request.response["account"]


def get_instrument_trading_specs(oanda_symbols):
    """Real per-instrument display precision (decimal places prices must
    be rounded to) and margin rate (1/leverage), fetched fresh from
    OANDA rather than assumed - found via testing that these genuinely
    differ per instrument (e.g. Gold's real margin rate is stricter than
    our own 30:1 risk policy cap) and that rounding prices to the wrong
    precision would get orders rejected outright."""
    client = get_live_client()
    request = accounts_api.AccountInstruments(
        accountID=OANDA_ACCOUNT_ID, params={"instruments": ",".join(oanda_symbols)}
    )
    client.request(request)
    specs = {}
    for inst in request.response["instruments"]:
        specs[inst["name"]] = {
            "display_precision": inst["displayPrecision"],
            "margin_rate": float(inst["marginRate"]),
            "minimum_trade_size": float(inst["minimumTradeSize"]),
        }
    return specs


def get_open_trades():
    """All currently open trades on the account (the caller filters by
    CLIENT_TAG if it only wants this bot's own trades)."""
    client = get_live_client()
    request = trades_api.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    client.request(request)
    return request.response["trades"]


def get_recent_closed_trades(count=20):
    """Most recently closed trades, newest first - used to derive the
    consecutive-loss count fresh each cycle rather than tracking it as
    fragile local state that has to survive restarts perfectly."""
    client = get_live_client()
    request = trades_api.TradesList(accountID=OANDA_ACCOUNT_ID, params={
        "state": "CLOSED", "count": count,
    })
    client.request(request)
    trades = request.response["trades"]
    trades.sort(key=lambda t: t.get("closeTime", ""), reverse=True)
    return trades


def place_market_order_with_stop(instrument, units, stop_distance, take_profit_price, price_precision):
    """
    Place a market order with a broker-held stop-loss AND take-profit
    attached in the SAME request, so OANDA creates them atomically with
    the fill - never a separate follow-up call that could fail or race.

    `units`: positive = buy (long), negative = sell (short).

    `stop_distance`: the stop-loss as a DISTANCE from the eventual fill
    price (already computed via risk_management.calculate_stop_and_target),
    NOT an absolute price - a market order's exact fill price isn't known
    until after it fills, so an absolute price computed in advance from
    the last known close could end up stale versus the real fill (spread/
    slippage/latency) and produce a nonsensical or rejected stop. OANDA
    anchors a distance-based stop to the ACTUAL fill price itself,
    preserving the intended R:R regardless of slippage - this also
    matches how the rest of this project already represents stops
    internally (distance first, converted to a price only once a fill
    price is known - see backtest_engine.py).

    `take_profit_price`: still an ABSOLUTE price, not a distance - OANDA's
    v20 API only supports price-based take-profit orders (no distance
    option, unlike stop-loss), so this is computed from the best
    available reference price (the signal candle's close) at call time.
    Slightly approximate for the same reason the stop-distance fix
    exists, but a take-profit being off by a small amount is a
    less-favorable-target risk, not a protection gap the way an
    imprecise stop-loss would be - OANDA will simply reject the order
    outright if this price is invalid relative to the real fill
    direction, rather than silently accepting something dangerous.

    `price_precision`: decimal places to round the stop distance/take-
    profit price to for this specific instrument (see
    get_instrument_trading_specs - this varies per instrument and a
    wrong value gets the order rejected).

    Returns the raw OANDA response dict. Raises on request failure - the
    caller (order_execution.py) is responsible for verifying the stop
    was actually attached and for the safety-abort-if-not logic.
    """
    client = get_live_client()

    order = MarketOrderRequest(
        instrument=instrument,
        units=units,
        stopLossOnFill=StopLossDetails(distance=f"{stop_distance:.{price_precision}f}").data,
        takeProfitOnFill=TakeProfitDetails(price=f"{take_profit_price:.{price_precision}f}").data,
        clientExtensions={"tag": CLIENT_TAG, "comment": "Mistry automated entry"},
    )
    request = orders_api.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order.data)
    client.request(request)
    return request.response


def close_trade(trade_id):
    """Market-close an open trade fully. Used as the safety fallback if
    a stop-loss attachment can't be confirmed after entry."""
    client = get_live_client()
    request = trades_api.TradeClose(accountID=OANDA_ACCOUNT_ID, tradeID=trade_id)
    client.request(request)
    return request.response
