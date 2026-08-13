"""
backtest_engine.py
--------------------
The custom multi-instrument, multi-timeframe backtest engine.

WHY NOT `backtesting.py` (the library used in earlier versions of this
project): that library runs one instrument through one isolated
strategy, with its own private equity curve. This spec needs a SHARED
account across 4 instruments at once - max 3 open positions total,
0.75% total open risk across all of them, one correlation-group trade at
a time, daily/weekly/drawdown circuit breakers on one shared equity
curve. None of that can be enforced by running 4 independent
single-asset backtests and comparing results afterwards - that would be
an approximation dressed up as a simulation. So this engine steps
through every instrument's 1H bars in true chronological lockstep,
against one shared `PortfolioAccount` (risk_management.py).

*** BACKTEST ONLY *** - this module never places a real order. It reads
already-fetched historical candle data and simulates fills against it in
memory. No broker/exchange connection exists anywhere in this file.

Execution model (confirmed with the user before building this):
  - A signal is evaluated using a bar's CLOSE (no mid-candle decisions).
  - If accepted, the order fills at the FOLLOWING bar's OPEN, at the
    appropriate side of the spread (ask for buys, bid for sells), plus
    slippage.
  - Stop-loss / take-profit hits are checked using each subsequent bar's
    Bid/Ask High/Low (a long position exits by selling, so it's checked
    against the Bid side; a short exits by buying, checked against Ask).
  - If both the stop and target fall within the same bar's range, we
    can't tell from OHLC data alone which was hit first - we
    conservatively assume the STOP hit first, so we never overstate
    performance from an ambiguous bar.
  - Take-profit fills assume no slippage (a take-profit behaves like a
    limit order - you get that price or better in real markets).
    Stop-loss and entry fills DO get slippage (they behave like market
    orders reacting to price crossing a level).
"""

import pandas as pd

from instruments import INSTRUMENTS, value_per_price_unit
from risk_management import (
    PortfolioAccount, calculate_stop_and_target, calculate_position_size,
    spread_is_acceptable, data_is_stale, near_economic_announcement,
    DEFAULT_CONFIG,
)

# Overnight financing "rollover" hour, in UTC. Real brokers roll over at
# 5pm New York time, which floats between 21:00 and 22:00 UTC depending
# on US daylight saving - we fix it at 21:00 UTC year-round rather than
# track the DST calendar exactly. Same spirit as the other approximations
# in this project (see instruments.py, econ_calendar.py): a reasonable,
# clearly-documented simplification, not a hidden shortcut.
ROLLOVER_HOUR_UTC = 21

SLIPPAGE_ATR_FRACTION = 0.02  # slippage on entry/stop fills, as a fraction of that bar's ATR


def _build_lookup_structures(instrument_frames):
    """For each instrument, build a {timestamp: row_dict} lookup and a
    {timestamp: next_timestamp} map (for scheduling next-bar-open fills).
    Doing this once up front makes the main loop far faster than
    repeated DataFrame.loc[] calls."""
    data_dict = {}
    next_ts_map = {}
    for symbol, df in instrument_frames.items():
        df = df.sort_index()
        data_dict[symbol] = df.to_dict("index")
        idx = df.index
        next_ts_map[symbol] = dict(zip(idx[:-1], idx[1:]))
    return data_dict, next_ts_map


def _slip(price, direction, amount):
    """Move `price` against the trader by `amount`: worse (higher) for a
    long buy, worse (lower) for a short sell."""
    return price + amount if direction == "long" else price - amount


def run_backtest(instrument_frames, starting_cash=10_000, commission_per_trade=0.0, config=DEFAULT_CONFIG):
    """
    Run the full multi-instrument backtest.

    `instrument_frames`: {symbol: DataFrame} - each already produced by
    signals.prepare_instrument_frame() and already sliced to the desired
    date range (e.g. the training or testing period).

    `config`: a risk_management.RiskConfig - all the tunable risk/safety
    numbers. Defaults to the spec's original values; pass a different
    RiskConfig to try a variation (see run_backtest.py's experiment runner).

    Returns a dict with the account, a trade log, a rejection log, and
    the shared equity curve.
    """
    data_dict, next_ts_map = _build_lookup_structures(instrument_frames)

    unified_timestamps = sorted(set().union(*[df.index for df in instrument_frames.values()]))

    account = PortfolioAccount(starting_cash, config=config)
    pending_orders = {}       # symbol -> order dict, awaiting fill at the next bar
    previous_ts_by_symbol = {}
    equity_curve = []

    for ts in unified_timestamps:
        current_prices = {}

        for symbol, df_dict in data_dict.items():
            row = df_dict.get(ts)
            if row is None:
                continue  # this instrument has no bar at this exact timestamp

            current_prices[symbol] = row["Close"]  # mid close, for mark-to-market
            spec = INSTRUMENTS[symbol]

            # --- Step 1: fill a pending order scheduled for this bar ------------
            if symbol in pending_orders and pending_orders[symbol]["fill_at"] == ts:
                order = pending_orders.pop(symbol)
                direction = order["direction"]

                if direction == "long":
                    raw_fill = row["Ask_Open"]
                    fill_price = _slip(raw_fill, "long", order["slippage_amount"])
                    stop_price = fill_price - order["stop_distance"]
                    take_profit_price = fill_price + order["target_distance"]
                else:
                    raw_fill = row["Bid_Open"]
                    fill_price = _slip(raw_fill, "short", order["slippage_amount"])
                    stop_price = fill_price + order["stop_distance"]
                    take_profit_price = fill_price - order["target_distance"]

                account.open_position(
                    symbol, direction, ts, fill_price, order["size"], stop_price, take_profit_price
                )

            # --- Step 2: manage an open position (financing, then SL/TP) --------
            if symbol in account.open_positions:
                position = account.open_positions[symbol]

                if ts.hour == ROLLOVER_HOUR_UTC:
                    notional_value = position.size_units * row["Close"]
                    account.apply_financing(symbol, ts, notional_value, position.direction)

                if position.direction == "long":
                    stop_hit = row["Bid_Low"] <= position.stop_price
                    tp_hit = row["Bid_High"] >= position.take_profit_price
                else:
                    stop_hit = row["Ask_High"] >= position.stop_price
                    tp_hit = row["Ask_Low"] <= position.take_profit_price

                if stop_hit or tp_hit:
                    if stop_hit:
                        # Conservative assumption when both are hit in the same bar
                        # (see module docstring): stop wins.
                        exit_price, reason = position.stop_price, "stop_loss"
                    else:
                        exit_price, reason = position.take_profit_price, "take_profit"
                    account.close_position(symbol, ts, exit_price, reason, commission=commission_per_trade)

            # --- Step 3: look for a new entry, only if flat and nothing pending --
            if symbol not in account.open_positions and symbol not in pending_orders:
                signal_long = row.get("signal_long", False)
                signal_short = row.get("signal_short", False)

                if signal_long or signal_short:
                    direction = "long" if signal_long else "short"
                    reject_reason = None

                    if pd.isna(row.get("atr")):
                        reject_reason = "atr_not_ready"
                    elif data_is_stale(ts, previous_ts_by_symbol.get(symbol), config=config):
                        reject_reason = "stale_data"
                    elif near_economic_announcement(ts):
                        reject_reason = "econ_calendar_blackout"
                    elif not spread_is_acceptable(
                        row["spread_close"], row["avg_spread_100"], spec.hard_spread_cap, config=config
                    ):
                        reject_reason = "spread_abnormal"
                    else:
                        ok, acct_reason = account.can_open_new_trade(symbol)
                        if not ok:
                            reject_reason = acct_reason

                    if reject_reason is None:
                        stop_price_est, take_profit_price_est, stop_distance = calculate_stop_and_target(
                            direction, row["Close"], row["atr"], config=config
                        )
                        target_distance = config.take_profit_atr_multiple * row["atr"]
                        value_per_unit = value_per_price_unit(symbol, row["Close"])
                        size = calculate_position_size(
                            account.balance, stop_distance, value_per_unit, spec.min_units,
                            row["Close"], config=config
                        )

                        if size == 0:
                            reject_reason = "min_size_exceeds_risk_budget"
                        else:
                            next_ts = next_ts_map[symbol].get(ts)
                            if next_ts is None:
                                reject_reason = "no_next_bar_to_fill_at"
                            else:
                                pending_orders[symbol] = {
                                    "direction": direction,
                                    "fill_at": next_ts,
                                    "size": size,
                                    "stop_distance": stop_distance,
                                    "target_distance": target_distance,
                                    "slippage_amount": SLIPPAGE_ATR_FRACTION * row["atr"],
                                }

                    if reject_reason is not None:
                        account.record_rejection(ts, symbol, reject_reason)

            previous_ts_by_symbol[symbol] = ts

        # --- After every instrument at this timestamp: update shared account state
        current_equity = account.equity(current_prices)
        account.handle_day_week_rollover(ts, current_equity)
        account.update_risk_flags(ts, current_equity)
        equity_curve.append({"time": ts, "equity": current_equity, "balance": account.balance})

    # Force-close anything still open at the end of the data, at the last known
    # price, so trade accounting is complete. Tagged distinctly - these aren't
    # real SL/TP exits, just "the backtest ran out of data".
    for symbol in list(account.open_positions.keys()):
        last_ts = max(data_dict[symbol].keys())
        last_price = data_dict[symbol][last_ts]["Close"]
        account.close_position(symbol, last_ts, last_price, "backtest_end", commission=commission_per_trade)

    return {
        "account": account,
        "equity_curve": pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame(),
        "trades": pd.DataFrame(account.closed_trades),
        "rejections": pd.DataFrame(account.rejection_log),
    }
