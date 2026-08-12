"""
risk_management.py
--------------------
Everything that decides whether a technically-valid signal (from
signals.py) actually gets traded, and how big that trade is:

  - Position sizing: risk a fixed % of account BALANCE per trade, based
    on the ATR-derived stop distance and the instrument's pip/point value.
  - ATR-based stop-loss / take-profit levels (1.5x ATR / 3x ATR).
  - `PortfolioAccount`: a stateful object the backtest engine drives bar
    by bar, tracking a SHARED account across all instruments and
    enforcing every safety limit from the spec: daily/weekly loss halts,
    drawdown suspension, consecutive-loss cooldown, max open positions,
    per-correlation-group exclusivity, total open risk cap, spread and
    staleness checks, and overnight financing (swap) costs.

A note on BALANCE vs EQUITY (this distinction is deliberate, not
sloppiness): position sizing uses BALANCE (realized P&L only - closed
trades), matching the spec's literal "account balance" wording, so
floating gains can't be used to justify bigger size and floating losses
don't shrink it either. The daily/weekly loss limits and the drawdown
suspension use EQUITY (balance + unrealized P&L of open positions),
since those are meant to catch real-time risk exposure, not just money
that's already locked in.
"""

import math
from dataclasses import dataclass, field

from instruments import INSTRUMENTS, value_per_price_unit
from econ_calendar import is_blackout_window

# --- Strategy-wide constants (the spec's numbers) ---------------------------
RISK_PER_TRADE_PCT = 0.0025        # risk 0.25% of balance per trade
STOP_LOSS_ATR_MULTIPLE = 1.5
TAKE_PROFIT_ATR_MULTIPLE = 3.0     # 3x / 1.5x = 2:1 reward-to-risk

MAX_OPEN_POSITIONS = 3
TOTAL_OPEN_RISK_CAP_PCT = 0.0075   # 0.75%
DAILY_LOSS_LIMIT_PCT = 0.01        # stop for the day after a 1% loss
WEEKLY_LOSS_LIMIT_PCT = 0.025      # stop for the week after a 2.5% loss
DRAWDOWN_SUSPEND_PCT = 0.08        # suspend ALL trading after an 8% equity drawdown from peak
DRAWDOWN_RESUME_PCT = 0.04         # resume once drawdown recovers back to within 4% of peak -
                                    # a hysteresis gap (not the same 8% line) so the account
                                    # doesn't flip suspended/unsuspended on every tiny wiggle
                                    # right at the boundary
DRAWDOWN_SUSPEND_MAX_COOLDOWN_DAYS = 30
                                    # Equity-recovery resume can never fire if there's nothing
                                    # left open to move equity (the only lever is trading, and
                                    # trading is exactly what's blocked - a real deadlock we hit
                                    # in testing). This is the fallback: resume on EITHER equity
                                    # recovering to DRAWDOWN_RESUME_PCT OR this many days
                                    # elapsing since suspension, whichever comes first.
CONSECUTIVE_LOSS_LIMIT = 3         # pause until the next trading day after 3 losses in a row

SPREAD_REJECT_MULTIPLE = 2.5       # reject if spread > 2.5x its own trailing 100-bar average
STALE_DATA_MAX_GAP_HOURS = 3       # reject if the last candle is older than this (non-weekend)
STALE_DATA_MAX_WEEKEND_GAP_HOURS = 72  # forex closes for the weekend - allow for that gap


# =============================================================================
# Position sizing and stop/target calculation
# =============================================================================

def calculate_stop_and_target(direction, entry_price, atr_value):
    """ATR-based stop-loss and take-profit price levels: 1.5x ATR stop,
    3x ATR target (a 2:1 reward-to-risk ratio)."""
    stop_distance = STOP_LOSS_ATR_MULTIPLE * atr_value
    target_distance = TAKE_PROFIT_ATR_MULTIPLE * atr_value

    if direction == "long":
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - target_distance

    return stop_price, take_profit_price, stop_distance


def calculate_position_size(balance, stop_distance_price_units, value_per_unit, min_units):
    """
    Risk-based position sizing:
        target_risk_dollars = balance * RISK_PER_TRADE_PCT
        size (units) = target_risk_dollars / (stop_distance * value_per_unit)

    Rounded DOWN to whole units (OANDA trades at 1-unit granularity). If
    even `min_units` would risk more than the target, returns 0 - the
    caller must treat 0 as "reject this trade", never open a smaller
    position than the risk budget allows.
    """
    target_risk_dollars = balance * RISK_PER_TRADE_PCT
    raw_size = target_risk_dollars / (stop_distance_price_units * value_per_unit)
    size = math.floor(raw_size)

    if size < min_units:
        return 0

    return size


# =============================================================================
# Pre-trade checks that aren't about account state (spread, stale data, news)
# =============================================================================

def spread_is_acceptable(current_spread, avg_spread_100, hard_cap):
    """Reject if the spread is abnormally wide vs. its own recent
    trailing average, or blows through a hard per-instrument cap (a
    backstop for the early backtest period before avg_spread_100 has
    enough history, and for data glitches)."""
    if avg_spread_100 is None or (isinstance(avg_spread_100, float) and math.isnan(avg_spread_100)):
        return False  # not enough history yet to judge "normal" - be conservative
    if current_spread > hard_cap:
        return False
    if current_spread > SPREAD_REJECT_MULTIPLE * avg_spread_100:
        return False
    return True


def data_is_stale(bar_timestamp, previous_bar_timestamp):
    """Reject if there's a suspiciously large gap since the last candle -
    could mean missing/stale data rather than a genuine market closure.
    Forex closes for the weekend, so a Friday-to-Sunday/Monday gap is
    allowed a much wider window than a same-week gap."""
    if previous_bar_timestamp is None:
        return False
    gap = bar_timestamp - previous_bar_timestamp
    spans_weekend = previous_bar_timestamp.weekday() == 4 and bar_timestamp.weekday() in (5, 6, 0)
    limit_hours = STALE_DATA_MAX_WEEKEND_GAP_HOURS if spans_weekend else STALE_DATA_MAX_GAP_HOURS
    return gap.total_seconds() / 3600 > limit_hours


def near_economic_announcement(timestamp):
    """See econ_calendar.py - this is an approximation, not a real
    calendar. Swap econ_calendar.is_blackout_window's implementation to
    plug in a real feed later without touching this file."""
    return is_blackout_window(timestamp)


# =============================================================================
# Position record + the shared portfolio account
# =============================================================================

@dataclass
class Position:
    symbol: str
    correlation_group: str
    direction: str          # "long" or "short"
    entry_time: object
    entry_price: float
    size_units: int
    stop_price: float
    take_profit_price: float
    risk_dollars: float     # $ actually at risk at entry (after rounding to whole units)
    risk_pct: float         # risk_dollars / balance-at-entry


class PortfolioAccount:
    """
    A single account shared across all instruments. The backtest engine
    drives this bar by bar: it's the one place that knows the true,
    combined state of the whole book, which is what lets it enforce
    portfolio-wide rules (max 3 positions total, 0.75% total open risk,
    one trade per correlation group, daily/weekly/drawdown circuit
    breakers) that a single-instrument backtest can't see.
    """

    def __init__(self, starting_cash):
        self.starting_cash = starting_cash
        self.balance = starting_cash          # realized only (see module docstring)
        self.peak_equity = starting_cash
        self.open_positions = {}              # symbol -> Position
        self.correlation_groups_open = set()
        self.closed_trades = []               # list of dicts, one per completed trade
        self.total_financing_paid = 0.0

        self.current_date = None
        self.current_week_key = None
        self.daily_start_equity = starting_cash
        self.weekly_start_equity = starting_cash
        self.daily_halted = False
        self.weekly_halted = False
        self.drawdown_suspended = False
        self.drawdown_suspended_since = None  # timestamp suspension started, for the cooldown fallback

        self.consecutive_losses = 0
        self.cooldown_until_date = None

        self.rejection_log = []               # list of (timestamp, symbol, reason)

    # --- Equity / mark-to-market -------------------------------------------

    def unrealized_pnl(self, position, current_price):
        direction_sign = 1 if position.direction == "long" else -1
        value_per_unit = value_per_price_unit(position.symbol, current_price)
        return (current_price - position.entry_price) * position.size_units * direction_sign * value_per_unit

    def equity(self, current_prices):
        """balance + unrealized P&L of every open position, marked to
        market using `current_prices` = {symbol: mid_price}. If a symbol
        isn't in current_prices (no bar yet at this timestamp), its
        position is valued at its entry price (i.e. contributes 0
        unrealized P&L) rather than crashing."""
        total = self.balance
        for symbol, pos in self.open_positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            total += self.unrealized_pnl(pos, price)
        return total

    # --- Day/week rollover and circuit breakers -----------------------------

    def handle_day_week_rollover(self, timestamp, current_equity):
        """Reset the daily/weekly loss counters at each new UTC calendar
        day/week, and clear a consecutive-loss cooldown once its day has
        passed. Must be called once per bar, in chronological order."""
        new_date = timestamp.date()
        if self.current_date != new_date:
            self.current_date = new_date
            self.daily_start_equity = current_equity
            self.daily_halted = False
            if self.cooldown_until_date is not None and new_date >= self.cooldown_until_date:
                self.cooldown_until_date = None
                self.consecutive_losses = 0

        week_key = timestamp.isocalendar()[:2]  # (ISO year, ISO week number)
        if self.current_week_key != week_key:
            self.current_week_key = week_key
            self.weekly_start_equity = current_equity
            self.weekly_halted = False

    def update_risk_flags(self, timestamp, current_equity):
        """Check the daily loss limit, weekly loss limit, and drawdown
        suspension against current equity. Call after handle_day_week_rollover
        (and after marking positions to market) at every bar."""
        if not self.daily_halted and self.daily_start_equity > 0:
            daily_change = (current_equity - self.daily_start_equity) / self.daily_start_equity
            if daily_change <= -DAILY_LOSS_LIMIT_PCT:
                self.daily_halted = True

        if not self.weekly_halted and self.weekly_start_equity > 0:
            weekly_change = (current_equity - self.weekly_start_equity) / self.weekly_start_equity
            if weekly_change <= -WEEKLY_LOSS_LIMIT_PCT:
                self.weekly_halted = True

        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity

            if not self.drawdown_suspended and drawdown >= DRAWDOWN_SUSPEND_PCT:
                self.drawdown_suspended = True
                self.drawdown_suspended_since = timestamp

            elif self.drawdown_suspended:
                # Resume on EITHER equity recovering back to within
                # DRAWDOWN_RESUME_PCT of peak (a lower threshold than the 8%
                # that triggered suspension - hysteresis, so it doesn't flip
                # every bar right at the boundary), OR a fixed cooldown
                # elapsing, whichever comes first. The cooldown fallback
                # exists because equity-recovery alone can never fire if
                # there's nothing left open to move equity (the only lever
                # is trading, and trading is exactly what's blocked) - a
                # real deadlock this strategy hit in practice.
                recovered = drawdown <= DRAWDOWN_RESUME_PCT
                cooldown_elapsed = (
                    self.drawdown_suspended_since is not None
                    and (timestamp - self.drawdown_suspended_since).days >= DRAWDOWN_SUSPEND_MAX_COOLDOWN_DAYS
                )
                if recovered or cooldown_elapsed:
                    self.drawdown_suspended = False
                    self.drawdown_suspended_since = None

    # --- Trade gating ---------------------------------------------------------

    def can_open_new_trade(self, symbol):
        """All the account-state gates a signal must clear before we even
        size it. Returns (True, None) or (False, reason_string)."""
        spec = INSTRUMENTS[symbol]

        if self.drawdown_suspended:
            return False, "drawdown_suspended"
        if self.daily_halted:
            return False, "daily_loss_limit"
        if self.weekly_halted:
            return False, "weekly_loss_limit"
        if self.cooldown_until_date is not None:
            return False, "consecutive_loss_cooldown"
        if symbol in self.open_positions:
            return False, "already_open"
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, "max_open_positions"
        if spec.correlation_group in self.correlation_groups_open:
            return False, "correlation_limit"

        current_open_risk_pct = sum(p.risk_pct for p in self.open_positions.values())
        if current_open_risk_pct + RISK_PER_TRADE_PCT > TOTAL_OPEN_RISK_CAP_PCT + 1e-9:
            return False, "total_open_risk_cap"

        return True, None

    def record_rejection(self, timestamp, symbol, reason):
        self.rejection_log.append({"time": timestamp, "symbol": symbol, "reason": reason})

    # --- Opening / closing positions -------------------------------------------

    def open_position(self, symbol, direction, entry_time, entry_price, size_units,
                       stop_price, take_profit_price):
        spec = INSTRUMENTS[symbol]
        value_per_unit = value_per_price_unit(symbol, entry_price)
        stop_distance = abs(entry_price - stop_price)
        risk_dollars = stop_distance * size_units * value_per_unit
        risk_pct = risk_dollars / self.balance if self.balance > 0 else 0.0

        position = Position(
            symbol=symbol, correlation_group=spec.correlation_group, direction=direction,
            entry_time=entry_time, entry_price=entry_price, size_units=size_units,
            stop_price=stop_price, take_profit_price=take_profit_price,
            risk_dollars=risk_dollars, risk_pct=risk_pct,
        )
        self.open_positions[symbol] = position
        self.correlation_groups_open.add(spec.correlation_group)
        return position

    def close_position(self, symbol, exit_time, exit_price, exit_reason, commission=0.0):
        position = self.open_positions.pop(symbol)
        self.correlation_groups_open.discard(position.correlation_group)

        pnl = self.unrealized_pnl(position, exit_price) - commission
        self.balance += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT and self.cooldown_until_date is None:
                # Pause new trades until the NEXT trading day.
                import datetime as dt
                self.cooldown_until_date = exit_time.date() + dt.timedelta(days=1)
        else:
            self.consecutive_losses = 0

        trade_record = {
            "symbol": symbol, "direction": position.direction,
            "entry_time": position.entry_time, "entry_price": position.entry_price,
            "exit_time": exit_time, "exit_price": exit_price,
            "size_units": position.size_units, "pnl": pnl,
            "risk_dollars": position.risk_dollars, "exit_reason": exit_reason,
        }
        self.closed_trades.append(trade_record)
        return trade_record

    def apply_financing(self, symbol, timestamp, notional_value, direction):
        """Overnight financing (swap) charge/credit - see instruments.py
        for the caveat that these are static, approximate annual rates,
        not real historical OANDA swap data. Applied once per calendar
        day a position is held through the rollover hour, tripled on
        Wednesdays (standard FX convention covering weekend settlement)."""
        spec = INSTRUMENTS[symbol]
        annual_rate_pct = spec.swap_long_pct_annual if direction == "long" else spec.swap_short_pct_annual
        multiplier = 3 if timestamp.weekday() == 2 else 1  # Wednesday
        charge = notional_value * (annual_rate_pct / 100) / 365 * multiplier
        self.balance += charge
        self.total_financing_paid += charge
        return charge
