"""
risk_management.py
--------------------
Everything that decides whether a technically-valid signal (from
signals.py) actually gets traded, and how big that trade is:

  - Position sizing: risk a fixed % of account BALANCE per trade, based
    on the ATR-derived stop distance and the instrument's pip/point value.
  - ATR-based stop-loss / take-profit levels.
  - `PortfolioAccount`: a stateful object the backtest engine drives bar
    by bar, tracking a SHARED account across all instruments and
    enforcing every safety limit from the spec: daily/weekly loss halts,
    drawdown suspension, consecutive-loss cooldown, max open positions,
    per-correlation-group exclusivity, total open risk cap, spread and
    staleness checks, and overnight financing (swap) costs.

All the tunable numbers (risk %, ATR multiples, safety-limit thresholds)
live in `RiskConfig` rather than as bare module constants, so
run_backtest.py can run several parameter variations back to back in one
process without one run's settings leaking into another's.

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
import datetime as dt
from dataclasses import dataclass, field

from instruments import INSTRUMENTS, value_per_price_unit
from econ_calendar import is_blackout_window


# =============================================================================
# Tunable configuration
# =============================================================================

@dataclass
class RiskConfig:
    """Every tunable number in the strategy's risk management, in one
    place. Defaults match the original spec; pass a modified RiskConfig
    into run_backtest()/PortfolioAccount to try a variation without
    touching any other code."""

    risk_per_trade_pct: float = 0.0025          # risk 0.25% of balance per trade
    stop_loss_atr_multiple: float = 1.5
    take_profit_atr_multiple: float = 3.0        # 2:1 reward-to-risk at the defaults

    max_open_positions: int = 3
    total_open_risk_cap_pct: float = 0.0075      # 0.75%
    daily_loss_limit_pct: float = 0.01           # stop for the day after a 1% loss
    weekly_loss_limit_pct: float = 0.025         # stop for the week after a 2.5% loss
    drawdown_suspend_pct: float = 0.08           # suspend ALL trading after this much drawdown
    drawdown_resume_pct: float = 0.04            # resume once drawdown recovers to within this
    drawdown_suspend_max_cooldown_days: int = 30 # ...or after this many days, whichever first
    consecutive_loss_limit: int = 3              # pause until next trading day after N losses in a row

    spread_reject_multiple: float = 2.5          # reject if spread > this x its trailing average
    stale_data_max_gap_hours: float = 3
    stale_data_max_weekend_gap_hours: float = 72

    max_leverage: float = 30.0                   # cap notional exposure at this x balance - a
                                                   # real broker margin limit the risk-per-trade
                                                   # formula alone doesn't account for. Without it,
                                                   # a very tight stop (low-ATR period) can produce
                                                   # unrealistically huge position sizes; found via
                                                   # stress-testing when a runaway swap-financing
                                                   # feedback loop showed up in one test window
                                                   # (huge notional -> huge financing -> inflated
                                                   # balance -> even huger next position). 30:1 is a
                                                   # common retail FX leverage limit (e.g. ESMA).

    # Trade management - OFF by default (spec-original behavior: fixed stop,
    # never moved). Backed by a diagnosed failure pattern, not a hypothesis:
    # the worst historical drawdown episode was ~200 uniform ~-1R losses
    # stacking up over months during a choppy-but-technically-trending
    # regime (13-trade losing streak at the worst point). Moving the stop
    # to breakeven once a trade shows some initial favorable movement
    # converts some of those eventual full losses into scratches, without
    # touching entry logic at all. Deliberately ONE new free parameter
    # (the trigger threshold) - see README.md for why searching over
    # multiple simultaneous parameters is a real risk, not paranoia.
    use_breakeven_stop: bool = False
    breakeven_trigger_r_multiple: float = 1.0    # move stop to breakeven once profit reaches
                                                   # this many multiples of the original stop distance


DEFAULT_CONFIG = RiskConfig()


# =============================================================================
# Position sizing and stop/target calculation
# =============================================================================

def calculate_stop_and_target(direction, entry_price, atr_value, config=DEFAULT_CONFIG):
    """ATR-based stop-loss and take-profit price levels."""
    stop_distance = config.stop_loss_atr_multiple * atr_value
    target_distance = config.take_profit_atr_multiple * atr_value

    if direction == "long":
        stop_price = entry_price - stop_distance
        take_profit_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        take_profit_price = entry_price - target_distance

    return stop_price, take_profit_price, stop_distance


def calculate_position_size(balance, stop_distance_price_units, value_per_unit, min_units,
                             notional_value_per_unit, reference_balance_for_leverage, config=DEFAULT_CONFIG):
    """
    Risk-based position sizing, capped at a realistic broker leverage limit:
        size_by_risk     = (balance * risk_per_trade_pct) / (stop_distance * value_per_unit)
        size_by_leverage = (config.max_leverage * reference_balance_for_leverage) / notional_value_per_unit
        size = floor(min(size_by_risk, size_by_leverage))

    The leverage cap matters more than it might look: during a very
    low-volatility stretch, ATR (and so stop_distance) can get tiny,
    which would otherwise blow the risk-based size up to an unrealistic
    size with no real-world margin constraint stopping it - exactly what
    happened in stress-testing before this cap was added.

    `notional_value_per_unit` (see instruments.notional_value_per_unit)
    is account-currency notional value per unit AT the current price -
    NOT the same thing as `value_per_unit` (P&L per price CHANGE).
    Conflating the two here was a real, previously-shipped bug: the
    formula used to divide balance by the raw instrument price
    unconditionally, which is only correct when the account currency
    matches the pair's quote currency. For USD_JPY on a USD account
    (where the account currency matches the BASE currency instead), that
    under-capped the leverage limit by roughly the USD_JPY price itself
    (~150x too restrictive) - confirmed to have measurably under-sized
    USD_JPY positions throughout the backtest history before this fix.

    `balance` vs `reference_balance_for_leverage` - DELIBERATELY separate,
    not a naming accident:
      - `balance` (risk-based sizing) SHOULD scale with the current
        account balance - risking a constant % of a growing account is
        correct, standard risk management.
      - `reference_balance_for_leverage` should NOT scale with current
        balance. Fixing the leverage-cap bug above exposed a second,
        distinct problem in testing: this project's swap-financing model
        is a static, approximate annual rate (see instruments.py), and
        once large leveraged notional became possible again, a real
        feedback loop appeared - large notional -> large approximate
        swap financing -> inflated balance -> an even bigger leverage cap
        on the next trade (since it scaled with that inflated balance)
        -> runaway compounding, producing results nowhere close to
        plausible. Passing a FIXED reference balance (the account's
        starting balance, never its current one - see call sites) for
        the leverage cap specifically breaks that loop while leaving
        risk-based sizing's intended compounding untouched.

    No default for `reference_balance_for_leverage` - every caller must
    consciously choose it, so this can't silently regress to using
    current balance again by a future call site forgetting to think
    about it.

    Rounded DOWN to whole units (OANDA trades at 1-unit granularity). If
    even `min_units` would exceed the risk budget OR the leverage cap,
    returns 0 - the caller must treat 0 as "reject this trade", never
    open a smaller position than either constraint allows.
    """
    target_risk_dollars = balance * config.risk_per_trade_pct
    size_by_risk = target_risk_dollars / (stop_distance_price_units * value_per_unit)
    size_by_leverage = (config.max_leverage * reference_balance_for_leverage) / notional_value_per_unit

    size = math.floor(min(size_by_risk, size_by_leverage))

    if size < min_units:
        return 0

    return size


# =============================================================================
# Pre-trade checks that aren't about account state (spread, stale data, news)
# =============================================================================

def spread_is_acceptable(current_spread, avg_spread_100, hard_cap, config=DEFAULT_CONFIG):
    """Reject if the spread is abnormally wide vs. its own recent
    trailing average, or blows through a hard per-instrument cap (a
    backstop for the early backtest period before avg_spread_100 has
    enough history, and for data glitches)."""
    if avg_spread_100 is None or (isinstance(avg_spread_100, float) and math.isnan(avg_spread_100)):
        return False  # not enough history yet to judge "normal" - be conservative
    if current_spread > hard_cap:
        return False
    if current_spread > config.spread_reject_multiple * avg_spread_100:
        return False
    return True


def data_is_stale(bar_timestamp, previous_bar_timestamp, config=DEFAULT_CONFIG):
    """Reject if there's a suspiciously large gap since the last candle -
    could mean missing/stale data rather than a genuine market closure.
    Forex closes for the weekend, so a Friday-to-Sunday/Monday gap is
    allowed a much wider window than a same-week gap."""
    if previous_bar_timestamp is None:
        return False
    gap = bar_timestamp - previous_bar_timestamp
    # previous_bar.weekday() == 4 (Friday) is correct for 1H/4H bars, whose
    # last pre-weekend candle opens Friday. Daily bars are the OPEN-time-
    # labeled exception: OANDA's daily candle boundary is 21:00 UTC, so the
    # candle covering the Friday session actually opens Thursday 21:00 UTC
    # (weekday() == 3) - found while adding daily-timeframe support,
    # because that mismatch meant every weekly weekend gap on daily bars
    # was being judged against the NORMAL (not weekend) threshold and
    # wrongly flagged as stale, every single week. Checking both weekdays
    # is safe for 1H/4H too - a real stale gap starting mid-week is still
    # caught by the normal threshold regardless of which weekday label the
    # weekend check itself uses.
    spans_weekend = previous_bar_timestamp.weekday() in (3, 4) and bar_timestamp.weekday() in (5, 6, 0)
    limit_hours = config.stale_data_max_weekend_gap_hours if spans_weekend else config.stale_data_max_gap_hours
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
    breakeven_triggered: bool = False  # has the stop already been ratcheted to breakeven?
    signal_source: str = "unknown"     # which strategy generated this entry - only meaningful
                                         # for combined multi-strategy frames (see
                                         # combined_signals_4h.py); single-strategy backtests
                                         # just leave this at the default.


def apply_breakeven_if_triggered(position, bid_high, ask_low, config=DEFAULT_CONFIG):
    """
    If `position` has moved favorably by `breakeven_trigger_r_multiple`
    times its ORIGINAL stop distance, move its stop to breakeven (entry
    price) - a one-way ratchet, never moved back once triggered. Mutates
    `position.stop_price` in place and returns True if it just moved this
    call, False otherwise (already triggered, feature off, or not reached).

    Uses the EXIT side of the market for the trigger check (Bid for a
    long, since exiting a long means selling; Ask for a short) - the same
    side already used for checking stop/target hits, for consistency.

    Caller's responsibility: call this AFTER checking stop/target hits for
    the current bar, not before - a breakeven adjustment made this bar
    should only affect stop-checking in FUTURE bars, never retroactively
    change the outcome of the bar it triggered on. OHLC data can't tell us
    the true intra-bar order of "price reached the trigger" vs "price hit
    the original stop", so assuming the trigger happened first within the
    same bar would be an unsupported, optimistic assumption - the same
    reasoning already applied to the stop-vs-target tie-break elsewhere.
    """
    if not config.use_breakeven_stop or position.breakeven_triggered:
        return False

    stop_distance = abs(position.entry_price - position.stop_price)
    trigger_distance = config.breakeven_trigger_r_multiple * stop_distance

    if position.direction == "long":
        trigger_price = position.entry_price + trigger_distance
        reached = bid_high >= trigger_price
    else:
        trigger_price = position.entry_price - trigger_distance
        reached = ask_low <= trigger_price

    if reached:
        position.stop_price = position.entry_price
        position.breakeven_triggered = True
        return True

    return False


class PortfolioAccount:
    """
    A single account shared across all instruments. The backtest engine
    drives this bar by bar: it's the one place that knows the true,
    combined state of the whole book, which is what lets it enforce
    portfolio-wide rules (max open positions, total open risk, one trade
    per correlation group, daily/weekly/drawdown circuit breakers) that a
    single-instrument backtest can't see.
    """

    def __init__(self, starting_cash, config=DEFAULT_CONFIG):
        self.config = config
        self.starting_cash = starting_cash
        self.balance = starting_cash          # realized only (see module docstring)
        self.peak_equity = starting_cash
        self.open_positions = {}              # symbol -> Position
        self.correlation_groups_open = set()
        self.closed_trades = []               # list of dicts, one per completed trade
        self.total_financing_paid = 0.0

        # Trading-only balance: mirrors `balance` but is NEVER touched by
        # apply_financing() - only by real trade P&L via close_position().
        # Exists purely for reporting/scoring (run_backtest.py's pass/fail
        # metrics use this), because this project's swap-financing model
        # is a static, approximate annual rate (see instruments.py), not
        # real historical OANDA data. Confirmed in testing that once
        # leverage was fixed to allow realistic position sizes, financing
        # under that approximation could reach ~30x the strategy's actual
        # trading P&L - reporting a "total return" that's 97% financing
        # assumption would be scoring the wrong thing. `balance` itself
        # (financing included) is still what position sizing and the
        # safety-limit circuit breakers use - that's real capital, and
        # should reflect the true account state during actual operation.
        # Only the SCORING criteria change, not the account's real behavior.
        self.trading_only_balance = starting_cash

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

    def trading_only_equity(self, current_prices):
        """Same as equity(), but built from trading_only_balance instead
        of balance - i.e. with financing excluded. Unrealized P&L (price
        movement on still-open positions) is never financing anyway, so
        it's added the same way in both."""
        total = self.trading_only_balance
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
        cfg = self.config

        if not self.daily_halted and self.daily_start_equity > 0:
            daily_change = (current_equity - self.daily_start_equity) / self.daily_start_equity
            if daily_change <= -cfg.daily_loss_limit_pct:
                self.daily_halted = True

        if not self.weekly_halted and self.weekly_start_equity > 0:
            weekly_change = (current_equity - self.weekly_start_equity) / self.weekly_start_equity
            if weekly_change <= -cfg.weekly_loss_limit_pct:
                self.weekly_halted = True

        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity

            if not self.drawdown_suspended and drawdown >= cfg.drawdown_suspend_pct:
                self.drawdown_suspended = True
                self.drawdown_suspended_since = timestamp

            elif self.drawdown_suspended:
                # Resume on EITHER equity recovering back to within
                # drawdown_resume_pct of peak (a lower threshold than the
                # one that triggered suspension - hysteresis, so it
                # doesn't flip every bar right at the boundary), OR the
                # fixed cooldown elapsing, whichever comes first. The
                # cooldown fallback exists because equity-recovery alone
                # can never fire if there's nothing left open to move
                # equity (the only lever is trading, and trading is
                # exactly what's blocked) - a real deadlock this strategy
                # hit in practice with the original permanent-halt design.
                recovered = drawdown <= cfg.drawdown_resume_pct
                cooldown_elapsed = (
                    self.drawdown_suspended_since is not None
                    and (timestamp - self.drawdown_suspended_since).days >= cfg.drawdown_suspend_max_cooldown_days
                )
                if recovered or cooldown_elapsed:
                    self.drawdown_suspended = False
                    self.drawdown_suspended_since = None

    # --- Trade gating ---------------------------------------------------------

    def can_open_new_trade(self, symbol):
        """All the account-state gates a signal must clear before we even
        size it. Returns (True, None) or (False, reason_string)."""
        cfg = self.config
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
        if len(self.open_positions) >= cfg.max_open_positions:
            return False, "max_open_positions"
        if spec.correlation_group in self.correlation_groups_open:
            return False, "correlation_limit"

        current_open_risk_pct = sum(p.risk_pct for p in self.open_positions.values())
        if current_open_risk_pct + cfg.risk_per_trade_pct > cfg.total_open_risk_cap_pct + 1e-9:
            return False, "total_open_risk_cap"

        return True, None

    def record_rejection(self, timestamp, symbol, reason):
        self.rejection_log.append({"time": timestamp, "symbol": symbol, "reason": reason})

    # --- Opening / closing positions -------------------------------------------

    def open_position(self, symbol, direction, entry_time, entry_price, size_units,
                       stop_price, take_profit_price, signal_source="unknown"):
        spec = INSTRUMENTS[symbol]
        value_per_unit = value_per_price_unit(symbol, entry_price)
        stop_distance = abs(entry_price - stop_price)
        risk_dollars = stop_distance * size_units * value_per_unit
        risk_pct = risk_dollars / self.balance if self.balance > 0 else 0.0

        position = Position(
            symbol=symbol, correlation_group=spec.correlation_group, direction=direction,
            entry_time=entry_time, entry_price=entry_price, size_units=size_units,
            stop_price=stop_price, take_profit_price=take_profit_price,
            risk_dollars=risk_dollars, risk_pct=risk_pct, signal_source=signal_source,
        )
        self.open_positions[symbol] = position
        self.correlation_groups_open.add(spec.correlation_group)
        return position

    def close_position(self, symbol, exit_time, exit_price, exit_reason, commission=0.0):
        cfg = self.config
        position = self.open_positions.pop(symbol)
        self.correlation_groups_open.discard(position.correlation_group)

        pnl = self.unrealized_pnl(position, exit_price) - commission
        self.balance += pnl
        self.trading_only_balance += pnl  # same pnl - this never included financing anyway

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= cfg.consecutive_loss_limit and self.cooldown_until_date is None:
                self.cooldown_until_date = exit_time.date() + dt.timedelta(days=1)
        else:
            self.consecutive_losses = 0

        trade_record = {
            "symbol": symbol, "direction": position.direction,
            "entry_time": position.entry_time, "entry_price": position.entry_price,
            "exit_time": exit_time, "exit_price": exit_price,
            "size_units": position.size_units, "pnl": pnl,
            "risk_dollars": position.risk_dollars, "exit_reason": exit_reason,
            "signal_source": position.signal_source,
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
