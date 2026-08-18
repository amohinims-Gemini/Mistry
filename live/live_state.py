"""
live_state.py
----------------
Tracks everything the daily/weekly loss limits, drawdown suspension, and
consecutive-loss cooldown need that OANDA doesn't hand us directly -
persisted locally so a restart doesn't lose the day's starting equity or
the all-time equity peak.

Deliberately does NOT track balance/NAV itself - OANDA's own
AccountSummary (see oanda_live_client.get_account_summary, wrapped by
live_account_sync.get_account_balances) is always the source of truth
for real money figures; this class only tracks the STATE MACHINE around
them (what day/week is it, has a circuit breaker tripped, when does a
cooldown end).

Shares its day/week-rollover and drawdown-suspension LOGIC with the
backtest's PortfolioAccount via risk_management.advance_day_week_rollover
/advance_risk_flags - both classes provide the same attribute names on
self and pass `self` to those shared functions, so there is exactly one
implementation of "how the circuit breakers transition," not two copies
that could quietly drift apart between backtest and live.
"""

import os
import sys
import json
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_management import advance_day_week_rollover, advance_risk_flags, DEFAULT_CONFIG

STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_data")
STATE_PATH = os.path.join(STATE_DIR, "state.json")


class LiveRiskState:
    """Same attribute names as PortfolioAccount's own risk-state fields
    (risk_management.py), deliberately - see this module's docstring."""

    def __init__(self, starting_peak_equity):
        self.peak_equity = starting_peak_equity
        self.current_date = None
        self.current_week_key = None
        self.daily_start_equity = starting_peak_equity
        self.weekly_start_equity = starting_peak_equity
        self.daily_halted = False
        self.weekly_halted = False
        self.drawdown_suspended = False
        self.drawdown_suspended_since = None
        self.cooldown_until_date = None
        self.consecutive_losses = 0

    def update(self, timestamp, current_equity, config=DEFAULT_CONFIG):
        """Call once per cycle, after fetching real NAV from OANDA and
        (separately) refreshing consecutive_losses from real trade
        history - see live_account_sync.get_fresh_consecutive_losses."""
        advance_day_week_rollover(self, timestamp, current_equity)
        advance_risk_flags(self, timestamp, current_equity, config=config)

    # --- Persistence -----------------------------------------------------

    def to_dict(self):
        return {
            "peak_equity": self.peak_equity,
            "current_date": self.current_date.isoformat() if self.current_date else None,
            "current_week_key": list(self.current_week_key) if self.current_week_key else None,
            "daily_start_equity": self.daily_start_equity,
            "weekly_start_equity": self.weekly_start_equity,
            "daily_halted": self.daily_halted,
            "weekly_halted": self.weekly_halted,
            "drawdown_suspended": self.drawdown_suspended,
            "drawdown_suspended_since": (
                self.drawdown_suspended_since.isoformat() if self.drawdown_suspended_since else None
            ),
            "cooldown_until_date": self.cooldown_until_date.isoformat() if self.cooldown_until_date else None,
            "consecutive_losses": self.consecutive_losses,
        }

    @classmethod
    def from_dict(cls, d):
        state = cls(starting_peak_equity=d["peak_equity"])
        state.current_date = dt.date.fromisoformat(d["current_date"]) if d["current_date"] else None
        state.current_week_key = tuple(d["current_week_key"]) if d["current_week_key"] else None
        state.daily_start_equity = d["daily_start_equity"]
        state.weekly_start_equity = d["weekly_start_equity"]
        state.daily_halted = d["daily_halted"]
        state.weekly_halted = d["weekly_halted"]
        state.drawdown_suspended = d["drawdown_suspended"]
        state.drawdown_suspended_since = (
            dt.datetime.fromisoformat(d["drawdown_suspended_since"]) if d["drawdown_suspended_since"] else None
        )
        state.cooldown_until_date = (
            dt.date.fromisoformat(d["cooldown_until_date"]) if d["cooldown_until_date"] else None
        )
        state.consecutive_losses = d["consecutive_losses"]
        return state

    def save(self, path=STATE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_or_create(cls, starting_equity, path=STATE_PATH):
        """Load persisted state if it exists, otherwise start fresh with
        `starting_equity` as the initial peak - the account's real NAV at
        first startup, NOT the backtest's fixed starting_cash concept
        (a live demo account can have any real balance already on it)."""
        if os.path.exists(path):
            with open(path) as f:
                return cls.from_dict(json.load(f))
        return cls(starting_peak_equity=starting_equity)
