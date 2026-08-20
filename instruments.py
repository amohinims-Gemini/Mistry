"""
instruments.py
---------------
Central registry of everything instrument-specific: the OANDA symbol,
which currencies are involved (needed to convert price moves into
account-currency risk), a rough correlation grouping (for the "only one
USD-pair trade open at a time" safety rule), and a few backtest-realism
knobs (a hard spread cap, and approximate swap/financing rates).

Adding a new instrument later (e.g. oil) is just adding one more entry to
INSTRUMENTS below - nothing else in the project needs to change.
"""

from dataclasses import dataclass


@dataclass
class InstrumentSpec:
    oanda_symbol: str          # OANDA's symbol, e.g. "EUR_USD"
    base_currency: str         # e.g. "EUR"
    quote_currency: str        # e.g. "USD"
    pip_size: float            # smallest "pip" increment - for human-readable reporting only
    min_units: int             # smallest tradeable size OANDA allows (unit granularity)
    correlation_group: str     # only one open trade allowed per group at a time
    hard_spread_cap: float     # absolute spread (price units) above which we always reject -
                                # a backstop against illiquid hours / bad data, on top of the
                                # rolling-average spread check in risk_management.py

    # Approximate ANNUAL financing/swap rates (% of notional), charged or
    # credited for every night a position is held, depending on direction.
    # These are rough, STATIC ballpark figures used only so the backtest
    # can include *some* realistic overnight cost - they are NOT fetched
    # from OANDA and do NOT reflect how swap rates actually moved year to
    # year (e.g. near 0% in 2021 vs 5%+ in 2023-2024 as central bank rates
    # rose). See README for this limitation.
    swap_long_pct_annual: float
    swap_short_pct_annual: float


INSTRUMENTS = {
    "EUR_USD": InstrumentSpec(
        oanda_symbol="EUR_USD", base_currency="EUR", quote_currency="USD",
        pip_size=0.0001, min_units=1, correlation_group="usd_fx",
        hard_spread_cap=0.0010,
        swap_long_pct_annual=-1.0, swap_short_pct_annual=0.2,
    ),
    "GBP_USD": InstrumentSpec(
        oanda_symbol="GBP_USD", base_currency="GBP", quote_currency="USD",
        pip_size=0.0001, min_units=1, correlation_group="usd_fx",
        hard_spread_cap=0.0015,
        swap_long_pct_annual=-0.5, swap_short_pct_annual=-0.3,
    ),
    "USD_JPY": InstrumentSpec(
        oanda_symbol="USD_JPY", base_currency="USD", quote_currency="JPY",
        pip_size=0.01, min_units=1, correlation_group="usd_fx",
        hard_spread_cap=0.150,
        swap_long_pct_annual=2.5, swap_short_pct_annual=-3.5,
    ),
    "XAU_USD": InstrumentSpec(
        oanda_symbol="XAU_USD", base_currency="XAU", quote_currency="USD",
        pip_size=0.01, min_units=1, correlation_group="gold",
        hard_spread_cap=0.80,
        swap_long_pct_annual=-4.0, swap_short_pct_annual=1.0,
    ),
    "AUD_USD": InstrumentSpec(
        oanda_symbol="AUD_USD", base_currency="AUD", quote_currency="USD",
        pip_size=0.0001, min_units=1, correlation_group="usd_fx",
        hard_spread_cap=0.0015,
        swap_long_pct_annual=-1.0, swap_short_pct_annual=0.1,
    ),
    # To add oil later: e.g. "BCO_USD" (Brent) or "WTICO_USD" (WTI) -
    # quote_currency="USD" so the pip-value math below already handles it
    # for free. Give it its own correlation_group (e.g. "oil") unless you
    # deliberately want it sharing a group with something else.
}

# The "official" multi-instrument portfolio every combined backtest
# script (run_backtest.py, run_backtest_4h.py, etc.) trades - kept
# separate from INSTRUMENTS itself so a new instrument can be added to
# the registry (e.g. for a standalone new-currency-pair check, like
# AUD_USD) WITHOUT silently pulling it into every existing portfolio-
# level script the moment it's added. Add a symbol here explicitly only
# once it's actually meant to join the main portfolio.
PORTFOLIO_SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]


def value_per_price_unit(instrument_symbol, current_price, account_currency="USD", current_prices=None):
    """
    How much account-currency value ONE UNIT of this instrument gains or
    loses for a 1.0 move in its quoted price. risk_management.py uses
    this to turn a stop distance (in price units) into an account-
    currency risk amount for position sizing.

    - If the instrument is quoted directly in the account currency
      (e.g. EUR/USD against a USD account), each unit's value changes
      1-for-1 with price: value = 1.
    - If the account currency is the BASE currency instead (e.g. USD/JPY
      against a USD account), P&L happens in the quote currency and has
      to be converted back using the current exchange rate: value = 1/price.
    - TRUE CROSS case: neither side is the account currency (e.g. a GBP
      account trading EUR/USD - GBP is neither EUR nor USD). P&L happens
      in the quote currency and needs a further conversion into the
      account currency, via `_find_conversion_rate`. This requires
      `current_prices` - a dict of {oanda_symbol: price} covering
      whatever pairs are needed to find or triangulate that rate (e.g.
      {"GBP_USD": 1.27} to convert USD into GBP). Raises if it's needed
      but not supplied - never silently returns a wrong number.

    Backward-compatibility note: for a USD account, every instrument in
    this project's INSTRUMENTS registry has USD as either its base or
    quote currency, so the cross-currency branch is never reached and
    `current_prices` is never needed - this generalization is purely
    additive for other account currencies and does not change the
    already-validated USD-account backtest's behavior.
    """
    spec = INSTRUMENTS[instrument_symbol]
    if spec.quote_currency == account_currency:
        return 1.0
    if spec.base_currency == account_currency:
        return 1.0 / current_price

    if current_prices is None:
        raise NotImplementedError(
            f"{instrument_symbol}: value_per_price_unit needs `current_prices` to "
            f"convert {spec.quote_currency} into {account_currency} - neither "
            f"currency of this instrument matches the account currency."
        )
    return _find_conversion_rate(spec.quote_currency, account_currency, current_prices)


def _find_conversion_rate(from_currency, to_currency, current_prices):
    """
    Find the rate to convert 1 unit of `from_currency` into `to_currency`,
    using whatever instrument prices are available in `current_prices`
    (a dict of {oanda_symbol: price}, e.g. {"GBP_USD": 1.27}). Tries a
    direct pair, then its inverse, then triangulates through USD (e.g.
    JPY -> GBP via JPY -> USD -> GBP, using USD_JPY and GBP_USD).

    Recursion terminates after at most one extra hop: the two recursive
    calls are always (X, "USD") and ("USD", Y), and on the next level
    down at least one side is already "USD", which skips the
    triangulation branch entirely.
    """
    if from_currency == to_currency:
        return 1.0

    direct = f"{from_currency}_{to_currency}"
    if direct in current_prices:
        return current_prices[direct]

    inverse = f"{to_currency}_{from_currency}"
    if inverse in current_prices:
        return 1.0 / current_prices[inverse]

    if from_currency != "USD" and to_currency != "USD":
        rate_from_to_usd = _find_conversion_rate(from_currency, "USD", current_prices)
        rate_usd_to_to = _find_conversion_rate("USD", to_currency, current_prices)
        return rate_from_to_usd * rate_usd_to_to

    raise ValueError(
        f"No conversion rate available from {from_currency} to {to_currency} "
        f"given prices for: {list(current_prices.keys())}"
    )


def notional_value_per_unit(instrument_symbol, current_price, account_currency="USD", current_prices=None):
    """
    How much account-currency NOTIONAL VALUE one unit (of the base
    currency) of this instrument represents at the current price. Used
    for the leverage/margin cap in position sizing: "how many units can
    max_leverage times my balance afford at today's price".

    This is a DIFFERENT question from value_per_price_unit (which is
    P&L per price CHANGE, a marginal/derivative quantity) - conflating
    the two was a real, previously-shipped bug. The old leverage-cap
    formula divided balance by the raw instrument price unconditionally,
    which only happens to be correct when the account currency matches
    the pair's QUOTE currency. For USD_JPY on a USD account, the account
    currency matches the BASE currency instead, and dividing by the
    price (~150) under-capped the leverage limit by roughly that same
    factor - confirmed to have measurably under-sized USD_JPY positions
    throughout the whole backtest history before this fix.

    - If account currency == BASE currency, 1 unit of base currency IS 1
      unit of account currency, independent of price: notional = 1.
    - If account currency == QUOTE currency, 1 unit of base currency is
      worth `current_price` units of account currency: notional = price.
    - Cross case: 1 unit of base currency is worth `current_price` units
      of quote currency, which then needs converting into the account
      currency - same `current_prices` mechanism as value_per_price_unit.
    """
    spec = INSTRUMENTS[instrument_symbol]
    if spec.base_currency == account_currency:
        return 1.0
    if spec.quote_currency == account_currency:
        return current_price

    if current_prices is None:
        raise NotImplementedError(
            f"{instrument_symbol}: notional_value_per_unit needs `current_prices` to "
            f"convert {spec.quote_currency} into {account_currency} - neither "
            f"currency of this instrument matches the account currency."
        )
    conversion_rate = _find_conversion_rate(spec.quote_currency, account_currency, current_prices)
    return current_price * conversion_rate
