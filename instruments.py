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
    # To add oil later: e.g. "BCO_USD" (Brent) or "WTICO_USD" (WTI) -
    # quote_currency="USD" so the pip-value math below already handles it
    # for free. Give it its own correlation_group (e.g. "oil") unless you
    # deliberately want it sharing a group with something else.
}


def value_per_price_unit(instrument_symbol, current_price, account_currency="USD"):
    """
    How much account-currency value ONE UNIT of this instrument gains or
    loses for a 1.0 move in its quoted price. risk_management.py uses
    this to turn a stop distance (in price units) into a dollar risk
    amount for position sizing.

    - If the instrument is quoted directly in the account currency
      (EUR/USD, GBP/USD, XAU/USD, all against a USD account), each unit's
      value changes 1-for-1 with price: value = 1.
    - If the account currency is the BASE currency instead (USD/JPY with
      a USD account), P&L happens in the quote currency (JPY) and has to
      be converted back to USD using the current exchange rate:
      value = 1 / price.
    - True cross pairs, where neither side is the account currency,
      aren't supported yet - none of the current instruments hit this case.
    """
    spec = INSTRUMENTS[instrument_symbol]
    if spec.quote_currency == account_currency:
        return 1.0
    if spec.base_currency == account_currency:
        return 1.0 / current_price
    raise NotImplementedError(
        f"{instrument_symbol}: value_per_price_unit doesn't support cross pairs "
        f"where neither currency is the account currency ({account_currency})."
    )
