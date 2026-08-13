"""
tests/test_gbp_position_sizing.py
------------------------------------
Rigorous tests for GBP-denominated position sizing - the account this
project's live trading connects to is GBP-denominated, not USD, and 3 of
the 4 instruments (EUR_USD, USD_JPY, XAU_USD) need a real cross-currency
conversion to size correctly against it (only GBP_USD has GBP as one of
its own two currencies). This is the most safety-critical piece of the
live-trading system - a sizing bug here means real (if demo) orders
sized wrong - so it gets the same rigor as the backtest's own validation.

Run with:
    source venv/bin/activate
    pip install -r requirements-dev.txt
    pytest tests/ -v

Prices used below are a real snapshot fetched from OANDA's practice
pricing feed (see the docstring on PRICES) - fixed rather than re-fetched
live, so these tests stay deterministic and reproducible over time. Every
expected value is derived BY HAND in comments, not just "call the
function and check it doesn't crash" - a wrong formula that's internally
consistent would still pass a test that only checks self-consistency.
"""

import math
import pytest

from instruments import INSTRUMENTS, value_per_price_unit, notional_value_per_unit, _find_conversion_rate
from risk_management import calculate_position_size, calculate_stop_and_target, RiskConfig

# Real OANDA practice-feed mid prices, captured 2026-08-13. Fixed here
# (not re-fetched) so these tests are deterministic - the exact numbers
# don't matter, only that they're realistic and internally consistent.
PRICES = {
    "EUR_USD": 1.152805,
    "GBP_USD": 1.34811,
    "USD_JPY": 159.389,
    "XAU_USD": 4377.37,
}


# =============================================================================
# Backward compatibility: the USD-account path (used throughout the
# already-validated backtest) must be COMPLETELY unaffected by this change.
# =============================================================================

def test_usd_account_eur_usd_unchanged():
    # quote (USD) == account currency -> 1.0, no cross-conversion ever reached
    assert value_per_price_unit("EUR_USD", PRICES["EUR_USD"], "USD") == 1.0


def test_usd_account_gbp_usd_unchanged():
    assert value_per_price_unit("GBP_USD", PRICES["GBP_USD"], "USD") == 1.0


def test_usd_account_xau_usd_unchanged():
    assert value_per_price_unit("XAU_USD", PRICES["XAU_USD"], "USD") == 1.0


def test_usd_account_usd_jpy_unchanged():
    # base (USD) == account currency -> 1/price
    expected = 1.0 / PRICES["USD_JPY"]
    assert value_per_price_unit("USD_JPY", PRICES["USD_JPY"], "USD") == pytest.approx(expected)


def test_usd_account_never_needs_current_prices():
    # None of the 4 instruments should ever hit the cross-currency branch
    # for a USD account - confirms the backtest's calls (which never pass
    # current_prices) are unaffected by this change existing at all.
    for symbol in INSTRUMENTS:
        price = PRICES[symbol]
        # Should not raise even though current_prices is omitted entirely.
        value_per_price_unit(symbol, price, "USD")


# =============================================================================
# GBP account: the direct case (GBP is one of the pair's own currencies)
# =============================================================================

def test_gbp_account_gbp_usd_direct():
    # base (GBP) == account currency -> 1/price, no cross-conversion needed
    expected = 1.0 / PRICES["GBP_USD"]
    result = value_per_price_unit("GBP_USD", PRICES["GBP_USD"], "GBP")
    assert result == pytest.approx(expected)
    # Sanity/smell test: GBP is worth more than 1 USD, so 1 USD of P&L
    # should convert to LESS than 1 GBP - catches an accidental inversion.
    assert 0.5 < result < 1.0


# =============================================================================
# GBP account: true cross-currency cases - neither side of the pair is GBP.
# Every expected value is derived by hand below, not just cross-checked
# against the function's own internal logic.
# =============================================================================

def test_gbp_account_eur_usd_cross():
    """
    EUR_USD: base=EUR, quote=USD, account=GBP - neither matches, so P&L
    (which happens in USD, the quote currency) must convert USD -> GBP.

    By hand: 1 unit (1 EUR notional) moving by 1.0 USD in price produces
    1.0 USD of P&L. Converting to GBP: rate(USD->GBP) = 1 / GBP_USD_price
    (since GBP_USD price IS "how many USD per 1 GBP", its reciprocal is
    "how many GBP per 1 USD").
    """
    expected = 1.0 / PRICES["GBP_USD"]
    result = value_per_price_unit(
        "EUR_USD", PRICES["EUR_USD"], account_currency="GBP", current_prices=PRICES
    )
    assert result == pytest.approx(expected)


def test_gbp_account_xau_usd_cross():
    """
    XAU_USD: base=XAU, quote=USD, account=GBP - same structure as
    EUR_USD above (quote is USD, needs USD -> GBP conversion).
    """
    expected = 1.0 / PRICES["GBP_USD"]
    result = value_per_price_unit(
        "XAU_USD", PRICES["XAU_USD"], account_currency="GBP", current_prices=PRICES
    )
    assert result == pytest.approx(expected)


def test_gbp_account_usd_jpy_cross_triangulated():
    """
    USD_JPY: base=USD, quote=JPY, account=GBP - neither matches, and this
    time the quote currency (JPY) isn't even ONE HOP from GBP; it has to
    triangulate JPY -> USD -> GBP.

    By hand: 1 unit (1 USD notional) moving by 1.0 JPY in price produces
    1.0 JPY of P&L.
      rate(JPY -> USD) = 1 / USD_JPY_price  ("USD_JPY price" = USD per JPY...
        no: USD_JPY price = JPY per 1 USD, so 1 JPY = 1/USD_JPY_price USD)
      rate(USD -> GBP) = 1 / GBP_USD_price
      rate(JPY -> GBP) = rate(JPY->USD) * rate(USD->GBP)
                        = (1 / USD_JPY_price) * (1 / GBP_USD_price)
    """
    expected = (1.0 / PRICES["USD_JPY"]) * (1.0 / PRICES["GBP_USD"])
    result = value_per_price_unit(
        "USD_JPY", PRICES["USD_JPY"], account_currency="GBP", current_prices=PRICES
    )
    assert result == pytest.approx(expected)
    # Smell test: JPY is a "small" currency (~159 JPY = 1 USD), so 1 JPY of
    # P&L should convert to a tiny fraction of a GBP, not something order-1.
    assert 0 < result < 0.01


def test_missing_current_prices_raises_clearly():
    """A cross-currency case with no current_prices supplied must raise,
    never silently return a wrong number (e.g. falling back to 1.0)."""
    with pytest.raises(NotImplementedError):
        value_per_price_unit("EUR_USD", PRICES["EUR_USD"], account_currency="GBP")


def test_missing_required_pair_raises_clearly():
    """If current_prices is supplied but doesn't actually contain what's
    needed to find or triangulate the rate, this must raise too."""
    with pytest.raises(ValueError):
        _find_conversion_rate("JPY", "GBP", current_prices={"EUR_USD": 1.10})


# =============================================================================
# Cross-check: conversion rates found different ways must agree with each
# other (round-trip consistency), not just look individually plausible.
# =============================================================================

def test_round_trip_usd_gbp_consistency():
    usd_to_gbp = _find_conversion_rate("USD", "GBP", PRICES)
    gbp_to_usd = _find_conversion_rate("GBP", "USD", PRICES)
    assert usd_to_gbp * gbp_to_usd == pytest.approx(1.0, rel=1e-9)


def test_triangulated_rate_matches_manual_two_step_conversion():
    """The JPY->GBP triangulation inside _find_conversion_rate must give
    the same answer as doing the two conversions manually, one at a time."""
    jpy_to_usd = _find_conversion_rate("JPY", "USD", PRICES)
    usd_to_gbp = _find_conversion_rate("USD", "GBP", PRICES)
    manual_two_step = jpy_to_usd * usd_to_gbp

    triangulated = _find_conversion_rate("JPY", "GBP", PRICES)
    assert triangulated == pytest.approx(manual_two_step, rel=1e-9)


# =============================================================================
# notional_value_per_unit: the leverage-cap helper. A DIFFERENT quantity
# from value_per_price_unit (P&L per price CHANGE) - conflating the two
# was a real, previously-shipped bug (see risk_management.py's
# calculate_position_size docstring). These tests exist specifically so
# that bug can never silently come back.
# =============================================================================

def test_notional_value_usd_account_quote_match():
    # EUR_USD, account=USD: quote(USD)==account -> notional = price directly
    assert notional_value_per_unit("EUR_USD", PRICES["EUR_USD"], "USD") == PRICES["EUR_USD"]


def test_notional_value_usd_account_base_match_regression():
    """
    THE BUG, pinned down as a regression test: USD_JPY on a USD account
    has base(USD)==account currency, so notional value per unit should be
    exactly 1.0 (independent of price) - NOT current_price, and NOT
    1/current_price. The old (buggy) leverage-cap formula effectively
    divided by current_price here, capping USD_JPY's allowed leverage at
    roughly 1/150th of the intended 30:1 - confirmed to have measurably
    under-sized real USD_JPY positions in the committed backtest history
    before this fix. This exact scenario is why the fix exists.
    """
    result = notional_value_per_unit("USD_JPY", PRICES["USD_JPY"], "USD")
    assert result == 1.0


def test_notional_value_gbp_account_base_match():
    # GBP_USD, account=GBP: base(GBP)==account -> notional = 1.0
    assert notional_value_per_unit("GBP_USD", PRICES["GBP_USD"], "GBP") == 1.0


def test_notional_value_gbp_account_cross_case():
    """
    EUR_USD, account=GBP: neither currency matches - 1 unit (1 EUR) is
    worth `price` USD, which then converts to GBP via GBP_USD.
    By hand: notional = EUR_USD_price * (1 / GBP_USD_price)
    """
    expected = PRICES["EUR_USD"] * (1.0 / PRICES["GBP_USD"])
    result = notional_value_per_unit(
        "EUR_USD", PRICES["EUR_USD"], account_currency="GBP", current_prices=PRICES
    )
    assert result == pytest.approx(expected)


def test_leverage_cap_now_gives_the_full_30x_for_usd_jpy_on_usd_account():
    """
    End-to-end confirmation that the fix actually changes the leverage
    cap's behavior the way it should: for a USD account with $10,000
    balance, the leverage-based ceiling for USD_JPY should now be
    max_leverage * balance directly (base currency matches account
    currency, so price doesn't enter into it at all) - not that value
    divided by the USD_JPY price.
    """
    cfg = RiskConfig()
    balance = 10_000.0
    notional = notional_value_per_unit("USD_JPY", PRICES["USD_JPY"], "USD")
    size_by_leverage = (cfg.max_leverage * balance) / notional
    assert size_by_leverage == pytest.approx(cfg.max_leverage * balance)  # == 300,000 units
    # The old, buggy formula would have given (30 * 10000) / 159.389 =~ 1,882 -
    # more than 150x smaller. Confirm we're nowhere near that.
    old_buggy_value = (cfg.max_leverage * balance) / PRICES["USD_JPY"]
    assert size_by_leverage > old_buggy_value * 100


# =============================================================================
# reference_balance_for_leverage: pinning the leverage cap to a FIXED
# balance rather than the (potentially inflated) current balance. This is
# the fix for the SECOND bug found in testing - fixing the leverage-cap
# math above correctly allowed full 30:1 leverage again, which re-exposed
# a runaway feedback loop (large notional -> large approximate swap
# financing -> inflated current balance -> an even bigger leverage cap on
# the next trade -> ...). One observed real consequence before this fix:
# a single USD_JPY trade sized at $21.8 MILLION notional, on an account
# that started with $10,000.
# =============================================================================

def test_leverage_cap_uses_reference_balance_not_current_balance():
    """
    The whole point of the fix: an inflated `balance` (e.g. from
    unrealistic compounding swap financing) must NOT inflate the
    leverage cap. Pass a huge `balance` but a small, fixed
    `reference_balance_for_leverage`, and confirm the leverage cap is
    computed from the SMALL reference, not the huge current balance.
    """
    cfg = RiskConfig()
    price = PRICES["USD_JPY"]
    atr = price * 0.02  # a fairly wide ATR, so risk-based sizing alone
                         # would want a large position too - we want the
                         # LEVERAGE cap to be the thing that binds here
    _, _, stop_distance = calculate_stop_and_target("long", price, atr, config=cfg)
    value_per_unit = value_per_price_unit("USD_JPY", price, "USD")
    notional_per_unit = notional_value_per_unit("USD_JPY", price, "USD")

    inflated_current_balance = 728_000.0  # mirrors the real runaway scenario observed
    fixed_reference_balance = 10_000.0    # the account's actual starting balance

    size = calculate_position_size(
        inflated_current_balance, stop_distance, value_per_unit, INSTRUMENTS["USD_JPY"].min_units,
        notional_per_unit, reference_balance_for_leverage=fixed_reference_balance, config=cfg,
    )

    max_notional_allowed = cfg.max_leverage * fixed_reference_balance  # 30 * 10,000 = 300,000
    actual_notional = size * notional_per_unit

    assert actual_notional <= max_notional_allowed + notional_per_unit  # +1 unit of rounding slack
    # The old (pre-fix) behavior would have used the inflated current
    # balance for the cap too: 30 * 728,000 = 21,840,000 - confirm we're
    # nowhere near that regardless of how large the current balance is.
    assert actual_notional < 21_000_000


def test_risk_based_sizing_still_scales_with_current_balance():
    """
    Confirms the fix didn't overcorrect: RISK-based sizing (the `balance`
    parameter) should still scale with the account's actual current
    balance - that's correct, intended compounding behavior, not the bug.
    Only the LEVERAGE cap should stay pinned to a fixed reference.
    """
    cfg = RiskConfig()
    price = PRICES["EUR_USD"]
    atr = price * 0.005  # tight ATR, so risk-based sizing (not leverage) binds
    _, _, stop_distance = calculate_stop_and_target("long", price, atr, config=cfg)
    value_per_unit = value_per_price_unit("EUR_USD", price, "USD")
    notional_per_unit = notional_value_per_unit("EUR_USD", price, "USD")

    small_size = calculate_position_size(
        10_000.0, stop_distance, value_per_unit, INSTRUMENTS["EUR_USD"].min_units,
        notional_per_unit, reference_balance_for_leverage=10_000.0, config=cfg,
    )
    larger_size = calculate_position_size(
        20_000.0, stop_distance, value_per_unit, INSTRUMENTS["EUR_USD"].min_units,
        notional_per_unit, reference_balance_for_leverage=10_000.0, config=cfg,  # reference UNCHANGED
    )

    # Risk-based sizing should roughly double when current balance doubles,
    # even though the leverage reference balance stayed fixed.
    assert larger_size == pytest.approx(small_size * 2, rel=0.01)


# =============================================================================
# End-to-end: full position sizing with a GBP account, for every
# instrument, checking the ACTUAL realized risk lands at the 0.25%
# target (within the tolerance introduced by flooring to whole units).
# =============================================================================

GBP_BALANCE = 100_000.0  # matches the real demo account's actual balance
RISK_CONFIG = RiskConfig()  # spec defaults: 0.25% risk, 1.5x/3x ATR, 30:1 leverage


@pytest.mark.parametrize("symbol", list(INSTRUMENTS.keys()))
def test_gbp_position_sizing_hits_target_risk(symbol):
    """
    For each instrument, size a position for a hypothetical long entry
    at the snapshot price with a plausible ATR, and confirm the ACTUAL
    GBP amount at risk (stop_distance * size * value_per_unit) is very
    close to the 0.25% target - not just "some positive number", but
    specifically within one unit's worth of rounding error from flooring.
    """
    price = PRICES[symbol]
    atr = price * 0.005  # a plausible ATR: 0.5% of price, illustrative

    stop_price, take_profit_price, stop_distance = calculate_stop_and_target(
        "long", price, atr, config=RISK_CONFIG
    )

    value_per_unit = value_per_price_unit(
        symbol, price, account_currency="GBP", current_prices=PRICES
    )
    notional_per_unit = notional_value_per_unit(
        symbol, price, account_currency="GBP", current_prices=PRICES
    )
    spec = INSTRUMENTS[symbol]
    size = calculate_position_size(
        GBP_BALANCE, stop_distance, value_per_unit, spec.min_units, notional_per_unit,
        reference_balance_for_leverage=GBP_BALANCE, config=RISK_CONFIG
    )

    assert size > 0, f"{symbol}: expected a nonzero position size for this balance/ATR"

    actual_risk_gbp = stop_distance * size * value_per_unit
    target_risk_gbp = GBP_BALANCE * RISK_CONFIG.risk_per_trade_pct

    # One whole unit's worth of risk is the maximum possible rounding
    # error from flooring `size` down to a whole number.
    one_unit_of_risk = stop_distance * value_per_unit
    assert actual_risk_gbp <= target_risk_gbp + 1e-6, (
        f"{symbol}: actual risk £{actual_risk_gbp:.4f} exceeds target £{target_risk_gbp:.4f}"
    )
    assert actual_risk_gbp >= target_risk_gbp - one_unit_of_risk - 1e-6, (
        f"{symbol}: actual risk £{actual_risk_gbp:.4f} is unexpectedly far below "
        f"target £{target_risk_gbp:.4f} (more than one unit's rounding error)"
    )


def test_gbp_leverage_cap_matches_real_oanda_margin_for_gold():
    """
    OANDA's real margin rate for XAU_USD is stricter than this project's
    own 30:1 policy cap (verified live: Gold's marginRate is 0.05, i.e.
    20:1, vs ~0.0333/30:1 for the FX pairs) - live execution additionally
    clamps to whichever is stricter (see run_live.py). This test just
    locks in the numbers that clamp is based on, so a future change to
    RiskConfig.max_leverage doesn't silently stop covering this case
    without a test failing to flag it.
    """
    assert RISK_CONFIG.max_leverage == 30.0
    real_gold_margin_rate = 0.05  # from a live AccountInstruments query - see live/oanda_live_client.py
    real_gold_max_leverage = 1.0 / real_gold_margin_rate
    assert real_gold_max_leverage < RISK_CONFIG.max_leverage, (
        "If this ever fails, OANDA's real Gold leverage limit is no longer "
        "stricter than our own cap - re-check whether the live per-instrument "
        "leverage clamp in run_live.py is still needed."
    )
