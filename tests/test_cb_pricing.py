from datetime import date

import pytest

from convertible_tool.pricing import (ConvertibleBond, MarketInputs, analyse, bond_floor,
                                      implied_volatility, parity, tf_price, accrued_interest)

VAL = date(2026, 1, 1)


def cb(**kw):
    base = dict(id="X", issuer="Test", stock_ticker="X", maturity=date(2031, 1, 1),
                coupon=1.0, conversion_price=50.0, credit_spread_bp=200)
    base.update(kw)
    return ConvertibleBond(**base)


def mkt(S, vol=0.3):
    return MarketInputs(stock_price=S, volatility=vol, risk_free=0.02, valuation_date=VAL)


def test_value_above_parity_and_floor():
    b = cb()
    for S in (10, 40, 50, 60, 120):
        v = tf_price(b, mkt(S))
        assert v >= parity(b, S) - 1e-6
        assert v >= bond_floor(b, mkt(S)) - 0.05


def test_deep_itm_converges_to_parity():
    # mit Dividende > Kuponrendite wird sofort gewandelt
    b = cb(dividend_yield=0.05)
    assert tf_price(b, mkt(500)) == pytest.approx(parity(b, 500), rel=1e-3)


def test_deep_otm_converges_to_floor():
    b = cb()
    assert tf_price(b, mkt(1)) == pytest.approx(bond_floor(b, mkt(1)), abs=0.3)


def test_value_increases_with_vol():
    b = cb()
    assert tf_price(b, mkt(50, 0.4)) > tf_price(b, mkt(50, 0.2))


def test_call_caps_value():
    assert tf_price(cb(call_date=VAL, call_price=100), mkt(45)) <= tf_price(cb(), mkt(45))


def test_put_floors_value():
    b = cb(coupon=0, credit_spread_bp=1500, put_date=date(2027, 1, 1), put_price=100)
    assert tf_price(b, mkt(5)) > tf_price(cb(coupon=0, credit_spread_bp=1500), mkt(5))


def test_implied_vol_roundtrip():
    b = cb()
    target = tf_price(b, mkt(48, 0.35), steps=200)
    assert implied_volatility(b, mkt(48, 0.2), target) == pytest.approx(0.35, abs=2e-3)


def test_analyse_fields():
    b = cb()
    res = analyse(b, mkt(50), clean_price=105.0, steps=150)
    assert 0 < res["delta"] < 1
    assert res["premium_pct"] == pytest.approx((105 / 100 - 1) * 100)
    assert 0 <= accrued_interest(b, VAL) < b.coupon
