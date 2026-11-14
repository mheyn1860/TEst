"""
Convertible-Bond-Bewertung
==========================

Kennzahlen pro 100 Nominal:
  * Parität (Conversion Value), Wandlungsprämie, Bond Floor, Investment Premium
  * Rendite (YTM) aus Marktpreis
  * Theoretischer Wert via Binomialbaum nach Tsiveriotis-Fernandes
    (Aktien- und Cash-Komponente getrennt diskontiert, Credit Spread nur auf Cash)
    inkl. Issuer-Call und Investor-Put
  * Greeks: Delta, Gamma, Vega, Rho-Credit (Spread-Sensitivität)
  * Implizite Volatilität aus dem Marktpreis

Vereinfachungen (bewusst, für ein Monitoring-Tool):
  * flache risikofreie Zinskurve, flacher Credit Spread
  * stetige Dividendenrendite
  * Aktie und Anleihe in derselben Währung (sonst FX-Kurs vorschalten)
  * Soft-Call-Trigger wird als harter Call ab call_date approximiert
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Stammdaten
# ---------------------------------------------------------------------------
@dataclass
class ConvertibleBond:
    id: str
    issuer: str
    stock_ticker: str
    maturity: date
    coupon: float                   # p.a. in % (z.B. 1.5)
    conversion_price: float         # in Aktienwährung
    currency: str = "EUR"
    isin: str = ""
    coupon_freq: int = 1            # Kupons pro Jahr
    redemption: float = 100.0       # Rückzahlung in % vom Nominal
    credit_spread_bp: float = 300.0
    dividend_yield: float = 0.0     # stetig, z.B. 0.02
    call_date: Optional[date] = None
    call_price: float = 100.0
    put_date: Optional[date] = None
    put_price: float = 100.0
    country: str = ""
    sector: str = ""
    verified: bool = False

    @property
    def conversion_ratio(self) -> float:
        """Aktien je 100 Nominal."""
        return 100.0 / self.conversion_price


@dataclass
class MarketInputs:
    stock_price: float
    volatility: float               # p.a., z.B. 0.30
    risk_free: float = 0.022        # stetig, p.a.
    valuation_date: date = field(default_factory=date.today)


def _yf(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.25


# ---------------------------------------------------------------------------
# Kupon-Schedule
# ---------------------------------------------------------------------------
def coupon_schedule(cb: ConvertibleBond, val_date: date) -> list[tuple[float, float]]:
    """Liste (t in Jahren ab val_date, Cashflow pro 100) aller künftigen Kupons
    inkl. Rückzahlung am Ende."""
    T = _yf(val_date, cb.maturity)
    if T <= 0:
        return []
    step = 1.0 / cb.coupon_freq
    cpn = cb.coupon / cb.coupon_freq
    flows = []
    t = T
    while t > 1e-9:
        flows.append((t, cpn))
        t -= step
    flows.reverse()
    t_last, c_last = flows[-1]
    flows[-1] = (t_last, c_last + cb.redemption)
    return flows


def accrued_interest(cb: ConvertibleBond, val_date: date) -> float:
    T = _yf(val_date, cb.maturity)
    if T <= 0 or cb.coupon == 0:
        return 0.0
    step = 1.0 / cb.coupon_freq
    time_to_next = T % step
    if time_to_next < 1e-9:
        time_to_next = step
    elapsed = step - time_to_next
    return cb.coupon * elapsed


# ---------------------------------------------------------------------------
# Bond-Kennzahlen
# ---------------------------------------------------------------------------
def bond_floor(cb: ConvertibleBond, mkt: MarketInputs) -> float:
    """Dirty-Barwert der Anleihe ohne Wandlungsrecht (r + Credit Spread)."""
    y = mkt.risk_free + cb.credit_spread_bp / 1e4
    return sum(cf * math.exp(-y * t) for t, cf in coupon_schedule(cb, mkt.valuation_date))


def yield_to_maturity(cb: ConvertibleBond, dirty_price: float, val_date: date) -> float:
    """Stetige Rendite, die den Dirty-Preis erklärt."""
    flows = coupon_schedule(cb, val_date)
    if not flows:
        return float("nan")
    f = lambda y: sum(cf * math.exp(-y * t) for t, cf in flows) - dirty_price
    try:
        return brentq(f, -0.5, 2.0)
    except ValueError:
        return float("nan")


def parity(cb: ConvertibleBond, stock_price: float) -> float:
    return cb.conversion_ratio * stock_price


# ---------------------------------------------------------------------------
# Binomialbaum (Tsiveriotis-Fernandes)
# ---------------------------------------------------------------------------
def tf_price(cb: ConvertibleBond, mkt: MarketInputs, steps: int = 300) -> float:
    """Theoretischer Dirty-Preis pro 100 Nominal."""
    return _tf_tree(cb, mkt, steps)[0]


def _tf_tree(cb: ConvertibleBond, mkt: MarketInputs, steps: int = 300) -> tuple[float, float, float]:
    """(Preis, dV/dS, d²V/dS²) – Greeks direkt aus den Knoten bei Schritt 2,
    damit kein Gitterrauschen durch Kurs-Bumps entsteht."""
    S0, sigma, r = mkt.stock_price, max(mkt.volatility, 1e-4), mkt.risk_free
    q = cb.dividend_yield
    cs = cb.credit_spread_bp / 1e4
    T = _yf(mkt.valuation_date, cb.maturity)
    if T <= 0:
        return max(parity(cb, S0), cb.redemption), float(cb.conversion_ratio * (S0 * cb.conversion_ratio > cb.redemption)), 0.0
    steps = max(steps, 3)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc_r = math.exp(-r * dt)
    disc_rc = math.exp(-(r + cs) * dt)
    cr = cb.conversion_ratio

    # Kupons auf Baumknoten abbilden (nicht die Rückzahlung, die kommt am Ende)
    coupon_at_step = np.zeros(steps + 1)
    for t, cf in coupon_schedule(cb, mkt.valuation_date):
        k = min(int(round(t / dt)), steps)
        coupon_at_step[k] += cf
    coupon_at_step[steps] -= cb.redemption  # Rückzahlung separat behandeln

    call_step = (int(round(_yf(mkt.valuation_date, cb.call_date) / dt))
                 if cb.call_date and cb.call_date > mkt.valuation_date else (0 if cb.call_date else None))
    put_step = (int(round(_yf(mkt.valuation_date, cb.put_date) / dt))
                if cb.put_date and cb.put_date > mkt.valuation_date else None)

    j = np.arange(steps + 1)
    S = S0 * u ** (steps - 2 * j)
    conv = cr * S
    final_bond = cb.redemption + coupon_at_step[steps]
    convert = conv >= final_bond
    E = np.where(convert, conv, 0.0)        # Aktienkomponente
    B = np.where(convert, 0.0, final_bond)  # Cash-Komponente

    for n in range(steps - 1, -1, -1):
        S = S0 * u ** (n - 2 * np.arange(n + 1))
        E = disc_r * (p * E[:-1] + (1 - p) * E[1:])
        B = disc_rc * (p * B[:-1] + (1 - p) * B[1:]) + coupon_at_step[n]
        V = E + B
        conv = cr * S

        # Investor-Put (genau am Put-Termin)
        if put_step is not None and n == put_step:
            put_ex = V < cb.put_price
            E = np.where(put_ex, 0.0, E)
            B = np.where(put_ex, cb.put_price, B)
            V = E + B

        # Issuer-Call (ab Call-Termin): Emittent ruft, wenn V > Call-Preis;
        # Investor wählt dann max(Call-Preis, Wandlung)
        if call_step is not None and n >= call_step:
            called = V > max(cb.call_price, 0.0)
            forced_conv = called & (conv >= cb.call_price)
            cash_call = called & ~forced_conv
            E = np.where(forced_conv, conv, np.where(cash_call, 0.0, E))
            B = np.where(forced_conv, 0.0, np.where(cash_call, cb.call_price, B))
            V = E + B

        # Freiwillige Wandlung (amerikanisch)
        voluntary = conv > V
        E = np.where(voluntary, conv, E)
        B = np.where(voluntary, 0.0, B)

        if n == 2:
            S2, V2 = S.copy(), E + B

    d_up = (V2[0] - V2[1]) / (S2[0] - S2[1])
    d_dn = (V2[1] - V2[2]) / (S2[1] - S2[2])
    delta = (V2[0] - V2[2]) / (S2[0] - S2[2])
    gamma = (d_up - d_dn) / (0.5 * (S2[0] - S2[2]))
    return float(E[0] + B[0]), float(delta), float(gamma)


# ---------------------------------------------------------------------------
# Greeks & implizite Vola
# ---------------------------------------------------------------------------
def greeks(cb: ConvertibleBond, mkt: MarketInputs, steps: int = 300) -> dict:
    base, dV_dS, d2V_dS2 = _tf_tree(cb, mkt, steps)

    def with_(**kw):
        return MarketInputs(**{**mkt.__dict__, **kw})

    v_up = tf_price(cb, with_(volatility=mkt.volatility + 0.01), steps)

    cs_orig = cb.credit_spread_bp
    cb.credit_spread_bp = cs_orig + 10
    cs_up = tf_price(cb, mkt, steps)
    cb.credit_spread_bp = cs_orig

    return {
        "model_price": base,
        # Delta als Anteil der Parität (0..1), branchenüblich
        "delta": dV_dS / cb.conversion_ratio,
        # Gamma: Delta-Änderung je +1 % Aktienkurs
        "gamma": d2V_dS2 / cb.conversion_ratio * mkt.stock_price * 0.01,
        "vega": v_up - base,                      # Punkte je 1 Vol-Punkt
        "credit_sens_10bp": cs_up - base,         # Punkte je +10bp Spread
    }


def implied_volatility(cb: ConvertibleBond, mkt: MarketInputs, dirty_price: float,
                       steps: int = 200) -> float:
    def f(sig):
        return tf_price(cb, MarketInputs(**{**mkt.__dict__, "volatility": sig}), steps) - dirty_price
    try:
        lo, hi = f(0.01), f(1.5)
        if lo > 0 or hi < 0:
            return float("nan")  # Preis außerhalb des Modellbereichs (Spread prüfen)
        return brentq(f, 0.01, 1.5, xtol=1e-4)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Alles in einem
# ---------------------------------------------------------------------------
def analyse(cb: ConvertibleBond, mkt: MarketInputs, clean_price: Optional[float] = None,
            steps: int = 300) -> dict:
    ai = accrued_interest(cb, mkt.valuation_date)
    par = parity(cb, mkt.stock_price)
    floor = bond_floor(cb, mkt)
    g = greeks(cb, mkt, steps)

    out = {
        "id": cb.id,
        "issuer": cb.issuer,
        "isin": cb.isin,
        "stock_ticker": cb.stock_ticker,
        "stock_price": mkt.stock_price,
        "maturity": cb.maturity.isoformat(),
        "coupon": cb.coupon,
        "conv_price": cb.conversion_price,
        "parity": par,
        "bond_floor": floor - ai,               # clean
        "accrued": ai,
        "hist_vol": mkt.volatility,
        "model_clean": g["model_price"] - ai,
        "delta": g["delta"],
        "gamma": g["gamma"],
        "vega": g["vega"],
        "credit_sens_10bp": g["credit_sens_10bp"],
        "market_clean": clean_price,
        "premium_pct": np.nan,
        "invest_premium_pct": np.nan,
        "ytm_pct": np.nan,
        "implied_vol": np.nan,
        "rich_cheap_pct": np.nan,
        "verified": cb.verified,
    }
    if clean_price is not None and not math.isnan(clean_price):
        dirty = clean_price + ai
        out["premium_pct"] = (clean_price / par - 1) * 100 if par > 0 else np.nan
        out["invest_premium_pct"] = (clean_price / (floor - ai) - 1) * 100
        out["ytm_pct"] = yield_to_maturity(cb, dirty, mkt.valuation_date) * 100
        out["implied_vol"] = implied_volatility(cb, mkt, dirty)
        out["rich_cheap_pct"] = (clean_price / out["model_clean"] - 1) * 100
    return out
