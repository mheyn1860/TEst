"""
Module 3 — Credit Spread Modelling.

Implements:
  - Merton structural model (Distance-to-Default / KMV)
  - Reduced-form hazard rate model
  - Credit spread decomposition (risk-free + credit premium + liquidity premium)
  - Shadow credit spread from fundamental factors
  - New issue concession pricing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Optional, Tuple

from .config import MARKET_LEVELS, MACRO_SCENARIOS, INSTRUMENTS


# ---------------------------------------------------------------------------
# Merton Structural Model
# ---------------------------------------------------------------------------

class MertonModel:
    """
    Equity = call option on firm assets (Black-Scholes).
    Solves simultaneously for V (asset value) and σ_V (asset vol).

    Inputs:
      E     — market cap (equity value)
      D     — face value of debt (typically 1Y horizon)
      r     — risk-free rate (annual, decimal)
      T     — horizon (years)
      sigma_E — equity volatility (annual, decimal)
    """

    def __init__(self, E: float, D: float, r: float, T: float, sigma_E: float):
        self.E = E
        self.D = D
        self.r = r
        self.T = T
        self.sigma_E = sigma_E
        self.V, self.sigma_V = self._solve_asset_params()

    def _solve_asset_params(self) -> Tuple[float, float]:
        """
        System of two equations:
          E = V*N(d1) - D*exp(-rT)*N(d2)
          sigma_E * E = N(d1) * sigma_V * V
        """
        def equations(params):
            V, sigma_V = params
            d1 = (np.log(V / self.D) + (self.r + 0.5 * sigma_V ** 2) * self.T) / (
                sigma_V * np.sqrt(self.T)
            )
            d2 = d1 - sigma_V * np.sqrt(self.T)
            E_calc = V * norm.cdf(d1) - self.D * np.exp(-self.r * self.T) * norm.cdf(d2)
            eq1 = E_calc - self.E
            eq2 = norm.cdf(d1) * sigma_V * V - self.sigma_E * self.E
            return [eq1, eq2]

        # Initial guess: V ≈ E + D
        from scipy.optimize import fsolve
        V0 = self.E + self.D * np.exp(-self.r * self.T)
        sigma_V0 = self.sigma_E * self.E / V0
        sol = fsolve(equations, [V0, sigma_V0], full_output=True)
        V, sigma_V = sol[0]
        return abs(V), abs(sigma_V)

    @property
    def d1(self) -> float:
        return (np.log(self.V / self.D) + (self.r + 0.5 * self.sigma_V ** 2) * self.T) / (
            self.sigma_V * np.sqrt(self.T)
        )

    @property
    def d2(self) -> float:
        return self.d1 - self.sigma_V * np.sqrt(self.T)

    @property
    def distance_to_default(self) -> float:
        """KMV Distance-to-Default: DD = d2 in risk-neutral, adjusted here."""
        return self.d2

    @property
    def risk_neutral_pd(self) -> float:
        """Risk-neutral probability of default over horizon T."""
        return norm.cdf(-self.d2)

    @property
    def implied_credit_spread_bp(self) -> float:
        """
        Implied spread from Merton model:
          s = -ln(N(d2) + (V/D*exp(rT))*N(-d1)) / T
        """
        d1, d2 = self.d1, self.d2
        V, D, r, T = self.V, self.D, self.r, self.T
        term = norm.cdf(d2) + (V / (D * np.exp(-r * T))) * norm.cdf(-d1)
        term = max(term, 1e-12)
        spread = -np.log(term) / T
        return spread * 10_000  # to basis points

    def summary(self) -> Dict:
        return {
            "Asset Value (V)": round(self.V, 2),
            "Asset Volatility (σ_V)": round(self.sigma_V * 100, 2),
            "Distance to Default (DD)": round(self.distance_to_default, 4),
            "Risk-Neutral PD (%)": round(self.risk_neutral_pd * 100, 4),
            "Implied Spread (bp)": round(self.implied_credit_spread_bp, 1),
        }


# ---------------------------------------------------------------------------
# Reduced-Form Hazard Rate Model
# ---------------------------------------------------------------------------

class HazardRateModel:
    """
    Constant hazard rate λ with recovery rate R.
    Survival probability: S(t) = exp(-λ*t)
    Credit spread: s = λ * (1 - R)

    Calibrated from market CDS or bond spreads.
    """

    def __init__(self, hazard_rate: float = 0.03, recovery_rate: float = 0.40):
        self.lam = hazard_rate
        self.R = recovery_rate

    @classmethod
    def from_spread(cls, spread_bp: float, recovery_rate: float = 0.40) -> "HazardRateModel":
        spread = spread_bp / 10_000
        lam = spread / (1 - recovery_rate)
        return cls(lam, recovery_rate)

    def survival_probability(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.lam * np.asarray(t))

    def default_probability(self, t1: float, t2: float) -> float:
        """Marginal PD between t1 and t2."""
        return self.survival_probability(np.array([t1]))[0] - self.survival_probability(np.array([t2]))[0]

    def cumulative_pd(self, t: np.ndarray) -> np.ndarray:
        return 1 - self.survival_probability(t)

    def implied_spread_bp(self) -> float:
        return self.lam * (1 - self.R) * 10_000

    def risky_annuity(self, maturity: float, freq: int = 2) -> float:
        """Present value of risky fixed cash flows (duration proxy)."""
        dt = 1 / freq
        times = np.arange(dt, maturity + dt / 2, dt)
        pv = np.sum(self.survival_probability(times) * dt)
        return pv

    def par_spread_bp(self, maturity: float, risk_free_rate: float = 0.025) -> float:
        """Par CDS spread."""
        times = np.arange(0.5, maturity + 0.25, 0.5)
        dt = 0.5
        numerator = (1 - self.R) * sum(
            self.default_probability(t - dt, t) * np.exp(-risk_free_rate * t)
            for t in times
        )
        denominator = np.sum(
            self.survival_probability(times) * np.exp(-risk_free_rate * times) * dt
        )
        return (numerator / denominator) * 10_000 if denominator > 0 else 0


# ---------------------------------------------------------------------------
# Fundamental Credit Spread Estimation
# ---------------------------------------------------------------------------

def fundamental_spread_bp(
    net_debt_ebitda: float,
    ebitda_interest_coverage: float,
    ffo_debt: float,
    fcf_conversion: float,
    equity_volatility: float,       # annualised, decimal (e.g. 0.30 = 30%)
    market_cap_mEUR: float,
    ebitda_growth_rate: float,
    scenario_key: str = "base_case",
) -> float:
    """
    Regression-style fundamental credit spread model.
    Weights approximate those from academic CDS spread regression studies.
    Returns spread in basis points.
    """
    # Base components (each driver has an empirically-inspired coefficient)
    leverage_score = max(0, (net_debt_ebitda - 2.0) * 40)        # +40bp per turn above 2x
    coverage_score = max(0, (4.0 - ebitda_interest_coverage) * 20)
    ffo_score = max(0, (0.20 - ffo_debt) * 300)
    fcf_score = max(0, (0.50 - fcf_conversion) * 60)
    vol_score = equity_volatility * 250                           # vol premium
    size_score = max(0, (500 - market_cap_mEUR) / 500 * 80)      # illiquidity premium
    growth_score = max(0, -ebitda_growth_rate * 100)              # negative growth → wider

    raw_spread = 50 + leverage_score + coverage_score + ffo_score + fcf_score + vol_score + size_score + growth_score

    # Macro scenario multiplier
    mult = MACRO_SCENARIOS[scenario_key]["credit_spread_multiplier"]
    # Overlay market-level iTraxx
    itraxx_adj = (MARKET_LEVELS["itraxx_crossover"] - 320) * 0.3

    return round(raw_spread * mult + itraxx_adj, 1)


# ---------------------------------------------------------------------------
# All-in Funding Cost for an Instrument / Tenor
# ---------------------------------------------------------------------------

def all_in_cost_pct(
    risk_free_rate_pct: float,
    credit_spread_bp: float,
    new_issue_concession_bp: float,
    liquidity_premium_bp: float,
    issuance_fee_pct: float,
    tenor_years: float,
) -> Dict[str, float]:
    """
    Decompose and sum all-in cost of a new issuance.
    Returns components and total, annualised.
    """
    coupon = risk_free_rate_pct + credit_spread_bp / 100
    nic = new_issue_concession_bp / 100
    liq = liquidity_premium_bp / 100
    fee_ann = issuance_fee_pct / tenor_years

    total = coupon + nic + liq + fee_ann

    return {
        "Risk-Free Rate (%)": round(risk_free_rate_pct, 4),
        "Credit Spread (bp)": round(credit_spread_bp, 1),
        "New Issue Concession (bp)": round(nic * 100, 1),
        "Liquidity Premium (bp)": round(liq * 100, 1),
        "Issuance Fee p.a. (bp)": round(fee_ann * 100, 1),
        "All-in Cost (%)": round(total, 4),
    }


# ---------------------------------------------------------------------------
# Multi-scenario spread matrix
# ---------------------------------------------------------------------------

def spread_scenario_matrix(
    base_spread_bp: float,
    tenors: List[float] = None,
) -> pd.DataFrame:
    """Show spread and all-in cost across scenarios and tenors."""
    from .yield_curve import NelsonSiegel
    if tenors is None:
        tenors = [3, 5, 7, 10]

    ns_base = NelsonSiegel()
    rows = []
    for sc_key, sc in MACRO_SCENARIOS.items():
        ns = ns_base.scenario_shift(sc_key)
        adjusted_spread = base_spread_bp * sc["credit_spread_multiplier"]
        nic = MARKET_LEVELS["new_issue_concession_hy"] if base_spread_bp > 200 else MARKET_LEVELS["new_issue_concession_ig"]
        for tenor in tenors:
            rf = ns.spot_rate(np.array([tenor]))[0]
            cost = all_in_cost_pct(rf, adjusted_spread, nic, 20, 0.50, tenor)
            rows.append({
                "Scenario": sc["label"],
                "Probability": f"{sc['probability']*100:.0f}%",
                "Tenor (Y)": tenor,
                "Risk-Free (%)": round(rf, 3),
                "Credit Spread (bp)": round(adjusted_spread, 1),
                "All-in Cost (%)": cost["All-in Cost (%)"],
            })

    return pd.DataFrame(rows)


# required for type hints in this module
from typing import List
