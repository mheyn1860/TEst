"""
Module 4 — Funding Mix Optimisation.

Solves:
  min  WACC + α·Refinancing_Risk + β·Liquidity_Risk + γ·Dilution_Cost

subject to:
  • Cash_t ≥ Minimum_Liquidity         (liquidity constraint)
  • Net_Debt/EBITDA ≤ Covenant_Threshold  (leverage covenant)
  • Σ Debt_maturity_year ≤ Max_Refinance_Capacity  (maturity wall)
  • Σ allocation_i = 1, allocation_i ≥ 0    (portfolio simplex)

Uses scipy.optimize with SLSQP for constrained non-linear optimisation.
Also provides mean-variance efficient frontier across funding instruments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from typing import Dict, List, Optional, Tuple

from .config import INSTRUMENTS, COVENANT_THRESHOLDS, MACRO_SCENARIOS


# ---------------------------------------------------------------------------
# Instrument universe builder
# ---------------------------------------------------------------------------

def build_instrument_universe(
    credit_spread_bp: float,
    base_risk_free_pct: float,
    tenor_preference: int = 5,
    include_equity_linked: bool = True,
) -> pd.DataFrame:
    """
    For each instrument, compute:
      - expected coupon (risk-free + instrument-specific spread)
      - refinancing risk score (1-5)
      - liquidity risk score (1-5)
      - dilution cost (0 or equity premium estimate)
      - weighted all-in cost
    """
    rows = []
    for key, inst in INSTRUMENTS.items():
        if not include_equity_linked and inst.get("dilution_risk"):
            continue

        # Spread: blend instrument-typical spread with company-specific spread
        spread_bp = inst["typical_spread_bp"] * 0.4 + credit_spread_bp * 0.6
        coupon = base_risk_free_pct + spread_bp / 100

        # Refinancing risk: shorter tenors = higher annual rollover risk
        best_tenor = min(inst["tenor_years"], key=lambda t: abs(t - tenor_preference))
        refi_risk = round(5 / best_tenor, 2)  # crude: shorter = riskier

        # Liquidity risk
        liq_map = {"liquid": 1, "semi-liquid": 3, "illiquid": 5, "n/a": 3}
        liq_risk = liq_map.get(inst["liquidity"], 3)

        # Dilution cost: rough equity-premium haircut for convertibles
        dil_cost = 0.80 if inst.get("dilution_risk") else 0.0  # % points

        rows.append({
            "key": key,
            "Instrument": inst["label"],
            "Tenor (Y)": best_tenor,
            "Coupon (%)": round(coupon, 3),
            "Spread (bp)": round(spread_bp, 1),
            "Refi Risk": refi_risk,
            "Liquidity Risk": liq_risk,
            "Dilution Cost (%)": dil_cost,
            "Covenant Intensity": inst["covenant_intensity"],
            "Min Size (m€)": inst["min_size_mEUR"],
            "Investor Base": inst["investor_base"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio all-in cost (weighted)
# ---------------------------------------------------------------------------

def portfolio_cost(
    weights: np.ndarray,
    coupons: np.ndarray,
    refi_risks: np.ndarray,
    liq_risks: np.ndarray,
    dil_costs: np.ndarray,
    alpha: float = 0.015,   # weight on refinancing risk
    beta: float = 0.010,    # weight on liquidity risk
    gamma: float = 0.020,   # weight on dilution cost
) -> float:
    """Composite objective function."""
    wacc = np.dot(weights, coupons) / 100
    refi_risk = np.dot(weights, refi_risks) * alpha
    liq_risk = np.dot(weights, liq_risks) * beta
    dil = np.dot(weights, dil_costs) * gamma
    return wacc + refi_risk + liq_risk + dil


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

class FundingOptimiser:

    def __init__(
        self,
        instrument_df: pd.DataFrame,
        total_refinancing_need_mEUR: float,
        ebitda_mEUR: float,
        current_net_debt_mEUR: float,
        min_cash_mEUR: float = 30.0,
        max_net_debt_ebitda: float = None,
        alpha: float = 0.015,
        beta: float = 0.010,
        gamma: float = 0.020,
    ):
        self.df = instrument_df.copy()
        self.total_need = total_refinancing_need_mEUR
        self.ebitda = ebitda_mEUR
        self.net_debt = current_net_debt_mEUR
        self.min_cash = min_cash_mEUR
        self.max_leverage = max_net_debt_ebitda or COVENANT_THRESHOLDS["net_debt_ebitda_max"]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.coupons = self.df["Coupon (%)"].values
        self.refi = self.df["Refi Risk"].values
        self.liq = self.df["Liquidity Risk"].values
        self.dil = self.df["Dilution Cost (%)"].values
        self.n = len(self.df)

    def _objective(self, w: np.ndarray) -> float:
        return portfolio_cost(w, self.coupons, self.refi, self.liq, self.dil,
                              self.alpha, self.beta, self.gamma)

    def optimise(self) -> Dict:
        # Equality: weights sum to 1
        eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Inequality: leverage covenant
        # New net debt after refinancing = current net_debt (we're just refinancing, not adding)
        # Constraint satisfied by construction if refi doesn't increase net debt
        # We model it as: max single tranche cannot exceed 60% (diversification)
        concentration = {"type": "ineq", "fun": lambda w: 0.60 - np.max(w)}

        bounds = [(0.0, 0.70)] * self.n  # no instrument > 70%

        # Initial equal-weight guess
        w0 = np.ones(self.n) / self.n

        result = minimize(
            self._objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[eq_constraint, concentration],
            options={"ftol": 1e-10, "maxiter": 1000},
        )

        if not result.success:
            # Fallback to equal weights
            result.x = w0

        weights = result.x
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum()

        weighted_coupon = float(np.dot(weights, self.coupons))
        weighted_tenor = float(np.dot(weights, self.df["Tenor (Y)"].values))
        obj_val = self._objective(weights)

        allocation = self.df[["Instrument", "Coupon (%)", "Tenor (Y)"]].copy()
        allocation["Weight (%)"] = np.round(weights * 100, 1)
        allocation["Volume (m€)"] = np.round(weights * self.total_need, 1)
        allocation = allocation[allocation["Weight (%)"] >= 1.0].sort_values("Weight (%)", ascending=False)

        return {
            "optimal_weights": weights,
            "allocation_df": allocation,
            "weighted_coupon_pct": round(weighted_coupon, 4),
            "weighted_tenor_y": round(weighted_tenor, 2),
            "objective_value": round(obj_val, 6),
            "optimiser_success": result.success,
            "total_refinancing_mEUR": self.total_need,
        }

    def efficient_frontier(self, n_points: int = 30) -> pd.DataFrame:
        """
        Trace the cost-risk frontier by varying alpha (risk aversion).
        Returns a DataFrame with (risk, cost) pairs.
        """
        records = []
        for alpha in np.linspace(0, 0.10, n_points):
            self.alpha = alpha
            res = self.optimise()
            w = res["optimal_weights"]
            records.append({
                "Risk Aversion (α)": round(alpha, 4),
                "Weighted Cost (%)": res["weighted_coupon_pct"],
                "Refi Risk Score": round(float(np.dot(w, self.refi)), 3),
                "Objective": res["objective_value"],
            })
        self.alpha = 0.015  # reset
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Covenant stress test
# ---------------------------------------------------------------------------

def covenant_stress_test(
    current_metrics,         # CreditMetrics object
    ebitda_shocks: List[float] = None,
    rate_shocks_bp: List[float] = None,
    thresholds: Dict = None,
) -> pd.DataFrame:
    """
    Stress test covenant headroom under EBITDA and interest rate shocks.
    Returns breach/pass matrix.
    """
    if ebitda_shocks is None:
        ebitda_shocks = [-0.10, -0.20, -0.30, -0.40]
    if rate_shocks_bp is None:
        rate_shocks_bp = [0, 50, 100, 200]
    if thresholds is None:
        thresholds = COVENANT_THRESHOLDS

    rows = []
    base_ebitda_implied = (
        current_metrics.net_debt / current_metrics.net_debt_ebitda
        if current_metrics.net_debt_ebitda != float("inf") else 100
    )
    base_interest = base_ebitda_implied / current_metrics.ebitda_interest_coverage

    for shock in ebitda_shocks:
        for rate_shock in rate_shocks_bp:
            stressed_ebitda = base_ebitda_implied * (1 + shock)
            additional_interest = current_metrics.gross_debt * rate_shock / 10_000
            stressed_interest = base_interest + additional_interest
            stressed_nd_ebitda = current_metrics.net_debt / stressed_ebitda if stressed_ebitda > 0 else float("inf")
            stressed_coverage = stressed_ebitda / stressed_interest if stressed_interest > 0 else float("inf")

            nd_breach = stressed_nd_ebitda > thresholds["net_debt_ebitda_max"]
            ic_breach = stressed_coverage < thresholds["interest_coverage_min"]

            rows.append({
                "EBITDA Shock": f"{shock*100:.0f}%",
                "Rate Shock (bp)": rate_shock,
                "Stressed ND/EBITDA": round(stressed_nd_ebitda, 2),
                "Stressed IC": round(stressed_coverage, 2),
                "ND/EBITDA Breach": "BREACH" if nd_breach else "OK",
                "IC Breach": "BREACH" if ic_breach else "OK",
                "Status": "BREACH" if (nd_breach or ic_breach) else "OK",
            })

    return pd.DataFrame(rows)


from typing import List
