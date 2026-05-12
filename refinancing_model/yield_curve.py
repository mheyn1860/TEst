"""
Module 2 — Yield Curve Modelling.

Implements:
  - Nelson-Siegel parametric curve fitting
  - Vasicek short-rate model (analytical bond pricing + simulation)
  - Cox-Ingersoll-Ross (CIR) short-rate model
  - Forward rate extraction
  - Scenario-shifted curves
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple

from .config import VASICEK_PARAMS, NELSON_SIEGEL_PARAMS, MACRO_SCENARIOS, MARKET_LEVELS


# ---------------------------------------------------------------------------
# Nelson-Siegel Yield Curve
# ---------------------------------------------------------------------------

class NelsonSiegel:
    """
    Nelson-Siegel parametric yield curve:
      y(t) = β0 + β1 * [(1-e^(-t/τ))/(t/τ)]
                 + β2 * [(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
    """

    def __init__(
        self,
        beta0: float = None,
        beta1: float = None,
        beta2: float = None,
        tau: float = None,
    ):
        p = NELSON_SIEGEL_PARAMS
        self.beta0 = beta0 if beta0 is not None else p["beta0"]
        self.beta1 = beta1 if beta1 is not None else p["beta1"]
        self.beta2 = beta2 if beta2 is not None else p["beta2"]
        self.tau = tau if tau is not None else p["tau"]

    def spot_rate(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t = np.where(t <= 0, 1e-6, t)
        factor = (1 - np.exp(-t / self.tau)) / (t / self.tau)
        return self.beta0 + self.beta1 * factor + self.beta2 * (factor - np.exp(-t / self.tau))

    def forward_rate(self, t: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Instantaneous forward rate via numerical differentiation."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        s_t = self.spot_rate(t) / 100
        s_t_dt = self.spot_rate(t + dt) / 100
        return ((s_t_dt * (t + dt) - s_t * t) / dt) * 100

    def discount_factor(self, t: np.ndarray) -> np.ndarray:
        r = self.spot_rate(t) / 100
        return np.exp(-r * t)

    @classmethod
    def fit(cls, maturities: np.ndarray, yields: np.ndarray) -> "NelsonSiegel":
        """Fit NS parameters to observed yields via least-squares."""
        def objective(params):
            ns = cls(*params)
            fitted = ns.spot_rate(maturities)
            return np.sum((fitted - yields) ** 2)

        x0 = [2.5, -0.5, 0.5, 2.0]
        bounds = [(0, 10), (-5, 5), (-5, 5), (0.1, 10)]
        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        return cls(*res.x)

    def scenario_shift(self, scenario_key: str) -> "NelsonSiegel":
        """Apply macroeconomic scenario to yield curve (parallel + slope shift)."""
        sc = MACRO_SCENARIOS[scenario_key]
        shift = sc["eur_swap_10y_terminal"] - MARKET_LEVELS["eur_swap_10y"]
        new_beta0 = self.beta0 + shift
        slope_adj = (sc["ecb_rate_shift_bp"] / 100) * 0.3
        new_beta1 = self.beta1 + slope_adj
        return NelsonSiegel(new_beta0, new_beta1, self.beta2, self.tau)

    def curve_dataframe(self, tenors: List[float] = None) -> pd.DataFrame:
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        rates = self.spot_rate(np.array(tenors))
        fwds = self.forward_rate(np.array(tenors))
        return pd.DataFrame({
            "Tenor (Y)": tenors,
            "Spot Rate (%)": np.round(rates, 4),
            "Forward Rate (%)": np.round(fwds, 4),
            "Discount Factor": np.round(self.discount_factor(np.array(tenors)), 6),
        })


# ---------------------------------------------------------------------------
# Vasicek Short-Rate Model
# ---------------------------------------------------------------------------

class Vasicek:
    """
    dr(t) = κ(θ - r(t)) dt + σ dW(t)

    Analytical bond price:
      P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
    """

    def __init__(self, kappa=None, theta=None, sigma=None, r0=None):
        p = VASICEK_PARAMS
        self.kappa = kappa or p["kappa"]
        self.theta = theta or p["theta"]
        self.sigma = sigma or p["sigma"]
        self.r0 = r0 or p["r0"]

    def B(self, tau: np.ndarray) -> np.ndarray:
        return (1 - np.exp(-self.kappa * tau)) / self.kappa

    def A(self, tau: np.ndarray) -> np.ndarray:
        b = self.B(tau)
        sigma2 = self.sigma ** 2
        kappa2 = self.kappa ** 2
        term = (self.theta - sigma2 / (2 * kappa2)) * (b - tau) - sigma2 * b ** 2 / (4 * self.kappa)
        return np.exp(term)

    def bond_price(self, r0: float, tau: np.ndarray) -> np.ndarray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        return self.A(tau) * np.exp(-self.B(tau) * r0)

    def yield_curve(self, r0: float, tenors: np.ndarray) -> np.ndarray:
        tenors = np.atleast_1d(np.asarray(tenors, dtype=float))
        P = self.bond_price(r0, tenors)
        return -np.log(P) / tenors * 100  # in %

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1 / 52,    # weekly steps
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate n_paths paths of the short rate over n_steps steps."""
        rng = np.random.default_rng(seed)
        r = np.zeros((n_steps + 1, n_paths))
        r[0, :] = self.r0
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            dW = rng.standard_normal(n_paths)
            r[t + 1] = (
                r[t]
                + self.kappa * (self.theta - r[t]) * dt
                + self.sigma * sqrt_dt * dW
            )
        return r

    def curve_dataframe(self, tenors: List[float] = None) -> pd.DataFrame:
        if tenors is None:
            tenors = [1, 2, 3, 5, 7, 10, 15, 20]
        rates = self.yield_curve(self.r0, np.array(tenors))
        return pd.DataFrame({"Tenor (Y)": tenors, "Vasicek Yield (%)": np.round(rates, 4)})


# ---------------------------------------------------------------------------
# CIR Model (Cox-Ingersoll-Ross)
# ---------------------------------------------------------------------------

class CIR:
    """
    dr(t) = κ(θ - r(t)) dt + σ√r(t) dW(t)

    Ensures non-negative rates when 2κθ > σ² (Feller condition).
    Analytical zero-coupon bond prices.
    """

    def __init__(self, kappa=0.30, theta=0.030, sigma=0.06, r0=0.024):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
        self._check_feller()

    def _check_feller(self):
        if 2 * self.kappa * self.theta <= self.sigma ** 2:
            import warnings
            warnings.warn("Feller condition violated: rates may reach zero.")

    def _h(self) -> float:
        return np.sqrt(self.kappa ** 2 + 2 * self.sigma ** 2)

    def bond_price(self, r0: float, tau: np.ndarray) -> np.ndarray:
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        h = self._h()
        k, th, s = self.kappa, self.theta, self.sigma
        exp_ht = np.exp(h * tau)
        denom_B = (h + k) * (exp_ht - 1) + 2 * h
        B = 2 * (exp_ht - 1) / denom_B
        A_ln = (
            2 * k * th / s ** 2
            * np.log(2 * h * np.exp((k + h) * tau / 2) / denom_B)
        )
        return np.exp(A_ln - B * r0)

    def yield_curve(self, r0: float, tenors: np.ndarray) -> np.ndarray:
        tenors = np.atleast_1d(np.asarray(tenors, dtype=float))
        P = self.bond_price(r0, tenors)
        return -np.log(P) / tenors * 100

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1 / 52,
        seed: int = 42,
    ) -> np.ndarray:
        """Full-truncation Euler scheme for CIR."""
        rng = np.random.default_rng(seed)
        r = np.zeros((n_steps + 1, n_paths))
        r[0, :] = self.r0
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            dW = rng.standard_normal(n_paths)
            r_pos = np.maximum(r[t], 0)
            r[t + 1] = (
                r[t]
                + self.kappa * (self.theta - r_pos) * dt
                + self.sigma * np.sqrt(r_pos) * sqrt_dt * dW
            )
            r[t + 1] = np.maximum(r[t + 1], 0)
        return r


# ---------------------------------------------------------------------------
# Aggregate: multi-scenario curve set
# ---------------------------------------------------------------------------

def build_scenario_curves(
    base_ns: NelsonSiegel,
) -> Dict[str, NelsonSiegel]:
    curves = {}
    for key in MACRO_SCENARIOS:
        curves[key] = base_ns.scenario_shift(key)
    return curves


def forward_rate_at_horizon(
    curve: NelsonSiegel, horizon_y: float, tenor_y: float
) -> float:
    """
    Implied forward rate for a `tenor_y`-year instrument
    starting `horizon_y` years from now.
    Uses the unbiased expectations approximation.
    """
    t1 = horizon_y
    t2 = horizon_y + tenor_y
    s1 = curve.spot_rate(np.array([t1]))[0] / 100
    s2 = curve.spot_rate(np.array([t2]))[0] / 100
    fwd = (s2 * t2 - s1 * t1) / tenor_y
    return fwd * 100  # back to %
