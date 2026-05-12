"""
Module 5 — Monte Carlo Simulation Engine.

Simulates jointly:
  - Short rate (Vasicek)
  - Credit spread (mean-reverting with macro correlation)
  - EBITDA (geometric Brownian motion)

Over N paths × T years to produce:
  - Probability distribution of refinancing costs
  - Covenant breach probabilities
  - Default probability estimates
  - Optimal issuance window identification
  - Value-at-Risk on funding costs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .config import MC_CONFIG, VASICEK_PARAMS, COVENANT_THRESHOLDS, MACRO_SCENARIOS


# ---------------------------------------------------------------------------
# Correlated Brownian motion generator
# ---------------------------------------------------------------------------

def correlated_brownian(
    n_paths: int,
    n_steps: int,
    dt: float,
    corr_matrix: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate correlated standard Brownian increments.
    Returns shape (n_steps, n_paths, n_factors).
    """
    rng = np.random.default_rng(seed)
    n_factors = corr_matrix.shape[0]
    L = np.linalg.cholesky(corr_matrix)
    z = rng.standard_normal((n_steps, n_paths, n_factors))
    return (L @ z[..., np.newaxis]).squeeze(-1) * np.sqrt(dt)


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

class SimulationResult:

    def __init__(
        self,
        rates: np.ndarray,          # (n_steps+1, n_paths)
        spreads: np.ndarray,        # (n_steps+1, n_paths) — in bp
        ebitda: np.ndarray,         # (n_steps+1, n_paths)
        dt: float,
        tenor_years: int,
    ):
        self.rates = rates
        self.spreads = spreads
        self.ebitda = ebitda
        self.dt = dt
        self.tenor_years = tenor_years
        self.n_paths = rates.shape[1]
        self.n_steps = rates.shape[0] - 1

    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_steps + 1) * self.dt

    def all_in_cost_at_horizon(self, horizon_step: int, nic_bp: float = 20) -> np.ndarray:
        """All-in cost (%) at a given issuance time step across all paths."""
        r = self.rates[horizon_step] * 100  # to %
        s = self.spreads[horizon_step] / 100  # bp → %
        return r + s + nic_bp / 100

    def optimal_issuance_window(
        self,
        lookback_steps: int = 4,
        cost_percentile: float = 25,
    ) -> np.ndarray:
        """
        Identify steps where median all-in cost is in the bottom quantile
        (i.e., attractive issuance windows).
        Returns boolean mask over time steps.
        """
        n_steps = self.n_steps
        costs = np.array([self.all_in_cost_at_horizon(t).mean() for t in range(n_steps + 1)])
        threshold = np.percentile(costs, cost_percentile)
        return costs <= threshold

    def covenant_breach_probability(
        self,
        net_debt: float,
        max_nd_ebitda: float = None,
    ) -> np.ndarray:
        """
        Probability of Net Debt / EBITDA exceeding threshold at each time step.
        """
        threshold = max_nd_ebitda or COVENANT_THRESHOLDS["net_debt_ebitda_max"]
        n_steps = self.n_steps
        probs = np.zeros(n_steps + 1)
        for t in range(n_steps + 1):
            nd_ebitda = net_debt / self.ebitda[t]
            probs[t] = np.mean(nd_ebitda > threshold)
        return probs

    def default_probability(
        self,
        net_debt: float,
        min_coverage: float = None,
        interest_expense_base: float = None,
    ) -> np.ndarray:
        """
        Simplified default proxy: EBITDA < interest expense at stressed rates.
        """
        min_cov = min_coverage or COVENANT_THRESHOLDS["interest_coverage_min"]
        interest_expense_base = interest_expense_base or net_debt * 0.04
        probs = np.zeros(self.n_steps + 1)
        for t in range(self.n_steps + 1):
            stressed_interest = interest_expense_base * (1 + self.rates[t] / self.rates[0].mean())
            coverage = self.ebitda[t] / stressed_interest
            probs[t] = np.mean(coverage < min_cov)
        return probs

    def cost_statistics(self, horizon_step: int) -> Dict:
        """Summary statistics for all-in cost distribution at a given horizon."""
        costs = self.all_in_cost_at_horizon(horizon_step)
        return {
            "Mean (%)": round(float(np.mean(costs)), 4),
            "Median (%)": round(float(np.median(costs)), 4),
            "Std Dev (%)": round(float(np.std(costs)), 4),
            "5th Pctile (%)": round(float(np.percentile(costs, 5)), 4),
            "25th Pctile (%)": round(float(np.percentile(costs, 25)), 4),
            "75th Pctile (%)": round(float(np.percentile(costs, 75)), 4),
            "95th Pctile (%)": round(float(np.percentile(costs, 95)), 4),
            "VaR 95% (%)": round(float(np.percentile(costs, 95)), 4),
            "CVaR 95% (%)": round(float(costs[costs >= np.percentile(costs, 95)].mean()), 4),
        }

    def refinancing_outcome_probabilities(
        self,
        net_debt: float,
        distress_nd_ebitda: float = 6.0,
        expensive_nd_ebitda: float = 4.5,
        horizon_step: int = None,
    ) -> Dict:
        """
        Classify outcomes:
          - Successful: ND/EBITDA < expensive threshold
          - Expensive: between thresholds
          - Distressed: ND/EBITDA > distress threshold
        """
        if horizon_step is None:
            horizon_step = self.n_steps // 2
        nd_ebitda = net_debt / self.ebitda[horizon_step]
        p_success = float(np.mean(nd_ebitda < expensive_nd_ebitda))
        p_expensive = float(np.mean(
            (nd_ebitda >= expensive_nd_ebitda) & (nd_ebitda < distress_nd_ebitda)
        ))
        p_distress = float(np.mean(nd_ebitda >= distress_nd_ebitda))
        return {
            "Successful Refinancing": round(p_success * 100, 1),
            "Expensive Refinancing": round(p_expensive * 100, 1),
            "Distressed Refinancing": round(p_distress * 100, 1),
        }


# ---------------------------------------------------------------------------
# Main Monte Carlo engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:

    def __init__(self, config: Dict = None):
        cfg = config or MC_CONFIG
        self.n_paths = cfg["n_simulations"]
        self.horizon_years = cfg["time_horizon_years"]
        self.steps_per_year = 52  # weekly
        self.dt = 1 / self.steps_per_year
        self.n_steps = self.horizon_years * self.steps_per_year
        self.seed = cfg["seed"]

        # Vasicek parameters
        vp = VASICEK_PARAMS
        self.kappa_r = vp["kappa"]
        self.theta_r = vp["theta"]
        self.sigma_r = vp["sigma"]
        self.r0 = vp["r0"]

        # Spread mean-reversion (Ornstein-Uhlenbeck)
        self.kappa_s = 0.50          # mean-reversion speed for spreads
        self.spread_vol_bp = cfg["spread_vol_bp"]

        # EBITDA GBM
        self.ebitda_vol = cfg["ebitda_vol"]

        # Correlations
        self.rho_rs = cfg["correlation_rate_spread"]
        self.rho_es = cfg["correlation_ebitda_spread"]

    def _build_corr_matrix(self) -> np.ndarray:
        """3x3 correlation matrix: [rate, spread, ebitda]."""
        rho_re = 0.10  # rate-EBITDA (mild positive)
        C = np.array([
            [1.0,          self.rho_rs, rho_re],
            [self.rho_rs,  1.0,         self.rho_es],
            [rho_re,       self.rho_es, 1.0],
        ])
        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(C)
        if np.any(eigvals < 0):
            C += np.eye(3) * (abs(eigvals.min()) + 1e-6)
            d = np.sqrt(np.diag(C))
            C = C / np.outer(d, d)
        return C

    def run(
        self,
        base_spread_bp: float,
        base_ebitda: float,
        ebitda_drift: float = 0.03,
        tenor_years: int = 5,
    ) -> SimulationResult:
        """
        Run joint simulation.

        Args:
            base_spread_bp: Starting credit spread in basis points
            base_ebitda: Starting EBITDA in m€
            ebitda_drift: Annual EBITDA growth rate (risk-neutral)
            tenor_years: Instrument tenor being modelled
        """
        C = self._build_corr_matrix()
        dW = correlated_brownian(self.n_paths, self.n_steps, self.dt, C, self.seed)
        # dW shape: (n_steps, n_paths, 3)

        rates = np.zeros((self.n_steps + 1, self.n_paths))
        spreads = np.zeros((self.n_steps + 1, self.n_paths))
        ebitda = np.zeros((self.n_steps + 1, self.n_paths))

        rates[0] = self.r0
        spreads[0] = base_spread_bp
        ebitda[0] = base_ebitda

        spread_vol_step = self.spread_vol_bp * np.sqrt(self.dt)
        ebitda_vol_step = self.ebitda_vol * np.sqrt(self.dt)
        theta_s = base_spread_bp  # spread mean-reverts to starting level (base case)

        for t in range(self.n_steps):
            dW_r = dW[t, :, 0]
            dW_s = dW[t, :, 1]
            dW_e = dW[t, :, 2]

            # Vasicek rate
            rates[t + 1] = (
                rates[t]
                + self.kappa_r * (self.theta_r - rates[t]) * self.dt
                + self.sigma_r * dW_r
            )

            # OU spread (always non-negative via clipping)
            spreads[t + 1] = np.maximum(
                spreads[t]
                + self.kappa_s * (theta_s - spreads[t]) * self.dt
                + spread_vol_step * dW_s,
                10,  # floor at 10bp
            )

            # Log-normal EBITDA (GBM)
            ebitda[t + 1] = ebitda[t] * np.exp(
                (ebitda_drift - 0.5 * self.ebitda_vol ** 2) * self.dt
                + ebitda_vol_step * dW_e
            )

        return SimulationResult(rates, spreads, ebitda, self.dt, tenor_years)

    def run_scenario_conditioned(
        self,
        base_spread_bp: float,
        base_ebitda: float,
        scenario_key: str,
        ebitda_drift: float = 0.03,
        tenor_years: int = 5,
    ) -> SimulationResult:
        """
        Run simulation with terminal conditions conditioned on macro scenario.
        Shifts theta_r and theta_s toward scenario terminals.
        """
        sc = MACRO_SCENARIOS[scenario_key]

        # Adjust long-run mean of rate toward scenario terminal
        shifted_theta_r = sc["eur_swap_10y_terminal"] / 100
        shifted_spread = base_spread_bp * sc["credit_spread_multiplier"] + sc["hY_oas_shift_bp"] * 0.3
        shifted_ebitda_drift = ebitda_drift + sc["gdp_growth"] * 0.3

        # Temporarily override params
        orig_theta = self.theta_r
        self.theta_r = shifted_theta_r
        result = self.run(shifted_spread, base_ebitda, shifted_ebitda_drift, tenor_years)
        self.theta_r = orig_theta  # restore

        return result


# ---------------------------------------------------------------------------
# Scenario-blended summary
# ---------------------------------------------------------------------------

def blended_scenario_summary(
    engine: MonteCarloEngine,
    base_spread_bp: float,
    net_debt: float,
    base_ebitda: float,
    tenor_years: int = 5,
) -> pd.DataFrame:
    """
    Run each macro scenario, weight by probability, produce summary statistics.
    """
    rows = []
    for sc_key, sc in MACRO_SCENARIOS.items():
        res = engine.run_scenario_conditioned(
            base_spread_bp, base_ebitda, sc_key, tenor_years=tenor_years
        )
        mid_step = engine.n_steps // 2
        cost_stats = res.cost_statistics(mid_step)
        outcomes = res.refinancing_outcome_probabilities(net_debt)
        breach_probs = res.covenant_breach_probability(net_debt)
        default_probs = res.default_probability(net_debt)

        rows.append({
            "Scenario": sc["label"],
            "Weight": sc["probability"],
            "Median Cost (%)": cost_stats["Median (%)"],
            "VaR-95 Cost (%)": cost_stats["VaR 95% (%)"],
            "CVaR-95 Cost (%)": cost_stats["CVaR 95% (%)"],
            "P(Successful Refi) %": outcomes["Successful Refinancing"],
            "P(Expensive Refi) %": outcomes["Expensive Refinancing"],
            "P(Distressed Refi) %": outcomes["Distressed Refinancing"],
            "P(Covenant Breach @T/2) %": round(breach_probs[mid_step] * 100, 2),
            "P(Default @T/2) %": round(default_probs[mid_step] * 100, 2),
        })

    df = pd.DataFrame(rows)

    # Probability-weighted row
    wt = df["Weight"].values
    num_cols = [c for c in df.columns if c not in ("Scenario", "Weight")]
    weighted_row = {"Scenario": "★ Probability-Weighted", "Weight": 1.0}
    for col in num_cols:
        weighted_row[col] = round(float(np.dot(df[col].values, wt)), 3)
    df = pd.concat([df, pd.DataFrame([weighted_row])], ignore_index=True)

    return df
