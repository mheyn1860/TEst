"""
Integrated Capital Markets Refinancing Model
============================================

A quantitative framework combining:
  - Structural credit modelling (Merton / KMV Distance-to-Default)
  - Yield curve modelling (Nelson-Siegel, Vasicek, CIR)
  - Macro scenario framework (5 ECB/GDP scenarios)
  - Monte Carlo simulation (10,000 paths, correlated GBM + OU)
  - Funding mix optimisation (SLSQP with covenant constraints)
  - IB-style visualisation dashboard

Quick start
-----------
>>> from refinancing_model import run_full_model
>>> results = run_full_model()
"""

from .financial_analysis import (
    BalanceSheet,
    IncomeStatement,
    CashFlowStatement,
    DebtSchedule,
    CreditMetrics,
    compute_credit_metrics,
    credit_metrics_to_df,
    implied_rating,
    project_financials,
)
from .yield_curve import NelsonSiegel, Vasicek, CIR, build_scenario_curves, forward_rate_at_horizon
from .credit_model import MertonModel, HazardRateModel, fundamental_spread_bp, spread_scenario_matrix
from .optimization import FundingOptimiser, build_instrument_universe, covenant_stress_test
from .monte_carlo import MonteCarloEngine, blended_scenario_summary
from .visualization import (
    plot_maturity_profile,
    plot_yield_curve_fan,
    plot_mc_cost_fan,
    plot_funding_mix,
    plot_covenant_heatmap,
    plot_scenario_outcomes,
    plot_efficient_frontier,
    plot_dashboard,
)

__version__ = "1.0.0"
__all__ = [
    "BalanceSheet", "IncomeStatement", "CashFlowStatement", "DebtSchedule",
    "CreditMetrics", "compute_credit_metrics", "credit_metrics_to_df",
    "implied_rating", "project_financials",
    "NelsonSiegel", "Vasicek", "CIR", "build_scenario_curves", "forward_rate_at_horizon",
    "MertonModel", "HazardRateModel", "fundamental_spread_bp", "spread_scenario_matrix",
    "FundingOptimiser", "build_instrument_universe", "covenant_stress_test",
    "MonteCarloEngine", "blended_scenario_summary",
    "plot_maturity_profile", "plot_yield_curve_fan", "plot_mc_cost_fan",
    "plot_funding_mix", "plot_covenant_heatmap", "plot_scenario_outcomes",
    "plot_efficient_frontier", "plot_dashboard",
]
