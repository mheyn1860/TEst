"""
Global configuration: macroeconomic scenarios, instrument parameters, model settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Macroeconomic scenarios
# ---------------------------------------------------------------------------

MACRO_SCENARIOS: Dict[str, Dict] = {
    "base_case": {
        "label": "Base Case / Soft Landing",
        "probability": 0.45,
        "ecb_rate_shift_bp": -50,          # cumulative cuts vs current
        "bund_10y_terminal": 2.40,
        "eur_swap_10y_terminal": 2.70,
        "hY_oas_shift_bp": +20,
        "itraxx_crossover_shift_bp": +15,
        "gdp_growth": 1.2,
        "inflation": 2.1,
        "credit_spread_multiplier": 1.00,
    },
    "re_acceleration": {
        "label": "Re-Acceleration / Higher for Longer",
        "probability": 0.20,
        "ecb_rate_shift_bp": +25,
        "bund_10y_terminal": 3.10,
        "eur_swap_10y_terminal": 3.40,
        "hY_oas_shift_bp": -30,
        "itraxx_crossover_shift_bp": -25,
        "gdp_growth": 2.3,
        "inflation": 3.2,
        "credit_spread_multiplier": 0.85,
    },
    "recession": {
        "label": "Recession",
        "probability": 0.20,
        "ecb_rate_shift_bp": -150,
        "bund_10y_terminal": 1.60,
        "eur_swap_10y_terminal": 1.90,
        "hY_oas_shift_bp": +250,
        "itraxx_crossover_shift_bp": +200,
        "gdp_growth": -1.5,
        "inflation": 1.2,
        "credit_spread_multiplier": 2.10,
    },
    "stagflation": {
        "label": "Stagflation",
        "probability": 0.15,
        "ecb_rate_shift_bp": +50,
        "bund_10y_terminal": 3.50,
        "eur_swap_10y_terminal": 3.80,
        "hY_oas_shift_bp": +150,
        "itraxx_crossover_shift_bp": +120,
        "gdp_growth": -0.3,
        "inflation": 4.8,
        "credit_spread_multiplier": 1.70,
    },
}

# ---------------------------------------------------------------------------
# Current market levels (as of model calibration)
# ---------------------------------------------------------------------------

MARKET_LEVELS = {
    "ecb_depo_rate": 2.40,          # %
    "bund_2y": 2.10,
    "bund_5y": 2.30,
    "bund_10y": 2.55,
    "eur_swap_2y": 2.45,
    "eur_swap_5y": 2.65,
    "eur_swap_10y": 2.80,
    "itraxx_main": 65,              # bp
    "itraxx_crossover": 320,        # bp
    "hY_oas_eur": 380,              # bp
    "ig_oas_eur": 110,              # bp
    "new_issue_concession_ig": 10,  # bp
    "new_issue_concession_hy": 25,  # bp
}

# ---------------------------------------------------------------------------
# Refinancing instrument definitions
# ---------------------------------------------------------------------------

INSTRUMENTS = {
    "schuldschein": {
        "label": "Schuldschein",
        "typical_spread_bp": 150,
        "tenor_years": [3, 5, 7],
        "covenant_intensity": "low",
        "dilution_risk": False,
        "min_size_mEUR": 50,
        "investor_base": "Banks / Insurers",
        "liquidity": "illiquid",
        "rating_required": False,
        "flexibility": "medium",
    },
    "nordic_bond": {
        "label": "Nordic / High-Yield Bond",
        "typical_spread_bp": 400,
        "tenor_years": [3, 5],
        "covenant_intensity": "medium",
        "dilution_risk": False,
        "min_size_mEUR": 100,
        "investor_base": "HY Funds / Asset Managers",
        "liquidity": "semi-liquid",
        "rating_required": False,
        "flexibility": "high",
    },
    "convertible_bond": {
        "label": "Convertible Bond",
        "typical_spread_bp": 50,           # very low coupon offset by equity premium
        "tenor_years": [5, 7],
        "covenant_intensity": "low",
        "dilution_risk": True,
        "min_size_mEUR": 150,
        "investor_base": "CB Funds / Arb Funds",
        "liquidity": "liquid",
        "rating_required": False,
        "flexibility": "high",
    },
    "private_placement": {
        "label": "US Private Placement (USPP)",
        "typical_spread_bp": 130,
        "tenor_years": [7, 10, 12],
        "covenant_intensity": "high",
        "dilution_risk": False,
        "min_size_mEUR": 100,
        "investor_base": "US Life Insurers",
        "liquidity": "illiquid",
        "rating_required": False,
        "flexibility": "low",
    },
    "ig_bond": {
        "label": "IG Public Bond",
        "typical_spread_bp": 80,
        "tenor_years": [5, 7, 10],
        "covenant_intensity": "minimal",
        "dilution_risk": False,
        "min_size_mEUR": 300,
        "investor_base": "Broad IG / CB / Insurance",
        "liquidity": "liquid",
        "rating_required": True,
        "flexibility": "medium",
    },
    "hybrid": {
        "label": "Hybrid / Perpetual Bond",
        "typical_spread_bp": 280,
        "tenor_years": [60],              # perpetual, callable after 5/7
        "covenant_intensity": "low",
        "dilution_risk": False,
        "min_size_mEUR": 250,
        "investor_base": "Hybrid Specialists / Income Funds",
        "liquidity": "semi-liquid",
        "rating_required": True,
        "flexibility": "low",
    },
    "bank_rdf": {
        "label": "Bank RCF / Term Loan",
        "typical_spread_bp": 200,
        "tenor_years": [3, 5],
        "covenant_intensity": "high",
        "dilution_risk": False,
        "min_size_mEUR": 50,
        "investor_base": "Banks",
        "liquidity": "n/a",
        "rating_required": False,
        "flexibility": "medium",
    },
}

# ---------------------------------------------------------------------------
# Covenant thresholds (typical)
# ---------------------------------------------------------------------------

COVENANT_THRESHOLDS = {
    "net_debt_ebitda_max": 4.5,
    "interest_coverage_min": 2.5,
    "equity_ratio_min": 0.25,
    "cash_min_mEUR": 30.0,
}

# ---------------------------------------------------------------------------
# Monte Carlo parameters
# ---------------------------------------------------------------------------

MC_CONFIG = {
    "n_simulations": 10_000,
    "time_horizon_years": 5,
    "ebitda_vol": 0.12,               # annual EBITDA volatility (log-normal)
    "rate_vol": 0.008,                # annual short-rate volatility (Vasicek)
    "spread_vol_bp": 80,              # annual credit spread volatility
    "correlation_rate_spread": -0.35, # negative: rate up → spread compression
    "correlation_ebitda_spread": -0.60,
    "seed": 42,
}

# ---------------------------------------------------------------------------
# Vasicek short-rate model parameters (calibrated to EUR curve)
# ---------------------------------------------------------------------------

VASICEK_PARAMS = {
    "kappa": 0.35,    # mean reversion speed
    "theta": 0.028,   # long-run mean (terminal rate)
    "sigma": 0.008,   # volatility
    "r0": 0.024,      # current short rate
}

# ---------------------------------------------------------------------------
# Nelson-Siegel initial parameters (EUR swap curve)
# ---------------------------------------------------------------------------

NELSON_SIEGEL_PARAMS = {
    "beta0": 2.80,    # long-run level (%)
    "beta1": -0.70,   # slope
    "beta2": 0.50,    # curvature
    "tau": 2.50,      # decay factor
}
