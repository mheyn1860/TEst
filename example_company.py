"""
Example company data — mid-cap German industrial / B2B software company.
Revenue ~€800m, EBITDA ~€150m, Net Debt ~€480m → ND/EBITDA ~3.2x (BB-range).
Refinancing need: €300m over 2025-2028.
"""

from refinancing_model.financial_analysis import (
    BalanceSheet, IncomeStatement, CashFlowStatement, DebtSchedule
)

# ---------------------------------------------------------------------------
# Historical financials (last reported year: 2024)
# ---------------------------------------------------------------------------

BALANCE_SHEET_2024 = BalanceSheet(
    year=2024,
    total_assets=1_420.0,
    cash_and_equivalents=85.0,
    short_term_debt=45.0,           # RCF drawn + current portion
    long_term_debt=520.0,           # Nordic Bond 250m + Schuldschein 150m + PP 120m
    finance_lease_liabilities=35.0,
    total_equity=390.0,
    goodwill_and_intangibles=310.0,
    working_capital=95.0,
)

INCOME_STATEMENT_2024 = IncomeStatement(
    year=2024,
    revenue=810.0,
    ebitda=152.0,
    ebit=95.0,
    interest_expense=28.5,
    tax_expense=18.5,
    net_income=48.0,
    depreciation_amortization=57.0,
)

CASH_FLOW_2024 = CashFlowStatement(
    year=2024,
    cfo=108.0,
    capex=-42.0,
    dividends=-8.0,
)

# ---------------------------------------------------------------------------
# Debt schedule (each tranche with maturity and coupon)
# ---------------------------------------------------------------------------

DEBT_SCHEDULE = DebtSchedule(tranches=[
    {"name": "RCF (drawn)",           "amount":  45.0, "maturity": 2026, "rate": 0.0575, "type": "bank"},
    {"name": "Schuldschein 3Y",        "amount":  60.0, "maturity": 2026, "rate": 0.0480, "type": "schuldschein"},
    {"name": "Schuldschein 5Y",        "amount":  90.0, "maturity": 2027, "rate": 0.0510, "type": "schuldschein"},
    {"name": "Nordic Bond 5Y",         "amount": 250.0, "maturity": 2027, "rate": 0.0695, "type": "nordic_bond"},
    {"name": "US Private Placement 7Y","amount": 120.0, "maturity": 2028, "rate": 0.0530, "type": "private_placement"},
    {"name": "Finance Leases",          "amount":  35.0, "maturity": 2029, "rate": 0.0340, "type": "lease"},
])

# ---------------------------------------------------------------------------
# Market data for Merton model
# ---------------------------------------------------------------------------

MERTON_INPUTS = {
    "E":       390.0,       # market cap (approx = equity book here; use market cap in practice)
    "D":       600.0,       # total debt face value (1Y default barrier)
    "r":       0.024,       # risk-free (1Y Bund)
    "T":       1.0,
    "sigma_E": 0.28,        # annualised equity vol (30D realised)
}

# ---------------------------------------------------------------------------
# Projection assumptions
# ---------------------------------------------------------------------------

PROJECTION = {
    "revenue_growth_rates": [0.06, 0.07, 0.06, 0.05, 0.05],   # 5Y
    "ebitda_margin": 0.19,
    "capex_pct_revenue": 0.055,
    "debt_repayment_per_year": 30.0,
    "interest_rate_on_debt": 0.055,
}

# ---------------------------------------------------------------------------
# Company metadata
# ---------------------------------------------------------------------------

COMPANY = {
    "name": "Example AG",
    "sector": "Industrial / B2B Software",
    "country": "Germany",
    "currency": "EUR",
    "total_refinancing_need_mEUR": 300.0,   # total to refinance 2025-2028
    "min_cash_mEUR": 50.0,
    "equity_volatility": 0.28,
    "market_cap_mEUR": 420.0,
    "ebitda_growth_rate": 0.065,
}
