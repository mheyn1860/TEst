"""
Module 1 — Financial Statement Analysis & Credit Metrics.

Ingests a company's balance sheet, P&L and cash-flow statement and derives
all credit and liquidity metrics needed by the rest of the model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BalanceSheet:
    year: int
    total_assets: float
    cash_and_equivalents: float
    short_term_debt: float             # current portion of LT debt + CP
    long_term_debt: float
    finance_lease_liabilities: float
    total_equity: float
    goodwill_and_intangibles: float
    working_capital: float             # current assets - current liabilities (excl. debt)


@dataclass
class IncomeStatement:
    year: int
    revenue: float
    ebitda: float
    ebit: float
    interest_expense: float
    tax_expense: float
    net_income: float
    depreciation_amortization: float


@dataclass
class CashFlowStatement:
    year: int
    cfo: float          # operating cash flow
    capex: float        # capital expenditure (negative = outflow)
    dividends: float
    free_cash_flow: float = 0.0

    def __post_init__(self):
        if self.free_cash_flow == 0.0:
            self.free_cash_flow = self.cfo + self.capex  # capex already negative


@dataclass
class DebtSchedule:
    """Maps each debt tranche to its maturity year and outstanding amount."""
    tranches: List[Dict]  # [{"name": str, "amount": float, "maturity": int, "rate": float, "type": str}]

    def maturity_profile(self) -> pd.DataFrame:
        df = pd.DataFrame(self.tranches)
        return df.groupby("maturity")["amount"].sum().reset_index().rename(
            columns={"amount": "debt_maturing"}
        )

    def total_debt(self) -> float:
        return sum(t["amount"] for t in self.tranches)

    def weighted_avg_maturity(self) -> float:
        total = self.total_debt()
        if total == 0:
            return 0.0
        return sum(t["amount"] * t["maturity"] for t in self.tranches) / total

    def weighted_avg_cost(self) -> float:
        total = self.total_debt()
        if total == 0:
            return 0.0
        return sum(t["amount"] * t["rate"] for t in self.tranches) / total


# ---------------------------------------------------------------------------
# Credit metrics computation
# ---------------------------------------------------------------------------

@dataclass
class CreditMetrics:
    year: int
    net_debt: float
    gross_debt: float
    net_debt_ebitda: float
    gross_debt_ebitda: float
    ebitda_interest_coverage: float
    ffo_debt: float                     # FFO / Gross Debt (Moody's style)
    debt_capital: float                 # Debt / (Debt + Equity)
    fcf_conversion: float               # FCF / EBITDA
    liquidity_runway_months: float
    tangible_equity: float
    roe: float
    roce: float


def compute_credit_metrics(
    bs: BalanceSheet,
    is_: IncomeStatement,
    cf: CashFlowStatement,
) -> CreditMetrics:
    gross_debt = bs.short_term_debt + bs.long_term_debt + bs.finance_lease_liabilities
    net_debt = gross_debt - bs.cash_and_equivalents
    ebitda = is_.ebitda

    net_debt_ebitda = net_debt / ebitda if ebitda > 0 else np.inf
    gross_debt_ebitda = gross_debt / ebitda if ebitda > 0 else np.inf
    ebitda_interest = ebitda / is_.interest_expense if is_.interest_expense > 0 else np.inf

    ffo = is_.net_income + is_.depreciation_amortization  # simplified FFO
    ffo_debt = ffo / gross_debt if gross_debt > 0 else np.inf

    total_capital = gross_debt + bs.total_equity
    debt_capital = gross_debt / total_capital if total_capital > 0 else np.nan

    fcf_conversion = cf.free_cash_flow / ebitda if ebitda > 0 else np.nan

    monthly_cash_burn = max(-cf.free_cash_flow / 12, 1)  # prevent div-by-zero
    liquidity_runway = bs.cash_and_equivalents / monthly_cash_burn

    tangible_equity = bs.total_equity - bs.goodwill_and_intangibles

    roe = is_.net_income / bs.total_equity if bs.total_equity > 0 else np.nan
    invested_capital = total_capital
    roce = is_.ebit / invested_capital if invested_capital > 0 else np.nan

    return CreditMetrics(
        year=bs.year,
        net_debt=net_debt,
        gross_debt=gross_debt,
        net_debt_ebitda=net_debt_ebitda,
        gross_debt_ebitda=gross_debt_ebitda,
        ebitda_interest_coverage=ebitda_interest,
        ffo_debt=ffo_debt,
        debt_capital=debt_capital,
        fcf_conversion=fcf_conversion,
        liquidity_runway_months=liquidity_runway,
        tangible_equity=tangible_equity,
        roe=roe,
        roce=roce,
    )


def credit_metrics_to_df(metrics_list: List[CreditMetrics]) -> pd.DataFrame:
    rows = []
    for m in metrics_list:
        rows.append({
            "Year": m.year,
            "Net Debt (m€)": round(m.net_debt, 1),
            "Gross Debt (m€)": round(m.gross_debt, 1),
            "Net Debt / EBITDA": round(m.net_debt_ebitda, 2),
            "Gross Debt / EBITDA": round(m.gross_debt_ebitda, 2),
            "EBITDA / Interest": round(m.ebitda_interest_coverage, 2),
            "FFO / Debt (%)": round(m.ffo_debt * 100, 1),
            "Debt / Capital (%)": round(m.debt_capital * 100, 1),
            "FCF Conversion (%)": round(m.fcf_conversion * 100, 1),
            "Liquidity Runway (mths)": round(m.liquidity_runway_months, 1),
            "Tangible Equity (m€)": round(m.tangible_equity, 1),
        })
    return pd.DataFrame(rows).set_index("Year")


# ---------------------------------------------------------------------------
# Implied rating from credit metrics (simplified shadow rating)
# ---------------------------------------------------------------------------

RATING_GRID = [
    # (net_debt_ebitda_max, interest_coverage_min, ffo_debt_min, implied_rating)
    (1.0, 10.0, 0.45, "AAA/AA"),
    (1.5,  8.0, 0.35, "A+/A"),
    (2.0,  6.0, 0.25, "A-/BBB+"),
    (2.5,  5.0, 0.20, "BBB"),
    (3.0,  4.0, 0.15, "BBB-"),
    (3.5,  3.0, 0.12, "BB+"),
    (4.0,  2.5, 0.08, "BB"),
    (5.0,  2.0, 0.05, "BB-/B+"),
    (6.5,  1.5, 0.02, "B"),
    (np.inf, 0.0, 0.0, "CCC and below"),
]


def implied_rating(metrics: CreditMetrics) -> str:
    for nd_ebitda, ic, ffo_d, rating in RATING_GRID:
        if (
            metrics.net_debt_ebitda <= nd_ebitda
            and metrics.ebitda_interest_coverage >= ic
            and metrics.ffo_debt >= ffo_d
        ):
            return rating
    return "CCC and below"


# ---------------------------------------------------------------------------
# Multi-year financial model projection
# ---------------------------------------------------------------------------

def project_financials(
    base_is: IncomeStatement,
    base_bs: BalanceSheet,
    base_cf: CashFlowStatement,
    revenue_growth_rates: List[float],
    ebitda_margin: float,
    capex_pct_revenue: float,
    debt_repayment_per_year: float,
    interest_rate_on_debt: float,
) -> List[tuple]:
    """Simple straight-line projection for base case planning."""
    results = []
    revenue = base_is.revenue
    cash = base_bs.cash_and_equivalents
    debt = base_bs.short_term_debt + base_bs.long_term_debt

    for i, growth in enumerate(revenue_growth_rates):
        year = base_is.year + i + 1
        revenue *= (1 + growth)
        ebitda = revenue * ebitda_margin
        da = base_is.depreciation_amortization * (1 + growth * 0.5)
        ebit = ebitda - da
        debt = max(0, debt - debt_repayment_per_year)
        interest = debt * interest_rate_on_debt
        tax = max(0, (ebit - interest) * 0.28)
        net_income = ebit - interest - tax
        capex = -revenue * capex_pct_revenue
        cfo = net_income + da
        fcf = cfo + capex
        cash += fcf - debt_repayment_per_year * 0  # simplification

        is_ = IncomeStatement(year, revenue, ebitda, ebit, interest, tax, net_income, da)
        bs = BalanceSheet(
            year=year,
            total_assets=base_bs.total_assets * (1 + growth * 0.3),
            cash_and_equivalents=max(cash, 10),
            short_term_debt=min(debt, 50),
            long_term_debt=max(0, debt - 50),
            finance_lease_liabilities=base_bs.finance_lease_liabilities,
            total_equity=base_bs.total_equity + net_income,
            goodwill_and_intangibles=base_bs.goodwill_and_intangibles,
            working_capital=base_bs.working_capital * (1 + growth * 0.4),
        )
        cf = CashFlowStatement(year, cfo, capex, 0.0)
        metrics = compute_credit_metrics(bs, is_, cf)
        results.append((is_, bs, cf, metrics))

    return results
