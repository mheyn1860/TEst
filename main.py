"""
Integrated Capital Markets Refinancing Model — Main Orchestrator
================================================================

Run:  python main.py
Output: printed tables + charts saved to ./output/
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── local package ──────────────────────────────────────────────────────────
from refinancing_model import (
    compute_credit_metrics, credit_metrics_to_df, implied_rating, project_financials,
    NelsonSiegel, Vasicek, CIR, build_scenario_curves, forward_rate_at_horizon,
    MertonModel, HazardRateModel, fundamental_spread_bp, spread_scenario_matrix,
    FundingOptimiser, build_instrument_universe, covenant_stress_test,
    MonteCarloEngine, blended_scenario_summary,
    plot_maturity_profile, plot_yield_curve_fan, plot_mc_cost_fan,
    plot_funding_mix, plot_covenant_heatmap, plot_scenario_outcomes,
    plot_efficient_frontier, plot_dashboard,
)
from example_company import (
    BALANCE_SHEET_2024, INCOME_STATEMENT_2024, CASH_FLOW_2024,
    DEBT_SCHEDULE, MERTON_INPUTS, PROJECTION, COMPANY,
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def save_dir() -> str:
    d = "./output"
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Step 1: Financial Analysis & Credit Metrics
# ---------------------------------------------------------------------------

def step1_financial_analysis():
    section("STEP 1 — FINANCIAL ANALYSIS & CREDIT METRICS")

    metrics = compute_credit_metrics(BALANCE_SHEET_2024, INCOME_STATEMENT_2024, CASH_FLOW_2024)

    print(f"\nCompany: {COMPANY['name']} | Sector: {COMPANY['sector']}")
    print(f"Reporting Year: {BALANCE_SHEET_2024.year}\n")

    key_metrics = {
        "Revenue (m€)":            f"{INCOME_STATEMENT_2024.revenue:.1f}",
        "EBITDA (m€)":             f"{INCOME_STATEMENT_2024.ebitda:.1f}",
        "EBITDA Margin (%)":       f"{INCOME_STATEMENT_2024.ebitda / INCOME_STATEMENT_2024.revenue * 100:.1f}",
        "Net Debt (m€)":           f"{metrics.net_debt:.1f}",
        "Net Debt / EBITDA":       f"{metrics.net_debt_ebitda:.2f}x",
        "EBITDA / Interest":       f"{metrics.ebitda_interest_coverage:.2f}x",
        "FCF Conversion (%)":      f"{metrics.fcf_conversion * 100:.1f}",
        "Liquidity Runway (mths)": f"{metrics.liquidity_runway_months:.1f}",
        "Implied Rating":          implied_rating(metrics),
    }

    for k, v in key_metrics.items():
        print(f"  {k:<28} {v:>12}")

    print(f"\nDebt Schedule Summary:")
    mp = DEBT_SCHEDULE.maturity_profile()
    print(mp.to_string(index=False))
    print(f"\n  WAM (Weighted Avg Maturity): {DEBT_SCHEDULE.weighted_avg_maturity():.1f}Y")
    print(f"  WACD (Weighted Avg Cost):    {DEBT_SCHEDULE.weighted_avg_cost()*100:.2f}%")

    # Multi-year projection
    print(f"\nBase Case 5Y Projection:")
    proj = project_financials(
        INCOME_STATEMENT_2024, BALANCE_SHEET_2024, CASH_FLOW_2024,
        **PROJECTION,
    )
    proj_rows = []
    for is_, bs, cf, m in proj:
        proj_rows.append({
            "Year": is_.year,
            "Revenue": f"{is_.revenue:.0f}",
            "EBITDA": f"{is_.ebitda:.0f}",
            "ND/EBITDA": f"{m.net_debt_ebitda:.2f}x",
            "IC": f"{m.ebitda_interest_coverage:.2f}x",
            "FCF": f"{cf.free_cash_flow:.0f}",
            "Rating": implied_rating(m),
        })
    print(pd.DataFrame(proj_rows).to_string(index=False))

    return metrics


# ---------------------------------------------------------------------------
# Step 2: Yield Curve Modelling
# ---------------------------------------------------------------------------

def step2_yield_curves():
    section("STEP 2 — YIELD CURVE MODELLING")

    ns = NelsonSiegel()
    print("\nNelson-Siegel EUR Swap Curve (Base):")
    print(ns.curve_dataframe().to_string(index=False))

    vasicek = Vasicek()
    print(f"\nVasicek Model Parameters: κ={vasicek.kappa}, θ={vasicek.theta*100:.2f}%, σ={vasicek.sigma*100:.2f}%")
    print(vasicek.curve_dataframe().to_string(index=False))

    scenario_curves = build_scenario_curves(ns)
    print("\nForward Rates at 2Y Horizon across scenarios (5Y instrument):")
    for sc_key, curve in scenario_curves.items():
        fwd = forward_rate_at_horizon(curve, horizon_y=2, tenor_y=5)
        print(f"  {sc_key:<20}: {fwd:.3f}%")

    return ns, vasicek, scenario_curves


# ---------------------------------------------------------------------------
# Step 3: Credit Spread Modelling
# ---------------------------------------------------------------------------

def step3_credit_modelling(metrics):
    section("STEP 3 — CREDIT SPREAD MODELLING")

    # Merton model
    m_inputs = MERTON_INPUTS
    merton = MertonModel(**m_inputs)
    print("\nMerton Structural Model:")
    for k, v in merton.summary().items():
        print(f"  {k:<35} {v}")

    # Fundamental spread
    fund_spread = fundamental_spread_bp(
        net_debt_ebitda=metrics.net_debt_ebitda,
        ebitda_interest_coverage=metrics.ebitda_interest_coverage,
        ffo_debt=metrics.ffo_debt,
        fcf_conversion=metrics.fcf_conversion,
        equity_volatility=COMPANY["equity_volatility"],
        market_cap_mEUR=COMPANY["market_cap_mEUR"],
        ebitda_growth_rate=COMPANY["ebitda_growth_rate"],
        scenario_key="base_case",
    )
    print(f"\nFundamental Credit Spread (Base Case): {fund_spread:.0f}bp")

    # Hazard rate
    hr = HazardRateModel.from_spread(fund_spread)
    print(f"\nHazard Rate Model (λ={hr.lam*100:.3f}%, R={hr.R*100:.0f}%):")
    for tenor in [1, 3, 5, 7]:
        cum_pd = hr.cumulative_pd(np.array([float(tenor)]))[0]
        print(f"  {tenor}Y Cumulative PD: {cum_pd*100:.3f}%  | Par Spread: {hr.par_spread_bp(tenor):.1f}bp")

    # Spread scenario matrix
    print(f"\nAll-in Funding Cost Matrix (credit spread={fund_spread:.0f}bp):")
    sm = spread_scenario_matrix(fund_spread, tenors=[3, 5, 7, 10])
    print(sm.to_string(index=False))

    return fund_spread


# ---------------------------------------------------------------------------
# Step 4: Optimisation
# ---------------------------------------------------------------------------

def step4_optimisation(fund_spread):
    section("STEP 4 — FUNDING MIX OPTIMISATION")

    ns = NelsonSiegel()
    rf_5y = ns.spot_rate(np.array([5.0]))[0]

    inst_df = build_instrument_universe(
        credit_spread_bp=fund_spread,
        base_risk_free_pct=rf_5y,
        tenor_preference=5,
        include_equity_linked=True,
    )
    print("\nInstrument Universe:")
    print(inst_df[["Instrument", "Tenor (Y)", "Coupon (%)", "Spread (bp)", "Refi Risk", "Liquidity Risk", "Dilution Cost (%)"]].to_string(index=False))

    optimiser = FundingOptimiser(
        instrument_df=inst_df,
        total_refinancing_need_mEUR=COMPANY["total_refinancing_need_mEUR"],
        ebitda_mEUR=INCOME_STATEMENT_2024.ebitda,
        current_net_debt_mEUR=BALANCE_SHEET_2024.short_term_debt + BALANCE_SHEET_2024.long_term_debt,
        min_cash_mEUR=COMPANY["min_cash_mEUR"],
    )
    result = optimiser.optimise()

    print(f"\nOptimal Funding Mix (Total: {result['total_refinancing_mEUR']:.0f}m€):")
    print(result["allocation_df"].to_string(index=False))
    print(f"\n  Weighted Coupon:   {result['weighted_coupon_pct']:.3f}%")
    print(f"  Weighted Tenor:    {result['weighted_tenor_y']:.1f}Y")
    print(f"  Objective Value:   {result['objective_value']:.6f}")
    print(f"  Optimiser Status:  {'SUCCESS' if result['optimiser_success'] else 'FALLBACK'}")

    # Covenant stress test
    from refinancing_model.financial_analysis import compute_credit_metrics
    metrics = compute_credit_metrics(BALANCE_SHEET_2024, INCOME_STATEMENT_2024, CASH_FLOW_2024)
    stress = covenant_stress_test(metrics)
    print("\nCovenant Stress Test (breach = red):")
    print(stress.to_string(index=False))

    frontier = optimiser.efficient_frontier(n_points=20)
    print(f"\nEfficient Frontier (first 5 points):")
    print(frontier.head(5).to_string(index=False))

    return result, inst_df, stress, frontier, optimiser


# ---------------------------------------------------------------------------
# Step 5: Monte Carlo Simulation
# ---------------------------------------------------------------------------

def step5_monte_carlo(fund_spread):
    section("STEP 5 — MONTE CARLO SIMULATION (10,000 paths × 5Y)")

    engine = MonteCarloEngine()

    base_ebitda = INCOME_STATEMENT_2024.ebitda
    net_debt = BALANCE_SHEET_2024.short_term_debt + BALANCE_SHEET_2024.long_term_debt

    print(f"\nRunning base case simulation (spread={fund_spread:.0f}bp, EBITDA={base_ebitda:.0f}m€)...")
    sim = engine.run(
        base_spread_bp=fund_spread,
        base_ebitda=base_ebitda,
        ebitda_drift=COMPANY["ebitda_growth_rate"],
        tenor_years=5,
    )

    print("\nAll-in Cost Distribution at T=2.5Y:")
    mid_step = engine.n_steps // 2
    stats = sim.cost_statistics(mid_step)
    for k, v in stats.items():
        print(f"  {k:<28} {v:.4f}%")

    print("\nOptimal Issuance Windows (cost in bottom 25th pctile):")
    window = sim.optimal_issuance_window(cost_percentile=25)
    time = sim.time_axis()
    window_years = time[window]
    if len(window_years) > 0:
        print(f"  Attractive windows: Y{window_years[0]:.2f} – Y{window_years[-1]:.2f} (approx)")

    breach_probs = sim.covenant_breach_probability(net_debt)
    default_probs = sim.default_probability(net_debt)
    print(f"\nCovenant Breach Probability at key horizons:")
    for horizon in [1, 2, 3, 5]:
        step = min(int(horizon * engine.steps_per_year), engine.n_steps)
        print(f"  {horizon}Y: {breach_probs[step]*100:.2f}%  |  Default Proxy: {default_probs[step]*100:.2f}%")

    # Scenario-blended summary
    print("\nRunning scenario-conditioned simulations...")
    summary = blended_scenario_summary(engine, fund_spread, net_debt, base_ebitda)
    print("\nScenario Outcome Summary:")
    display_cols = ["Scenario", "Median Cost (%)", "P(Successful Refi) %",
                    "P(Expensive Refi) %", "P(Distressed Refi) %"]
    print(summary[display_cols].to_string(index=False))

    return sim, summary, engine


# ---------------------------------------------------------------------------
# Step 6: Visualisation
# ---------------------------------------------------------------------------

def step6_visualisation(
    scenario_curves, sim, summary, allocation_df, stress_df, frontier_df
):
    section("STEP 6 — GENERATING CHARTS")

    out = save_dir()
    mp = DEBT_SCHEDULE.maturity_profile()
    liq = BALANCE_SHEET_2024.cash_and_equivalents

    try:
        print("  Saving: maturity_profile.png")
        plot_maturity_profile(mp, liq, save_path=f"{out}/maturity_profile.png")

        print("  Saving: yield_curve_fan.png")
        plot_yield_curve_fan(scenario_curves, save_path=f"{out}/yield_curve_fan.png")

        print("  Saving: mc_cost_fan.png")
        plot_mc_cost_fan(sim, save_path=f"{out}/mc_cost_fan.png")

        print("  Saving: funding_mix.png")
        plot_funding_mix(allocation_df, save_path=f"{out}/funding_mix.png")

        print("  Saving: covenant_heatmap.png")
        plot_covenant_heatmap(stress_df, save_path=f"{out}/covenant_heatmap.png")

        print("  Saving: scenario_outcomes.png")
        plot_scenario_outcomes(summary, save_path=f"{out}/scenario_outcomes.png")

        print("  Saving: efficient_frontier.png")
        plot_efficient_frontier(frontier_df, save_path=f"{out}/efficient_frontier.png")

        print("  Saving: dashboard.png")
        plot_dashboard(
            mp, scenario_curves, sim, allocation_df, stress_df, summary,
            liquidity_buffer=liq,
            save_path=f"{out}/dashboard.png",
        )

        print(f"\n  All charts saved to: {os.path.abspath(out)}/")

    except Exception as e:
        print(f"  Chart generation error: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_model():
    print("\n" + "█" * 70)
    print("  INTEGRATED CAPITAL MARKETS REFINANCING MODEL")
    print(f"  {COMPANY['name']} | {COMPANY['sector']} | {COMPANY['currency']}")
    print("█" * 70)

    metrics       = step1_financial_analysis()
    ns, vasicek, scenario_curves = step2_yield_curves()
    fund_spread   = step3_credit_modelling(metrics)
    opt_result, inst_df, stress, frontier, optimiser = step4_optimisation(fund_spread)
    sim, summary, engine = step5_monte_carlo(fund_spread)
    step6_visualisation(
        scenario_curves, sim, summary,
        opt_result["allocation_df"], stress, frontier,
    )

    section("MODEL COMPLETE — KEY OUTPUTS")
    print(f"""
  Implied Rating:         {__import__('refinancing_model').implied_rating(metrics)}
  Fundamental Spread:     {fund_spread:.0f}bp
  Optimal Funding Mix:    {opt_result['weighted_coupon_pct']:.3f}% all-in / {opt_result['weighted_tenor_y']:.1f}Y WAM
  Refinancing Volume:     {COMPANY['total_refinancing_need_mEUR']:.0f}m€
  P(Successful Refi):     {summary[summary['Scenario']=='★ Probability-Weighted']['P(Successful Refi) %'].values[0]:.1f}%
  P(Distressed Refi):     {summary[summary['Scenario']=='★ Probability-Weighted']['P(Distressed Refi) %'].values[0]:.1f}%
  Dashboard:              ./output/dashboard.png
""")

    return {
        "metrics": metrics,
        "fund_spread_bp": fund_spread,
        "optimal_allocation": opt_result,
        "simulation": sim,
        "scenario_summary": summary,
    }


if __name__ == "__main__":
    results = run_full_model()
