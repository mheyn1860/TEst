"""
Module 6 — Visualisation.

Produces publication-quality charts in Investment Banking / Rating Agency style:
  1.  Debt Maturity Profile (bar chart + heatmap)
  2.  Yield Curve Fan Chart (multi-scenario)
  3.  Credit Spread Distribution (histogram + scenario overlays)
  4.  Refinancing Cost Curve (current vs. projected)
  5.  Optimal Funding Mix (donut / waterfall)
  6.  Covenant Headroom Dashboard
  7.  Monte Carlo Cost Fan Chart
  8.  Scenario Outcome Matrix (heatmap)
  9.  Efficient Frontier
  10. Summary Dashboard (multi-panel)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional

from .config import MACRO_SCENARIOS


# ---------------------------------------------------------------------------
# Style constants (IB-style: dark navy / light grey palette)
# ---------------------------------------------------------------------------

COLORS = {
    "navy":       "#0D2137",
    "blue":       "#1F5FA8",
    "light_blue": "#4A9FD5",
    "gold":       "#C9A227",
    "green":      "#2E8B57",
    "red":        "#C0392B",
    "orange":     "#E67E22",
    "grey":       "#6B7280",
    "light_grey": "#F3F4F6",
    "white":      "#FFFFFF",
}

SCENARIO_COLORS = {
    "base_case":      COLORS["blue"],
    "re_acceleration": COLORS["green"],
    "recession":      COLORS["red"],
    "stagflation":    COLORS["orange"],
}

FONT = {"family": "DejaVu Sans", "size": 9}
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": COLORS["white"],
    "figure.facecolor": COLORS["white"],
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _save_or_show(fig, path: Optional[str]):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 1. Debt Maturity Profile
# ---------------------------------------------------------------------------

def plot_maturity_profile(
    maturity_df: pd.DataFrame,          # columns: maturity, debt_maturing
    liquidity_buffer: float,
    title: str = "Debt Maturity Profile",
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    years = maturity_df["maturity"].values
    amounts = maturity_df["debt_maturing"].values

    bars = ax.bar(years, amounts, color=COLORS["blue"], width=0.6, zorder=3, label="Debt Maturing")
    ax.axhline(liquidity_buffer, color=COLORS["gold"], linestyle="--", linewidth=1.5,
               label=f"Liquidity Buffer ({liquidity_buffer:.0f}m€)", zorder=4)

    # Colour bars > liquidity buffer red
    for bar, amt in zip(bars, amounts):
        if amt > liquidity_buffer:
            bar.set_color(COLORS["red"])
            bar.set_alpha(0.85)

    ax.set_xlabel("Maturity Year")
    ax.set_ylabel("Amount (m€)")
    ax.set_title(title, fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}m€"))
    ax.set_xticks(years)

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Yield Curve Fan Chart (multi-scenario)
# ---------------------------------------------------------------------------

def plot_yield_curve_fan(
    scenario_curves: Dict,              # {scenario_key: NelsonSiegel}
    tenors: List[float] = None,
    save_path: Optional[str] = None,
):
    if tenors is None:
        tenors = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]
    tenors_arr = np.array(tenors)

    fig, ax = plt.subplots(figsize=(10, 5))

    for sc_key, curve in scenario_curves.items():
        sc = MACRO_SCENARIOS[sc_key]
        rates = curve.spot_rate(tenors_arr)
        ax.plot(
            tenors_arr, rates,
            color=SCENARIO_COLORS.get(sc_key, COLORS["grey"]),
            linewidth=2,
            label=f"{sc['label']} (p={sc['probability']*100:.0f}%)",
            marker="o", markersize=4,
        )

    ax.set_xlabel("Tenor (Years)")
    ax.set_ylabel("Spot Rate (%)")
    ax.set_title("EUR Yield Curve — Scenario Fan", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Monte Carlo Cost Fan Chart
# ---------------------------------------------------------------------------

def plot_mc_cost_fan(
    sim_result,                     # SimulationResult
    n_sample_paths: int = 100,
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(11, 5))

    time = sim_result.time_axis()
    n_steps = sim_result.n_steps

    costs = np.array([sim_result.all_in_cost_at_horizon(t) for t in range(n_steps + 1)])

    p5  = np.percentile(costs, 5,  axis=1)
    p25 = np.percentile(costs, 25, axis=1)
    p50 = np.percentile(costs, 50, axis=1)
    p75 = np.percentile(costs, 75, axis=1)
    p95 = np.percentile(costs, 95, axis=1)

    ax.fill_between(time, p5, p95, alpha=0.12, color=COLORS["blue"], label="5th–95th Pctile")
    ax.fill_between(time, p25, p75, alpha=0.25, color=COLORS["blue"], label="25th–75th Pctile")
    ax.plot(time, p50, color=COLORS["blue"], linewidth=2.5, label="Median")
    ax.plot(time, p5,  color=COLORS["blue"], linewidth=0.8, linestyle="--")
    ax.plot(time, p95, color=COLORS["blue"], linewidth=0.8, linestyle="--")

    # Sample paths
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(sim_result.n_paths, min(n_sample_paths, sim_result.n_paths), replace=False)
    for i in sample_idx:
        path_costs = np.array([sim_result.all_in_cost_at_horizon(t)[i] for t in range(n_steps + 1)])
        ax.plot(time, path_costs, alpha=0.05, linewidth=0.5, color=COLORS["grey"])

    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("All-in Funding Cost (%)")
    ax.set_title("Monte Carlo — All-in Funding Cost Fan Chart", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Optimal Funding Mix (Donut Chart)
# ---------------------------------------------------------------------------

def plot_funding_mix(
    allocation_df: pd.DataFrame,    # columns: Instrument, Weight (%), Volume (m€)
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Donut ---
    ax = axes[0]
    labels = allocation_df["Instrument"].tolist()
    sizes = allocation_df["Weight (%)"].values
    palette = [COLORS["navy"], COLORS["blue"], COLORS["light_blue"],
               COLORS["gold"], COLORS["green"], COLORS["orange"], COLORS["grey"]]
    colors = palette[:len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        startangle=90, colors=colors,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(
        wedges, labels, title="Instrument",
        loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False, fontsize=8,
    )
    ax.set_title("Optimal Funding Mix", fontsize=12, fontweight="bold", color=COLORS["navy"])

    # --- Volume bar chart ---
    ax2 = axes[1]
    bar_colors = colors
    bars = ax2.barh(labels, allocation_df["Volume (m€)"].values, color=bar_colors, height=0.5)
    ax2.set_xlabel("Volume (m€)")
    ax2.set_title("Financing Volume by Instrument", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax2.invert_yaxis()
    for bar, val in zip(bars, allocation_df["Volume (m€)"].values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}m€", va="center", fontsize=8)

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Covenant Headroom Heatmap
# ---------------------------------------------------------------------------

def plot_covenant_heatmap(
    stress_df: pd.DataFrame,        # from optimization.covenant_stress_test
    save_path: Optional[str] = None,
):
    pivot = stress_df.pivot(index="EBITDA Shock", columns="Rate Shock (bp)", values="Status")
    numeric = pivot.map(lambda x: 0 if x == "OK" else 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = LinearSegmentedColormap.from_list("covenant", [COLORS["green"], COLORS["red"]])
    im = ax.imshow(numeric.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(numeric.columns)))
    ax.set_xticklabels([f"+{c}bp" for c in numeric.columns])
    ax.set_yticks(range(len(numeric.index)))
    ax.set_yticklabels(numeric.index)
    ax.set_xlabel("Interest Rate Shock")
    ax.set_ylabel("EBITDA Shock")
    ax.set_title("Covenant Stress Test — Breach Heatmap", fontsize=12, fontweight="bold", color=COLORS["navy"])

    for i in range(numeric.shape[0]):
        for j in range(numeric.shape[1]):
            txt = pivot.values[i, j]
            color = "white" if numeric.values[i, j] == 1 else COLORS["navy"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    patches = [
        mpatches.Patch(color=COLORS["green"], label="OK"),
        mpatches.Patch(color=COLORS["red"], label="BREACH"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=False)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Scenario Outcome Probability Table (styled)
# ---------------------------------------------------------------------------

def plot_scenario_outcomes(
    summary_df: pd.DataFrame,       # from monte_carlo.blended_scenario_summary
    save_path: Optional[str] = None,
):
    display_cols = [
        "Scenario", "Median Cost (%)", "VaR-95 Cost (%)",
        "P(Successful Refi) %", "P(Expensive Refi) %", "P(Distressed Refi) %",
        "P(Covenant Breach @T/2) %",
    ]
    df = summary_df[display_cols].copy()

    fig, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.55 + 1.5)))
    ax.axis("off")

    col_widths = [0.22, 0.11, 0.11, 0.14, 0.13, 0.14, 0.15]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    header_color = COLORS["navy"]
    for j, _ in enumerate(df.columns):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Colour outcome cells
    for i in range(1, len(df) + 1):
        row_label = df.iloc[i - 1]["Scenario"]
        is_weighted = "Weighted" in row_label
        for j in range(len(df.columns)):
            cell = table[i, j]
            if is_weighted:
                cell.set_facecolor(COLORS["light_grey"])
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(COLORS["white"])

        # Distress probability → red tint
        p_dist_val = df.iloc[i - 1].get("P(Distressed Refi) %", 0)
        try:
            p_dist = float(p_dist_val)
            if p_dist > 10:
                table[i, 5].set_facecolor("#FFCCCC")
        except (ValueError, TypeError):
            pass

    ax.set_title("Scenario Outcome Matrix", fontsize=12, fontweight="bold",
                 color=COLORS["navy"], pad=15)
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Efficient Frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,      # from FundingOptimiser.efficient_frontier
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(
        frontier_df["Refi Risk Score"],
        frontier_df["Weighted Cost (%)"],
        c=frontier_df["Risk Aversion (α)"],
        cmap="Blues",
        s=40, zorder=3, edgecolors=COLORS["navy"], linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="Risk Aversion (α)")

    ax.set_xlabel("Refinancing Risk Score")
    ax.set_ylabel("Weighted Funding Cost (%)")
    ax.set_title("Funding Efficient Frontier", fontsize=12, fontweight="bold", color=COLORS["navy"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Full Dashboard (multi-panel)
# ---------------------------------------------------------------------------

def plot_dashboard(
    maturity_df: pd.DataFrame,
    scenario_curves: Dict,
    sim_result,
    allocation_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    liquidity_buffer: float,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # 1. Maturity Profile
    years = maturity_df["maturity"].values
    amounts = maturity_df["debt_maturing"].values
    bars = ax1.bar(years, amounts, color=COLORS["blue"], width=0.6, zorder=3)
    for bar, amt in zip(bars, amounts):
        if amt > liquidity_buffer:
            bar.set_color(COLORS["red"])
    ax1.axhline(liquidity_buffer, color=COLORS["gold"], linestyle="--", linewidth=1.5,
                label=f"Liquidity ({liquidity_buffer:.0f}m€)")
    ax1.set_title("Debt Maturity Profile", fontweight="bold", color=COLORS["navy"])
    ax1.set_xlabel("Year"); ax1.set_ylabel("m€")
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_xticks(years)

    # 2. Yield Curve Fan
    tenors_arr = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20])
    for sc_key, curve in scenario_curves.items():
        sc = MACRO_SCENARIOS[sc_key]
        rates = curve.spot_rate(tenors_arr)
        ax2.plot(tenors_arr, rates, color=SCENARIO_COLORS.get(sc_key, COLORS["grey"]),
                 linewidth=1.8, label=sc["label"][:20], marker="o", markersize=3)
    ax2.set_title("EUR Yield Curve Scenarios", fontweight="bold", color=COLORS["navy"])
    ax2.set_xlabel("Tenor (Y)"); ax2.set_ylabel("Rate (%)")
    ax2.legend(frameon=False, fontsize=7)

    # 3. MC Cost Fan
    time = sim_result.time_axis()
    n_steps = sim_result.n_steps
    costs = np.array([sim_result.all_in_cost_at_horizon(t) for t in range(n_steps + 1)])
    p5  = np.percentile(costs, 5,  axis=1)
    p25 = np.percentile(costs, 25, axis=1)
    p50 = np.percentile(costs, 50, axis=1)
    p75 = np.percentile(costs, 75, axis=1)
    p95 = np.percentile(costs, 95, axis=1)
    ax3.fill_between(time, p5, p95, alpha=0.12, color=COLORS["blue"])
    ax3.fill_between(time, p25, p75, alpha=0.25, color=COLORS["blue"])
    ax3.plot(time, p50, color=COLORS["blue"], linewidth=2)
    ax3.plot(time, p5,  color=COLORS["blue"], linewidth=0.7, linestyle="--")
    ax3.plot(time, p95, color=COLORS["blue"], linewidth=0.7, linestyle="--")
    ax3.set_title("MC Funding Cost Fan", fontweight="bold", color=COLORS["navy"])
    ax3.set_xlabel("Years"); ax3.set_ylabel("All-in Cost (%)")

    # 4. Funding Mix Donut
    labels = allocation_df["Instrument"].tolist()
    sizes = allocation_df["Weight (%)"].values
    palette = [COLORS["navy"], COLORS["blue"], COLORS["light_blue"],
               COLORS["gold"], COLORS["green"], COLORS["orange"], COLORS["grey"]]
    colors = palette[:len(labels)]
    ax4.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        startangle=90, colors=colors,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 7},
    )
    ax4.set_title("Optimal Funding Mix", fontweight="bold", color=COLORS["navy"])

    # 5. Covenant Heatmap
    pivot = stress_df.pivot(index="EBITDA Shock", columns="Rate Shock (bp)", values="Status")
    numeric = pivot.map(lambda x: 0 if x == "OK" else 1)
    cmap = LinearSegmentedColormap.from_list("cov", [COLORS["green"], COLORS["red"]])
    ax5.imshow(numeric.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax5.set_xticks(range(len(numeric.columns)))
    ax5.set_xticklabels([f"+{c}bp" for c in numeric.columns], fontsize=7)
    ax5.set_yticks(range(len(numeric.index)))
    ax5.set_yticklabels(numeric.index, fontsize=7)
    for i in range(numeric.shape[0]):
        for j in range(numeric.shape[1]):
            ax5.text(j, i, pivot.values[i, j], ha="center", va="center", fontsize=7,
                     color="white" if numeric.values[i, j] else COLORS["navy"], fontweight="bold")
    ax5.set_title("Covenant Stress (EBITDA × Rate)", fontweight="bold", color=COLORS["navy"])
    ax5.set_xlabel("Rate Shock"); ax5.set_ylabel("EBITDA Shock")

    # 6. Scenario outcome bars
    sc_labels = summary_df["Scenario"].tolist()[:-1]  # exclude weighted row
    p_succ  = summary_df["P(Successful Refi) %"].values[:-1]
    p_exp   = summary_df["P(Expensive Refi) %"].values[:-1]
    p_distr = summary_df["P(Distressed Refi) %"].values[:-1]
    x = np.arange(len(sc_labels))
    w = 0.25
    ax6.bar(x - w,   p_succ,  w, color=COLORS["green"],  label="Successful")
    ax6.bar(x,       p_exp,   w, color=COLORS["orange"], label="Expensive")
    ax6.bar(x + w,   p_distr, w, color=COLORS["red"],    label="Distressed")
    ax6.set_xticks(x)
    ax6.set_xticklabels([s[:15] for s in sc_labels], rotation=15, ha="right", fontsize=7)
    ax6.set_ylabel("Probability (%)")
    ax6.set_title("Refinancing Outcome Probabilities", fontweight="bold", color=COLORS["navy"])
    ax6.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Integrated Capital Markets Refinancing Model — Executive Dashboard",
        fontsize=14, fontweight="bold", color=COLORS["navy"], y=1.01,
    )

    _save_or_show(fig, save_path)
    return fig
