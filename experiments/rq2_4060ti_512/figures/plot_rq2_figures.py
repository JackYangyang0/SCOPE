#!/usr/bin/env python3
"""Generate compact, publication-quality RQ2/RQ3 ablation figures."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
RQ2_COMPLETE_CSV = OUTPUT_DIR.parent / "rq2_complete_runs.csv"

# Muted, colorblind-safe palette for paper figures.
NAVY = "#2F5D8A"
BLUE = "#4C88B8"
SKY = "#A8D3E8"
TEAL = "#2A9D8F"
GREEN = "#3A9D78"
WARM_GRAY = "#A6ADB4"
LIGHT_GRAY = "#DDE2E7"
GRID = "#E4E8EC"
TEXT = "#25313C"
SECONDARY = "#64717D"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7B8792",
            "axes.linewidth": 0.7,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "hatch.linewidth": 0.7,
        }
    )


def style_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    if axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.65)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=GRID, linewidth=0.65)
        ax.yaxis.grid(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def load_complete_rows() -> dict[str, dict[str, str]]:
    with RQ2_COMPLETE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return {row["variant"]: row for row in csv.DictReader(handle)}


def tflops(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) / 1000.0 if value else np.nan


def plot_ablation_summary_table() -> None:
    """Render the RQ2 evidence as one neutral, publication-style table."""
    rows = [
        ("Construction", "Full$^{\u2020}$", "3/3", "--", "10.577", "--", "100.0%"),
        ("", "w/o State-in-Prompt", "1/3", "5.642", "5.986", "+6.09%", "56.6%"),
        ("", "Intent-only Control$^{\u2021}$", "0/0", "--", "N/A", "--", "--"),
        ("", "w/o Local Checks", "3/3", "5.903", "7.470", "+26.55%", "70.6%"),
        ("", "w/o Stage Ordering", "2/3", "6.045", "6.912", "+14.33%", "65.3%"),
        ("Parameter search", "w/o Parameter Search", "--", "--", "8.897", "--", "84.1%"),
        ("", "Full$^{\u2020}$", "--", "--", "10.577", "--", "100.0%"),
    ]

    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column anchors. Numeric columns are right-aligned for fast comparison.
    x_group, x_config = 0.015, 0.175
    x_accept, x_before, x_selected, x_gain, x_full = 0.535, 0.655, 0.770, 0.885, 0.985
    header_y = 0.895
    row_top = 0.805
    row_h = 0.092

    ax.text(x_group, header_y, "Component", ha="left", va="center",
            fontsize=8.2, weight="semibold")
    ax.text(x_config, header_y, "Configuration", ha="left", va="center",
            fontsize=8.2, weight="semibold")
    ax.text(x_accept, header_y, "Accepted /\nterminal", ha="right", va="center",
            fontsize=7.7, weight="semibold", linespacing=0.95)
    ax.text(x_before, header_y, "Before\nfeedback", ha="right", va="center",
            fontsize=7.7, weight="semibold", linespacing=0.95)
    ax.text(x_selected, header_y, "Selected\nthroughput", ha="right", va="center",
            fontsize=7.7, weight="semibold", linespacing=0.95)
    ax.text(x_gain, header_y, "Feedback\ngain", ha="right", va="center",
            fontsize=7.7, weight="semibold", linespacing=0.95)
    ax.text(x_full, header_y, "Relative\nto Full", ha="right", va="center",
            fontsize=7.7, weight="semibold", linespacing=0.95)

    # Booktabs-like rules: no vertical borders and no decorative color blocks.
    ax.plot([0.01, 0.99], [0.965, 0.965], color=TEXT, linewidth=1.05, clip_on=False)
    ax.plot([0.01, 0.99], [0.842, 0.842], color=TEXT, linewidth=0.68, clip_on=False)

    for idx, (group, config, accepted_text, before_text, selected_text, gain_text, full_text) in enumerate(rows):
        y = row_top - idx * row_h
        if idx == 5:
            ax.plot([0.01, 0.99], [y + row_h * 0.54, y + row_h * 0.54],
                    color="#8A929A", linewidth=0.55, clip_on=False)
        if group:
            ax.text(x_group, y, group, ha="left", va="center", fontsize=7.7,
                    weight="semibold" if idx in (0, 5) else "normal")
        config_weight = "semibold" if config.startswith("Full") else "normal"
        ax.text(x_config, y, config, ha="left", va="center", fontsize=7.7,
                weight=config_weight)
        ax.text(x_accept, y, accepted_text, ha="right", va="center", fontsize=7.7)
        ax.text(x_before, y, before_text, ha="right", va="center", fontsize=7.7)
        ax.text(x_selected, y, selected_text, ha="right", va="center", fontsize=7.7,
                weight="semibold" if selected_text == "10.577" else "normal")
        ax.text(x_gain, y, gain_text, ha="right", va="center", fontsize=7.7)
        ax.text(x_full, y, full_text, ha="right", va="center", fontsize=7.7)

    bottom_rule = row_top - (len(rows) - 0.45) * row_h
    ax.plot([0.01, 0.99], [bottom_rule, bottom_rule], color=TEXT,
            linewidth=1.05, clip_on=False)
    ax.text(0.01, 0.115,
            "Throughput is reported in TFLOP/s. Feedback columns apply only to feedback-enabled ablations.",
            ha="left", va="center", fontsize=6.5, color=SECONDARY)
    ax.text(0.01, 0.067,
            "$^{\u2020}$Archived historical Full result; $^{\u2021}$no terminal candidate, so throughput is N/A rather than zero.",
            ha="left", va="center", fontsize=6.5, color=SECONDARY)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.98, bottom=0.04)
    save_figure(fig, "rq2_ablation_summary_table")

    latex = r"""\begin{table*}[t]
\centering
\caption{RQ2 ablation and optimization summary. Throughput is reported in TFLOP/s.}
\label{tab:rq2-ablation-summary}
\small
\begin{tabular}{llrrrrr}
\toprule
Component & Configuration & Accepted / terminal & Before feedback & Selected throughput & Feedback gain & Relative to Full \\
\midrule
Construction & Full$^{\dagger}$ & 3/3 & -- & \textbf{10.577} & -- & 100.0\% \\
& w/o State-in-Prompt & 1/3 & 5.642 & 5.986 & +6.09\% & 56.6\% \\
& Intent-only Control$^{\ddagger}$ & 0/0 & -- & N/A & -- & -- \\
& w/o Local Checks & 3/3 & 5.903 & 7.470 & +26.55\% & 70.6\% \\
& w/o Stage Ordering & 2/3 & 6.045 & 6.912 & +14.33\% & 65.3\% \\
\midrule
Parameter search & w/o Parameter Search & -- & -- & 8.897 & -- & 84.1\% \\
& Full$^{\dagger}$ & -- & -- & \textbf{10.577} & -- & 100.0\% \\
\bottomrule
\end{tabular}
\vspace{1mm}
\begin{minipage}{0.98\linewidth}
\footnotesize $^{\dagger}$Archived historical Full result. $^{\ddagger}$Intent-only produced no terminal candidate; N/A is not interpreted as zero throughput. Feedback columns apply only to feedback-enabled ablations.
\end{minipage}
\end{table*}
"""
    (OUTPUT_DIR / "rq2_ablation_summary_table.tex").write_text(latex, encoding="utf-8")


def plot_combined_ablation_figure() -> None:
    """Place all RQ2 evidence in one coherent four-panel figure."""
    accent = "#315F8C"
    accent_light = "#82A9C7"
    neutral = "#9AA3AB"
    neutral_light = "#D9DEE3"
    labels = ["Full$^{\u2020}$", "w/o\nState", "Intent-\nonly", "w/o\nChecks", "w/o\nOrder"]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(7.25, 5.35))
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.95, bottom=0.14,
                          wspace=0.31, hspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Construction success: accepted versus rejected terminal candidates.
    accepted = np.array([3, 1, 0, 3, 2], dtype=float)
    totals = np.array([3, 3, 0, 3, 3], dtype=float)
    rejected = totals - accepted
    valid = totals > 0
    width = 0.62
    ax_a.bar(x[valid], accepted[valid], width=width, color=accent,
             edgecolor="white", linewidth=0.7, label="Accepted", zorder=3)
    ax_a.bar(x[valid], rejected[valid], width=width, bottom=accepted[valid],
             color=neutral_light, edgecolor="white", linewidth=0.7,
             label="Rejected", zorder=3)
    ratios = ["3/3", "1/3", "0/0", "3/3", "2/3"]
    for i in np.where(valid)[0]:
        ax_a.text(i, totals[i] + 0.08, ratios[i], ha="center", va="bottom",
                  fontsize=7.1, weight="semibold")
    ax_a.text(2, 1.52, "No terminal\ncandidate", ha="center", va="center",
              fontsize=6.7, color=SECONDARY, linespacing=0.95)
    ax_a.text(2, 0.08, "0/0", ha="center", va="bottom", fontsize=6.8,
              color=SECONDARY)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, 4.0)
    ax_a.set_yticks([0, 1, 2, 3])
    ax_a.set_ylabel("Terminal candidates")
    ax_a.set_title("(a) Candidate acceptance", loc="left", pad=6)
    style_axis(ax_a, "y")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
                columnspacing=1.1, handletextpad=0.4)

    # (b) Selected output quality for the same construction ablations.
    throughput = np.array([10.577, 5.986, np.nan, 7.470, 6.912])
    full_value = throughput[0]
    for i, value in enumerate(throughput):
        if np.isnan(value):
            continue
        if i == 0:
            ax_b.bar(i, value, width=width, facecolor="white", edgecolor=accent,
                     linewidth=1.15, hatch="////", zorder=3)
        else:
            ax_b.bar(i, value, width=width, color=accent_light,
                     edgecolor="white", linewidth=0.7, zorder=3)
        ax_b.text(i, value + 0.18, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=7.2, weight="semibold")
        if i > 0:
            ax_b.text(i, value - 0.36, f"{value / full_value * 100:.0f}%",
                      ha="center", va="top", fontsize=6.3, color="white",
                      weight="semibold", zorder=5)
    ax_b.text(2, 0.55, "N/A", ha="center", va="bottom", fontsize=7.0,
              color=SECONDARY)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, 12)
    ax_b.set_yticks([0, 3, 6, 9, 12])
    ax_b.set_ylabel("Throughput (TFLOP/s)")
    ax_b.set_title("(b) Selected-output throughput", loc="left", pad=6)
    style_axis(ax_b, "y")

    # (c) Feedback improvement on each viable ablation.
    feedback_names = ["w/o State", "w/o Checks", "w/o Order"]
    before = np.array([5.642, 5.903, 6.045])
    after = np.array([5.986, 7.470, 6.912])
    gains = ["+6.09%", "+26.55%", "+14.33%"]
    y = np.arange(len(feedback_names))
    for i in range(len(y)):
        if i % 2 == 0:
            ax_c.axhspan(i - 0.38, i + 0.38, color="#F5F6F7", zorder=0)
        ax_c.plot([before[i], after[i]], [i, i], color=neutral,
                  linewidth=1.45, zorder=2)
    ax_c.scatter(before, y, s=39, facecolor="white", edgecolor=neutral,
                 linewidth=1.2, zorder=4)
    ax_c.scatter(after, y, s=41, facecolor=accent, edgecolor="white",
                 linewidth=0.55, zorder=5)
    for i, (start, end, gain) in enumerate(zip(before, after, gains)):
        ax_c.text(start - 0.035, i - 0.16, f"{start:.3f}", ha="right",
                  va="bottom", fontsize=6.7, color=SECONDARY)
        ax_c.text(end + 0.035, i - 0.16, f"{end:.3f}", ha="left",
                  va="bottom", fontsize=6.7, color=accent, weight="semibold")
        ax_c.text(7.92, i, gain, ha="right", va="center", fontsize=7.0,
                  color=accent, weight="semibold")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(feedback_names)
    ax_c.invert_yaxis()
    ax_c.set_ylim(2.55, -0.62)
    ax_c.set_xlim(5.15, 8.0)
    ax_c.set_xticks([5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    ax_c.set_xlabel("Throughput (TFLOP/s)")
    ax_c.set_title("(c) Feedback optimization", loc="left", pad=6)
    style_axis(ax_c, "x")
    ax_c.scatter([5.37], [-0.42], s=28, facecolor="white", edgecolor=neutral,
                 linewidth=1.0, zorder=5)
    ax_c.text(5.46, -0.42, "Before", va="center", fontsize=6.7)
    ax_c.scatter([6.20], [-0.42], s=30, facecolor=accent, edgecolor="white",
                 linewidth=0.5, zorder=5)
    ax_c.text(6.29, -0.42, "After", va="center", fontsize=6.7)

    # (d) Parameter-search contribution, with the external library reference.
    search_labels = ["w/o Parameter\nSearch", "Full$^{\u2020}$"]
    search_values = np.array([8.897, 10.577])
    cublas = 9.062
    bars = ax_d.bar(np.arange(2), search_values, width=0.58,
                    color=[neutral_light, accent], edgecolor="white",
                    linewidth=0.75, zorder=3)
    for i, (bar, value) in enumerate(zip(bars, search_values)):
        ax_d.text(bar.get_x() + bar.get_width() / 2, value + 0.20,
                  f"{value:.3f}", ha="center", va="bottom", fontsize=7.4,
                  weight="semibold")
        ax_d.text(bar.get_x() + bar.get_width() / 2, value * 0.83,
                  f"{value / cublas:.3f}\u00d7 cuBLAS", ha="center", va="center",
                  fontsize=6.5, color=TEXT if i == 0 else "white")
    ax_d.axhline(cublas, color="#666D74", linewidth=0.9,
                 linestyle=(0, (3, 2)), zorder=2)
    ax_d.text(0.5, cublas + 0.10, "cuBLAS 9.062", ha="center", va="bottom",
              fontsize=6.5, color=SECONDARY,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax_d.set_xticks(np.arange(2))
    ax_d.set_xticklabels(search_labels)
    ax_d.set_ylim(0, 12)
    ax_d.set_yticks([0, 3, 6, 9, 12])
    ax_d.set_ylabel("Throughput (TFLOP/s)")
    ax_d.set_title("(d) Parameter-search ablation", loc="left", pad=6)
    style_axis(ax_d, "y")

    fig.text(0.5, 0.035,
             "$^{\u2020}$Archived historical Full result. Intent-only produced no terminal candidate; N/A is not zero throughput.",
             ha="center", va="center", fontsize=6.4, color=SECONDARY)
    save_figure(fig, "rq2_combined_ablation_analysis")


def plot_compact_ablation_chart() -> None:
    """Combine construction, feedback, and parameter search in one chart."""
    accent = "#315F8C"
    before_color = "#C9CFD5"
    rows = load_complete_rows()
    variants = [
        "full",
        "minus_state_in_prompt",
        "intent_only_control",
        "minus_local_checks",
        "minus_stage_ordering",
        "minus_parameter_search",
    ]
    x = np.array([0.0, 0.78, 1.56, 2.34, 3.12, 4.04])
    before = np.array([
        np.nan,
        tflops(rows[variants[1]], "phase1_best_gflops"),
        tflops(rows[variants[2]], "phase1_best_gflops"),
        tflops(rows[variants[3]], "phase1_best_gflops"),
        tflops(rows[variants[4]], "phase1_best_gflops"),
        np.nan,
    ])
    selected = np.array([tflops(rows[name], "final_gflops") for name in variants])
    gains = {
        i: f"+{100.0 * (selected[i] / before[i] - 1.0):.1f}%"
        for i in range(len(variants))
        if np.isfinite(before[i]) and np.isfinite(selected[i])
    }
    labels = [
        "Full\n3/3 accepted",
        "w/o State\n1/3 accepted",
        "Intent-only\n2/3 accepted",
        "w/o Checks\n3/3 accepted",
        "w/o Order\n2/3 accepted",
        "w/o Parameter\nSearch",
    ]

    fig, ax = plt.subplots(figsize=(7.25, 3.08))
    width = 0.29
    paired = np.isfinite(before)

    before_bars = ax.bar(x[paired] - width / 1.7, before[paired], width=width,
                         color=before_color, edgecolor="white", linewidth=0.7,
                         label="Before feedback", zorder=3)

    for i, value in enumerate(selected):
        if not np.isfinite(value):
            continue
        xpos = x[i] + width / 1.7 if paired[i] else x[i]
        if i == 0:
            ax.bar(xpos, value, width=width * 1.18, facecolor="white",
                   edgecolor=accent, linewidth=1.2, hatch="////", zorder=3)
        else:
            ax.bar(xpos, value, width=width, color=accent, edgecolor="white",
                   linewidth=0.7, zorder=3)
        if i == 5:
            ax.text(xpos, value - 0.22, f"{value:.3f}", ha="center", va="top",
                    fontsize=9.8, weight="semibold", color="white")
        else:
            ax.text(xpos, value + 0.17, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=9.5, weight="semibold", color=TEXT)
        if i in gains:
            ax.text(xpos, value - 0.24, gains[i], ha="center", va="top",
                    fontsize=8.5, color="white", weight="semibold", zorder=5)

    for bar, value in zip(before_bars, before[paired]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9.0, color=SECONDARY,
                weight="medium")

    separator_x = (x[4] + x[5]) / 2
    ax.axvline(separator_x, color="#B7BEC5", linewidth=0.75,
               linestyle=(0, (2, 2)), zorder=1)

    cublas = 9.062
    ax.axhline(cublas, color="#646B72", linewidth=0.95,
               linestyle=(0, (4, 2)), label="cuBLAS reference", zorder=2)
    ax.text(4.38, cublas + 0.10, "cuBLAS 9.062", ha="right", va="bottom",
            fontsize=9.2, color=SECONDARY,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})

    # Explicit proxy keeps the legend independent of missing bars.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=before_color, edgecolor="white", label="Pre-feedback kernel"),
        Patch(facecolor=accent, edgecolor="white", label="Final selected kernel"),
        Line2D([0], [0], color="#646B72", linewidth=0.95,
               linestyle=(0, (4, 2)), label="cuBLAS reference"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.015),
              ncol=3, columnspacing=1.35, handletextpad=0.45, fontsize=9.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.34, 4.42)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylabel("Throughput (TFLOP/s)", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9.0)
    ax.tick_params(axis="y", labelsize=10.0)
    style_axis(ax, "y")
    ax.text(1.56, 11.55, "Construction and feedback", ha="center", va="center",
            fontsize=8.6, color=SECONDARY, weight="semibold")
    ax.text(x[5], 11.55, "Parameter search", ha="center", va="center",
            fontsize=8.6, color=SECONDARY, weight="semibold")

    fig.subplots_adjust(left=0.105, right=0.99, top=0.80, bottom=0.27)
    save_figure(fig, "rq2_compact_ablation_analysis")


def plot_optimization_analysis() -> None:
    names = ["w/o State-in-Prompt", "w/o Local Checks", "w/o Stage Ordering"]
    before = np.array([5.642, 5.903, 6.045])
    after = np.array([5.986, 7.470, 6.912])
    gains = ["+6.09%", "+26.55%", "+14.33%"]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.25, 2.72),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.38},
    )

    y = np.arange(len(names))
    for i in range(len(names)):
        if i % 2 == 0:
            ax_a.axhspan(i - 0.38, i + 0.38, color="#F6F8FA", zorder=0)
        ax_a.plot([before[i], after[i]], [i, i], color="#9DA8B2", linewidth=1.6, zorder=2)

    ax_a.scatter(before, y, s=46, facecolor="white", edgecolor=WARM_GRAY,
                 linewidth=1.35, zorder=4)
    ax_a.scatter(after, y, s=48, facecolor=TEAL, edgecolor="white",
                 linewidth=0.65, zorder=5)

    for i, (start, end, gain) in enumerate(zip(before, after, gains)):
        ax_a.text(start - 0.035, i - 0.16, f"{start:.3f}", ha="right", va="bottom",
                  fontsize=7.2, color=SECONDARY)
        ax_a.text(end + 0.035, i - 0.16, f"{end:.3f}", ha="left", va="bottom",
                  fontsize=7.2, color=TEAL, weight="semibold")
        ax_a.text(7.88, i, gain, ha="right", va="center", fontsize=8.1,
                  color=NAVY, weight="semibold")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(names)
    ax_a.invert_yaxis()
    ax_a.set_ylim(2.55, -0.62)
    ax_a.set_xlim(5.15, 8.0)
    ax_a.set_xticks([5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    ax_a.set_xlabel("Throughput (TFLOP/s)")
    ax_a.set_title("(a) Feedback optimization", loc="left", pad=7)
    style_axis(ax_a, "x")
    ax_a.scatter([5.36], [-0.42], s=34, facecolor="white", edgecolor=WARM_GRAY,
                 linewidth=1.15, zorder=5)
    ax_a.text(5.45, -0.42, "Before feedback", va="center", fontsize=7.1)
    ax_a.scatter([6.38], [-0.42], s=36, facecolor=TEAL, edgecolor="white",
                 linewidth=0.55, zorder=5)
    ax_a.text(6.47, -0.42, "After feedback", va="center", fontsize=7.1)

    configs = ["w/o Parameter\nSearch", "Full$^{\u2020}$"]
    values = np.array([8.897, 10.577])
    cublas = 9.062
    relative = values / cublas
    x = np.arange(len(configs))
    bars = ax_b.bar(x, values, width=0.58, color=[SKY, NAVY], edgecolor="white",
                    linewidth=0.8, zorder=3)
    for bar, value, ratio in zip(bars, values, relative):
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=8.2, weight="semibold")
        ax_b.text(bar.get_x() + bar.get_width() / 2, value * 0.86,
                  f"{ratio:.3f}\u00d7 cuBLAS", ha="center", va="top", fontsize=7.0,
                  color="white" if value > 9.5 else TEXT)
    ax_b.axhline(cublas, color="#6E7781", linewidth=1.0,
                 linestyle=(0, (3, 2)), zorder=2)
    ax_b.text(0.72, cublas + 0.10, "cuBLAS 9.062", ha="center", va="bottom",
              fontsize=6.8, color=SECONDARY,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(configs)
    ax_b.set_ylim(0, 12)
    ax_b.set_yticks([0, 3, 6, 9, 12])
    ax_b.set_ylabel("Throughput (TFLOP/s)")
    ax_b.set_title("(b) Parameter-search ablation", loc="left", pad=7)
    style_axis(ax_b, "y")
    ax_b.text(0.5, -0.23,
              "$^{\u2020}$Archived historical record; not a paired optimization trajectory.",
              transform=ax_b.transAxes, ha="center", va="top", fontsize=6.7,
              color=SECONDARY)

    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.24)
    save_figure(fig, "rq2_optimization_analysis")


def plot_construction_ablation() -> None:
    labels = ["Full$^{\u2020}$", "w/o State-\nin-Prompt", "Intent-only\nControl",
              "w/o Local\nChecks", "w/o Stage\nOrdering"]
    accepted = np.array([3, 1, 0, 3, 2], dtype=float)
    totals = np.array([3, 3, 0, 3, 3], dtype=float)
    rejected = totals - accepted
    throughput = np.array([10.577, 5.986, np.nan, 7.470, 6.912])
    x = np.arange(len(labels))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.25, 2.85),
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.29},
    )
    width = 0.62
    valid = totals > 0

    ax_a.bar(x[valid], accepted[valid], width=width, color=GREEN, edgecolor="white",
             linewidth=0.75, label="Accepted", zorder=3)
    ax_a.bar(x[valid], rejected[valid], width=width, bottom=accepted[valid],
             color=LIGHT_GRAY, edgecolor="white", linewidth=0.75,
             label="Rejected", zorder=3)
    ratio_text = ["3/3\n100%", "1/3\n33%", "0/0", "3/3\n100%", "2/3\n67%"]
    for i in np.where(valid)[0]:
        ax_a.text(i, totals[i] + 0.08, ratio_text[i], ha="center", va="bottom",
                  fontsize=7.1, linespacing=0.95)
    ax_a.text(2, 1.55, "No terminal\ncandidate", ha="center", va="center",
              fontsize=7.2, color=SECONDARY)
    ax_a.text(2, 0.08, "0/0", ha="center", va="bottom", fontsize=7.1,
              color=SECONDARY)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, 4.05)
    ax_a.set_yticks([0, 1, 2, 3])
    ax_a.set_ylabel("Terminal candidates")
    ax_a.set_title("(a) Candidate acceptance", loc="left", pad=7)
    style_axis(ax_a, "y")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.50, 0.99), ncol=2,
                columnspacing=1.1, handletextpad=0.45)

    full_value = throughput[0]
    throughput_bars = []
    for i, value in enumerate(throughput):
        if np.isnan(value):
            throughput_bars.append(None)
            continue
        if i == 0:
            bar = ax_b.bar(i, value, width=width, facecolor="white", edgecolor=NAVY,
                           linewidth=1.25, hatch="////", zorder=3)[0]
        else:
            bar = ax_b.bar(i, value, width=width, color=BLUE, edgecolor="white",
                           linewidth=0.75, zorder=3)[0]
        throughput_bars.append(bar)

    for i, (value, bar) in enumerate(zip(throughput, throughput_bars)):
        if bar is None:
            continue
        ratio = value / full_value * 100
        value_offset = 0.20 if i == 0 else 0.62
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + value_offset, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=7.7, weight="semibold", zorder=6)
        if i > 0:
            ax_b.text(bar.get_x() + bar.get_width() / 2, value + 0.12,
                      f"{ratio:.0f}% of Full", ha="center", va="bottom", fontsize=6.3,
                      color=SECONDARY, weight="medium", zorder=6, clip_on=False)
    ax_b.text(2, 0.55, "N/A", ha="center", va="bottom", fontsize=7.5,
              color=SECONDARY)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, 12)
    ax_b.set_yticks([0, 3, 6, 9, 12])
    ax_b.set_ylabel("Throughput (TFLOP/s)")
    ax_b.set_title("(b) Selected-output throughput", loc="left", pad=7)
    style_axis(ax_b, "y")

    fig.text(0.5, 0.015,
             "$^{\u2020}$Archived historical reference. Intent-only produced no terminal candidate.",
             ha="center", va="bottom", fontsize=6.7, color=SECONDARY)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.25)
    save_figure(fig, "rq2_construction_ablation")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_compact_ablation_chart()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
