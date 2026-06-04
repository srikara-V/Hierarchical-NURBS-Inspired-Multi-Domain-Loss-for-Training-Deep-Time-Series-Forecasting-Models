#!/usr/bin/env python3
"""
Generate publication-style multi-panel figures for the HNMD paper.

Loads ./results/ (pred.npy, true.npy, metrics.npy) and runs MSSD NURBS decomposition
via PyTorch. Exits with an error if any required run or dependency is missing — no
synthetic or placeholder series are ever plotted.

Panel (a) of fig1 is a method schematic only (no numeric data).

Usage (from repo root):
    env/bin/python paper/generate_figures.py

Outputs PDF + PNG under paper/figures/.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.figure_data import (  # noqa: E402
    FIG4_DATASETS,
    REF_CHANNEL,
    REF_DATASET,
    REF_PRED_LEN,
    REF_SAMPLE,
    decomposed_series_for_plot,
    extract_window,
    extract_window_mv,
    get_hyperparams,
    level_gradient_weights,
    load_run_arrays,
    mlp_dataset_averages,
    nurbs_decompose_1d,
    validate_figure_prerequisites,
)

OUT_DIR = Path(__file__).resolve().parent / "figures"

LOSS_COLORS = {"MSE": "#4C72B0", "Tilde-Q": "#DD8452", "HNMD": "#C44E52"}
LOSS_ORDER = ["MSE", "Tilde-Q", "HNMD"]
LEVEL_CMAP = plt.cm.viridis


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def panel_label(ax, label: str, x=-0.12, y=1.06):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def subplot_title(ax, panel: str, title: str, pad: float = 10):
    """Panel letter + title on one line (avoids overlap with separate label)."""
    ax.set_title(f"({panel}) {title}", loc="left", fontsize=10, fontweight="bold", pad=pad)


def save_fig(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path}")


def _method_flowchart_html() -> str:
    """HTML + inline SVG arrows (rendered via Playwright when available)."""
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #fff; font-family: "Times New Roman", "DejaVu Serif", serif; }
  .wrap { padding: 10px 14px 6px; }
  .flow { display: flex; align-items: center; justify-content: center; gap: 0; }
  .node {
    border: 1px solid #2d2d2d; border-radius: 10px; padding: 14px 12px;
    text-align: center; background: #E8F4FC; line-height: 1.35; font-size: 14px;
  }
  .proc {
    background: #F5FAFD; min-width: 340px; max-width: 380px; padding: 12px 16px;
    text-align: left; line-height: 1.5; font-size: 13px;
  }
  .proc .hdr { font-weight: bold; margin-bottom: 6px; }
  .arrow-slot { width: 44px; height: 28px; flex: 0 0 44px; }
  .note { text-align: center; font-size: 11px; font-style: italic; color: #333; margin-top: 6px; }
</style></head><body><div class="wrap">
<div class="flow">
  <div class="node">Input series<br/><i>y</i>(<i>t</i>)</div>
  <svg class="arrow-slot" viewBox="0 0 44 28" aria-hidden="true">
    <defs><marker id="m1" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="#2d2d2d"/></marker></defs>
    <line x1="4" y1="14" x2="38" y2="14" stroke="#2d2d2d" stroke-width="1.6" marker-end="url(#m1)"/>
  </svg>
  <div class="node proc">
    <div class="hdr">For each level <i>l</i> = 1, …, <i>M</i>:</div>
    <div><i>W</i><sub><i>l</i></sub> ← softmax(var(<i>R</i><sub><i>l</i>−1</sub>))</div>
    <div><i>D</i><sub><i>l</i></sub> ← NURBS(<i>R</i><sub><i>l</i>−1</sub>, <i>W</i><sub><i>l</i></sub>)</div>
    <div><i>R</i><sub><i>l</i></sub> ← <i>R</i><sub><i>l</i>−1</sub> − <i>D</i><sub><i>l</i></sub></div>
    <div style="margin-top:6px;font-size:12px;">initialize <i>R</i><sub>0</sub> = <i>y</i>; repeat until <i>l</i> = <i>M</i></div>
  </div>
  <svg class="arrow-slot" viewBox="0 0 44 28" aria-hidden="true">
    <defs><marker id="m2" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="#2d2d2d"/></marker></defs>
    <line x1="4" y1="14" x2="38" y2="14" stroke="#2d2d2d" stroke-width="1.6" marker-end="url(#m2)"/>
  </svg>
  <div class="node">Hierarchical levels<br/><i>D</i><sub>1</sub>, …, <i>D</i><sub><i>M</i></sub></div>
</div>
<p class="note">Higher <i>l</i> captures finer residual structure</p>
</div></body></html>"""


def _render_flowchart_html_png() -> np.ndarray | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    html = _method_flowchart_html()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 920, "height": 200}, device_scale_factor=2)
            page.set_content(html, wait_until="load")
            png_bytes = page.screenshot(type="png")
            browser.close()
        return plt.imread(io.BytesIO(png_bytes))
    except Exception:
        return None


def _flow_arrow_between(ax, x_start: float, x_end: float, y: float = 0.52):
    """Single arrow in gap between boxes (axes fraction); no overlap with box fills."""
    ax.add_patch(
        mpatches.FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.2,
            color="#2d2d2d",
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=5,
        )
    )


def _panel_method_flowchart_matplotlib(ax):
    """Matplotlib fallback when HTML renderer is unavailable."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Box edges in axes fraction (left, bottom, width, height)
    boxes = [
        (0.03, 0.20, 0.16, 0.60, "#E8F4FC", r"Input series" + "\n" + r"$y(t)$", "center", 8.5),
        (0.23, 0.12, 0.50, 0.76, "#F5FAFD", None, "left", 8),
        (0.77, 0.20, 0.20, 0.60, "#E8F4FC", r"Hierarchical levels" + "\n" + r"$D_1, \ldots, D_M$", "center", 8.5),
    ]
    for x0, y0, w, h, fc, text, ha, fs in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#2d2d2d",
                facecolor=fc,
                transform=ax.transAxes,
                clip_on=False,
                zorder=2,
            )
        )
        if text:
            ax.text(
                x0 + w / 2,
                y0 + h / 2,
                text,
                ha=ha,
                va="center",
                fontsize=fs,
                transform=ax.transAxes,
                zorder=3,
            )

    steps = [
        (0.27, 0.78, r"For each level $l = 1, \ldots, M$:", "bold", 8.5),
        (0.27, 0.66, r"$W_l \leftarrow \mathrm{softmax}(\mathrm{var}(R_{l-1}))$", "normal", 8),
        (0.27, 0.54, r"$D_l \leftarrow \mathrm{NURBS}(R_{l-1},\, W_l)$", "normal", 8),
        (0.27, 0.42, r"$R_l \leftarrow R_{l-1} - D_l$", "normal", 8),
        (0.27, 0.28, r"initialize $R_0 = y$; repeat until $l = M$", "normal", 7.5),
    ]
    for x, y, s, weight, fs in steps:
        ax.text(
            x, y, s, ha="left", va="center", fontsize=fs, fontweight=weight, transform=ax.transAxes, zorder=3
        )

    cy = 0.50
    _flow_arrow_between(ax, 0.19, 0.23, cy)  # input → process
    _flow_arrow_between(ax, 0.73, 0.77, cy)  # process → output

    ax.text(
        0.50,
        0.03,
        r"Higher $l$ captures finer residual structure",
        ha="center",
        fontsize=7.5,
        style="italic",
        transform=ax.transAxes,
    )


def _panel_method_flowchart(ax):
    img = _render_flowchart_html_png()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if img is not None:
        ax.imshow(img, aspect="auto", extent=(0.02, 0.98, 0.02, 0.98), transform=ax.transAxes, zorder=1)
        return
    _panel_method_flowchart_matplotlib(ax)


def _reference_true_window() -> tuple[np.ndarray, np.ndarray, str]:
    pred, true, _, run_dir = load_run_arrays(loss="HNMD")
    y_mv, _ = extract_window_mv(pred, true, REF_SAMPLE)
    y = y_mv[:, REF_CHANNEL]
    return y, y_mv, run_dir


def fig_method_overview():
    # Top row: wide flowchart; bottom row: plots (more room, no crowding)
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        hspace=0.58,
        wspace=0.40,
        top=0.82,
        bottom=0.12,
        left=0.11,
        right=0.97,
    )

    ax0 = fig.add_subplot(gs[0, :])
    subplot_title(ax0, "a", "Weighted decomposition", pad=14)
    _panel_method_flowchart(ax0)

    ax1 = fig.add_subplot(gs[1, 0])
    subtitle_b = f"Hierarchical components ({REF_DATASET}, $T$={REF_PRED_LEN})"
    y, y_mv, _ = _reference_true_window()
    hp = get_hyperparams()
    levels, resid, _, _ = nurbs_decompose_1d(y, hp, y_multivariate=y_mv)
    n_show = min(3, len(levels))
    comps = levels[:n_show] + [resid]
    y_sum = y
    t = np.arange(len(y))
    span = max(max(np.ptp(c) for c in comps + [y_sum]), 0.1)
    step = span * 1.15
    names = [f"$D_{i+1}$" for i in range(n_show)] + ["Residual", "Sum"]
    colors = [LEVEL_CMAP(0.15 + 0.25 * i) for i in range(n_show)] + ["#888888", "#222222"]
    series = [(c, i * step, nm, col) for i, (c, nm, col) in enumerate(zip(comps, names[:-1], colors[:-1]))]
    series.append((y_sum, len(comps) * step, names[-1], colors[-1]))
    subplot_title(ax1, "b", subtitle_b)
    for c, off, name, col in series:
        lw = 1.5 if name == "Sum" else 1.2
        ax1.plot(t, c + off, color=col, lw=lw)
    ax1.set_xlim(0, len(t) - 1)
    ax1.set_xlabel("Time step")
    ax1.set_yticks([])
    ymin, ymax = ax1.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0
    for c, off, name, col in series:
        y_mid = float(np.median(c + off))
        y_norm = (y_mid - ymin) / yspan
        ax1.text(
            -0.14,
            y_norm,
            name,
            transform=ax1.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color=col,
            clip_on=False,
        )
    ax1.spines["left"].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 1])
    hp = get_hyperparams()
    domains = ["Time", "Derivative", "Frequency"]
    weights = [hp.alpha, hp.beta, hp.gamma]
    w_max = max(weights) * 1.15 if max(weights) > 0 else 1.0
    subplot_title(ax2, "c", f"Loss weights ({REF_DATASET}, $T$={REF_PRED_LEN})")
    xpos = np.arange(len(domains))
    bars = ax2.bar(
        xpos,
        weights,
        color=["#4C72B0", "#55A868", "#C44E52"],
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(domains, fontsize=8.5)
    ax2.set_ylabel(r"Weight ($\alpha$, $\beta$, $\gamma$)", fontsize=8.5)
    ax2.set_ylim(0, w_max)
    for b, w in zip(bars, weights):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.04,
            f"{w:g}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax2.set_xlabel("domain: value | derivative | FFT", fontsize=7.5, labelpad=4)

    fig.suptitle(
        "HNMD: hierarchical decomposition and multi-domain training loss",
        fontsize=10.5,
        y=0.96,
    )
    save_fig(fig, "fig1_method_overview")
    plt.close(fig)


def fig_decomposition():
    title_note = f"{REF_DATASET} test ground truth, $T$={REF_PRED_LEN}"
    pred, true, _, _ = load_run_arrays(loss="HNMD")
    y_mv, _ = extract_window_mv(pred, true, REF_SAMPLE)
    y = y_mv[:, REF_CHANNEL]
    hp = get_hyperparams()
    levels, resid, weights, _ = nurbs_decompose_1d(y, hp, y_multivariate=y_mv)
    n_show = min(3, len(levels))
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.1, 1], hspace=0.42, wspace=0.35)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(y, color="black", lw=1.5, label="Series $y(t)$")
    recon = sum(levels)
    ax_top.plot(recon, color="#C44E52", lw=1.2, ls="--", label="Sum of levels")
    ax_top.legend(loc="upper right", frameon=True, fontsize=8)
    ax_top.set_ylabel("Amplitude")
    ax_top.set_title(f"NURBS decomposition on {title_note}")
    panel_label(ax_top, "a", x=-0.02)

    labels = "bcde"
    for i in range(n_show):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(levels[i], color=LEVEL_CMAP(i / max(len(levels) - 1, 1)), lw=1.3)
        ax.set_title(f"Level {i + 1}", fontsize=9)
        ax.set_xlabel("Time step")
        panel_label(ax, labels[i])

    ax_r = fig.add_subplot(gs[1, 3])
    ax_r2 = ax_r.twinx()
    ax_r.plot(resid, color="#888888", lw=1.0)
    ax_r2.fill_between(np.arange(len(weights)), 0, weights, color="#DD8452", alpha=0.35)
    ax_r2.plot(weights, color="#DD8452", lw=1.0)
    ax_r.set_xlabel("Time step")
    ax_r.set_ylabel("Residual")
    ax_r2.set_ylabel("Weight", color="#DD8452", fontsize=8)
    ax_r.set_title("Residual & $W_l$", fontsize=9)
    panel_label(ax_r, "e")

    save_fig(fig, "fig2_decomposition")
    plt.close(fig)


def fig_multidomain():
    subtitle = f"{REF_DATASET} HNMD run, sample {REF_SAMPLE}"
    pred, true, _, _ = load_run_arrays(loss="HNMD")
    y_true_mv, y_pred_mv = extract_window_mv(pred, true, REF_SAMPLE)
    y_true, y_pred = decomposed_series_for_plot(y_true_mv, y_pred_mv)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    t = np.arange(len(y_true))

    axes[0].plot(t, y_true, "k-", lw=1.4, label="Target (decomposed)")
    axes[0].plot(t, y_pred, color="#C44E52", lw=1.2, ls="--", label="Prediction")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Time domain")
    axes[0].legend(fontsize=7, loc="upper right")
    panel_label(axes[0], "a")

    d_true = np.diff(y_true)
    d_pred = np.diff(y_pred)
    axes[1].plot(t[1:], d_true, "k-", lw=1.4, label="Target")
    axes[1].plot(t[1:], d_pred, color="#C44E52", lw=1.2, ls="--", label="Prediction")
    axes[1].set_title("Derivative domain")
    axes[1].set_ylabel("$\\partial_t$")
    panel_label(axes[1], "b")

    f_true = np.abs(np.fft.rfft(y_true))
    f_pred = np.abs(np.fft.rfft(y_pred))
    freqs = np.fft.rfftfreq(len(y_true))
    axes[2].plot(freqs, f_true, "k-", lw=1.4, label="Target")
    axes[2].plot(freqs, f_pred, color="#C44E52", lw=1.2, ls="--", label="Prediction")
    axes[2].set_xlim(0, 0.25)
    axes[2].set_title("Frequency domain")
    axes[2].set_xlabel("Normalized frequency")
    axes[2].set_ylabel("|FFT|")
    panel_label(axes[2], "c")

    axes[0].set_xlabel("Time step")
    axes[1].set_xlabel("Time step")

    fig.suptitle(
        f"Decomposed target vs prediction ({subtitle})",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()
    save_fig(fig, "fig3_multidomain")
    plt.close(fig)


def fig_mlp_results():
    mlp_avg = mlp_dataset_averages()
    datasets = FIG4_DATASETS
    x = np.arange(len(datasets))
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})

    for j, loss in enumerate(LOSS_ORDER):
        vals = [mlp_avg[d][loss] for d in datasets]
        axes[0].bar(
            x + (j - 1) * width,
            vals,
            width,
            label=loss,
            color=LOSS_COLORS[loss],
            edgecolor="white",
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=25, ha="right")
    axes[0].set_ylabel("Test MSE (dataset average)")
    axes[0].set_title("MLP: loss comparison across benchmarks")
    axes[0].legend(frameon=True, ncol=3, loc="upper right", fontsize=7)
    panel_label(axes[0], "a")

    # Relative improvement HNMD vs best baseline per dataset
    rel = []
    names = []
    for d in datasets:
        mse = mlp_avg[d]["MSE"]
        tq = mlp_avg[d]["Tilde-Q"]
        hn = mlp_avg[d]["HNMD"]
        best = min(mse, tq)
        rel.append(100 * (best - hn) / best)
        names.append(d)

    colors = ["#55A868" if r > 0 else "#C44E52" for r in rel]
    axes[1].barh(names, rel, color=colors, edgecolor="white")
    axes[1].axvline(0, color="#333", lw=0.8)
    axes[1].set_xlabel("% MSE reduction vs best of MSE / Tilde-Q")
    axes[1].set_title("HNMD gain (MLP averages)")
    panel_label(axes[1], "b")

    fig.tight_layout()
    save_fig(fig, "fig4_mlp_results")
    plt.close(fig)


def fig_forecast_comparison():
    dataset, pred_len = REF_DATASET, REF_PRED_LEN
    pred_m, true_m, met_m, _ = load_run_arrays(loss="MSE")
    pred_h, true_h, met_h, _ = load_run_arrays(loss="HNMD")
    tr, pm = extract_window(pred_m, true_m, REF_SAMPLE, REF_CHANNEL)
    _, ph = extract_window(pred_h, true_h, REF_SAMPLE, REF_CHANNEL)
    mse_test, hn_test = float(met_m[1]), float(met_h[1])
    t = np.arange(len(tr))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True)

    axes[0, 0].plot(t, tr, "k-", lw=1.5, label="Ground truth")
    axes[0, 0].plot(t, pm, color=LOSS_COLORS["MSE"], lw=1.2, ls="--", label="MSE-trained")
    axes[0, 0].set_title(f"{dataset}, horizon {pred_len}")
    axes[0, 0].legend(fontsize=7)
    panel_label(axes[0, 0], "a")

    axes[0, 1].plot(t, tr, "k-", lw=1.5, label="Ground truth")
    axes[0, 1].plot(t, ph, color=LOSS_COLORS["HNMD"], lw=1.2, ls="--", label="HNMD-trained")
    axes[0, 1].legend(fontsize=7)
    panel_label(axes[0, 1], "b")

    err_mse = np.abs(pm - tr)
    err_hn = np.abs(ph - tr)
    axes[1, 0].plot(t, err_mse, color=LOSS_COLORS["MSE"], lw=1.2)
    axes[1, 0].set_ylabel("|Error|")
    axes[1, 0].set_title("MSE model error")
    panel_label(axes[1, 0], "c")

    axes[1, 1].plot(t, err_hn, color=LOSS_COLORS["HNMD"], lw=1.2)
    axes[1, 1].set_title("HNMD model error")
    axes[1, 1].set_xlabel("Forecast step")
    panel_label(axes[1, 1], "d")

    for ax in axes[1]:
        ax.set_xlabel("Forecast step")

    cap = f"Test-set MSE (full split): MSE={mse_test:.3f}, HNMD={hn_test:.3f}"
    fig.suptitle(f"Forecast window on {dataset} ($T={pred_len}$, sample {REF_SAMPLE}; {cap})", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig5_forecast_comparison")
    plt.close(fig)


def fig_gradient_weights():
    """Level heatmap and gradient weights from MSSD decomposition on a test window."""
    pred, true, _, _ = load_run_arrays(loss="HNMD")
    y_true_mv, y_pred_mv = extract_window_mv(pred, true, REF_SAMPLE)
    importance, levels = level_gradient_weights(y_true_mv, y_pred_mv)
    n_base = len(levels)
    importance_base = importance[:n_base]
    importance_base = importance_base / (importance_base.sum() + 1e-8)
    stack = np.stack(levels, axis=0)
    title_b = f"Gradient weights ({REF_DATASET}, sample {REF_SAMPLE})"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    vmax = np.max(np.abs(stack)) + 1e-8
    im = axes[0].imshow(stack, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(range(len(levels)))
    axes[0].set_yticklabels([f"L{i+1}" for i in range(len(levels))])
    axes[0].set_xlabel("Time step")
    axes[0].set_title("Decomposition levels (heatmap)")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    panel_label(axes[0], "a")

    axes[1].bar(
        np.arange(1, len(importance_base) + 1),
        importance_base,
        color=LEVEL_CMAP(np.linspace(0.2, 0.9, len(importance_base))),
    )
    axes[1].set_xlabel("Hierarchy level")
    axes[1].set_ylabel("Relative weight")
    axes[1].set_title(title_b)
    panel_label(axes[1], "b")

    fig.tight_layout()
    save_fig(fig, "fig6_level_importance")
    plt.close(fig)


def write_figures_tex():
    tex = r"""% Auto-generated figure includes for main.tex
% Run: python paper/generate_figures.py

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig1_method_overview.pdf}
    \caption{Overview of HNMD. \textbf{(a)} Variance-conditioned NURBS decomposition of the residual at each level.
    \textbf{(b)} Hierarchical components from NURBS decomposition of an ETTm1 test window ($T{=}96$).
    \textbf{(c)} Tuned $\alpha$, $\beta$, $\gamma$ for ETTm1 / MLP / $T{=}96$ (Table~\ref{tab:hyper}).}
    \label{fig:method}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig2_decomposition.pdf}
    \caption{NURBS decomposition of an ETTm1 test ground-truth window ($T{=}96$).
    \textbf{(a)} Original series and sum of fitted levels. \textbf{(b--d)} Individual levels. \textbf{(e)} Final residual and regional variance weights.}
    \label{fig:decomposition}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig3_multidomain.pdf}
    \caption{Decomposed target vs.\ HNMD prediction (ETTm1, $T{=}96$) in time, derivative, and frequency domains.}
    \label{fig:multidomain}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig4_mlp_results.pdf}
    \caption{MLP test MSE (mean over horizons) from \texttt{./results/}. \textbf{(a)} MSE, Tilde-Q, and HNMD. \textbf{(b)} Percent improvement of HNMD over the better of MSE/Tilde-Q where both exist.}
    \label{fig:mlp-results}
\end{figure}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig5_forecast_comparison.pdf}
    \caption{Forecast window from saved \texttt{pred.npy}/\texttt{true.npy} (ETTm1, $T{=}96$). \textbf{(a,b)} MSE- vs HNMD-trained MLP. \textbf{(c,d)} Absolute errors. Suptitle gives full test-split MSE.}
    \label{fig:forecasts}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/fig6_level_importance.pdf}
    \caption{NURBS levels (heatmap) and per-level gradient weights from MSSD on an ETTm1 test window ($T{=}96$).}
    \label{fig:level-importance}
\end{figure}
"""
    path = OUT_DIR / "figures.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"  wrote {path}")


def main():
    validate_figure_prerequisites()
    apply_style()
    print(f"Output directory: {OUT_DIR}")
    print(
        f"Data: ./results/ | ref={REF_DATASET} MLP T={REF_PRED_LEN} "
        f"sample={REF_SAMPLE} ch={REF_CHANNEL}"
    )
    fig_method_overview()
    fig_decomposition()
    fig_multidomain()
    fig_mlp_results()
    fig_gradient_weights()
    fig_forecast_comparison()
    write_figures_tex()
    print("Done.")


if __name__ == "__main__":
    main()
