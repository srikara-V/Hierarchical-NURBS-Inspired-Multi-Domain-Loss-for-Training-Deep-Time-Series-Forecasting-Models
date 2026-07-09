#!/usr/bin/env python3
"""Generate LaTeX tables for the HNMD preprint from experiment result folders.

Reads ./results, ./results_ablation/<variant>, ./results_default,
./results_seeds/s<seed> and writes paper/tables/*.tex plus a JSON summary of
headline numbers (paper/tables/summary.json). All table numbers are derived
from metrics.npy files — nothing is hand-typed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.hyperparams import get_mssd_hyperparams
from utils.results_parser import collect_results

OUT_DIR = ROOT / "paper" / "tables"
PRED_LENS = [96, 192, 336, 720]
LOSS_ORDER = ["mse", "tildeq", "mssd"]
LOSS_LABEL = {"mse": "MSE", "tildeq": "TILDE-Q", "mssd": "HNMD"}

DATASET_ORDER = [
    "ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Exchange", "Traffic", "Weather", "Solar",
]
DATASET_LABEL = {
    "ETTm1": "ETTm1", "ETTm2": "ETTm2", "ETTh1": "ETTh1", "ETTh2": "ETTh2",
    "ECL": "ECL", "Exchange": "Exchange", "Traffic": "Traffic",
    "Weather": "Weather", "Solar": "Solar-Energy",
}
ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

ABLATION_CELLS = [("ETTh1", 96), ("ETTh1", 336), ("ETTm1", 96), ("ETTm1", 336), ("ETTm2", 96), ("ETTm2", 336)]
ABLATION_VARIANTS = [
    ("noalpha", r"w/o time term ($\alpha{=}0$)"),
    ("nobeta", r"w/o derivative term ($\beta{=}0$)"),
    ("nogamma", r"w/o frequency term ($\gamma{=}0$)"),
    ("timeonly", r"time term only ($\beta{=}\gamma{=}0$)"),
    ("levels1", r"single level ($M{=}1$)"),
    ("levels3", r"shallow hierarchy ($M{=}3$)"),
    ("xp0", r"uniform level weights ($x{=}0$)"),
    ("nodecomp", r"w/o decomposition ($M{=}0$)"),
]

INTERNAL_KEY = {
    "ETTm1": "ETTm1", "ETTm2": "ETTm2", "ETTh1": "ETTh1", "ETTh2": "ETTh2",
    "ECL": "electricity", "Exchange": "exchange_rate", "Traffic": "traffic",
    "Weather": "weather", "Solar": "Solar",
}

# Published MSE-trained baseline averages (input length 96, mean over the four
# horizons) from Liu et al., iTransformer, ICLR 2024, Table 4.
PUBLISHED_BASELINES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "iTransformer": {
        "ETTm1": (0.407, 0.410), "ETTm2": (0.288, 0.332), "ETTh1": (0.454, 0.447),
        "ETTh2": (0.383, 0.407), "ECL": (0.178, 0.270), "Exchange": (0.360, 0.403),
        "Traffic": (0.428, 0.282), "Weather": (0.258, 0.278), "Solar": (0.233, 0.262),
    },
    "RLinear": {
        "ETTm1": (0.414, 0.407), "ETTm2": (0.286, 0.327), "ETTh1": (0.446, 0.434),
        "ETTh2": (0.374, 0.398), "ECL": (0.219, 0.298), "Exchange": (0.378, 0.417),
        "Traffic": (0.626, 0.378), "Weather": (0.272, 0.291), "Solar": (0.369, 0.356),
    },
    "PatchTST": {
        "ETTm1": (0.387, 0.400), "ETTm2": (0.281, 0.326), "ETTh1": (0.469, 0.454),
        "ETTh2": (0.387, 0.407), "ECL": (0.205, 0.290), "Exchange": (0.367, 0.404),
        "Traffic": (0.481, 0.304), "Weather": (0.259, 0.281), "Solar": (0.270, 0.307),
    },
    "DLinear": {
        "ETTm1": (0.403, 0.407), "ETTm2": (0.350, 0.401), "ETTh1": (0.456, 0.452),
        "ETTh2": (0.559, 0.515), "ECL": (0.212, 0.300), "Exchange": (0.354, 0.414),
        "Traffic": (0.625, 0.383), "Weather": (0.265, 0.317), "Solar": (0.330, 0.401),
    },
    "Autoformer": {
        "ETTm1": (0.588, 0.517), "ETTm2": (0.327, 0.371), "ETTh1": (0.496, 0.487),
        "ETTh2": (0.450, 0.459), "ECL": (0.227, 0.338), "Exchange": (0.613, 0.539),
        "Traffic": (0.628, 0.379), "Weather": (0.338, 0.382), "Solar": (0.885, 0.711),
    },
}


def canonical(row: pd.Series) -> Optional[str]:
    data, dp = str(row.get("data", "")), str(row.get("data_path", ""))
    if data in {"ETTm1", "ETTm2", "ETTh1", "ETTh2"}:
        return data
    if data == "Solar" or dp.startswith("solar"):
        return "Solar"
    m = {"electricity": "ECL", "exchange_rate": "Exchange", "traffic": "Traffic", "weather": "Weather"}
    return m.get(dp, m.get(data))


def load_root(path: Path) -> pd.DataFrame:
    df = collect_results(str(path))
    if df.empty:
        return df
    df = df.copy()
    df["dataset"] = df.apply(canonical, axis=1)
    df = df.dropna(subset=["dataset"])
    return df


def get_cell(df: pd.DataFrame, model: str, dataset: str, pl: int, loss: str) -> Optional[Tuple[float, float]]:
    m = df[(df["model"] == model) & (df["dataset"] == dataset) & (df["pred_len"] == pl) & (df["loss"] == loss)]
    if m.empty:
        return None
    return float(m.iloc[0]["mse"]), float(m.iloc[0]["mae"])


def avg_cells(cells: List[Optional[Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
    if any(c is None for c in cells):
        return None
    return float(np.mean([c[0] for c in cells])), float(np.mean([c[1] for c in cells]))


def fmt(v: Optional[float], best: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v:.3f}"
    return rf"\textbf{{{s}}}" if best else s


def row_cells(values: List[Optional[Tuple[float, float]]]) -> str:
    """Format (mse, mae) pairs, bolding best mse and best mae independently."""
    mses = [v[0] if v else None for v in values]
    maes = [v[1] if v else None for v in values]
    best_mse = min([m for m in mses if m is not None], default=None)
    best_mae = min([m for m in maes if m is not None], default=None)
    parts = []
    for mse, mae in zip(mses, maes):
        parts.append(fmt(mse, best=(mse is not None and mse == best_mse)))
        parts.append(fmt(mae, best=(mae is not None and mae == best_mae)))
    return " & ".join(parts)


def table_main(df: pd.DataFrame, summary: dict) -> str:
    lines = []
    win = {loss: 0 for loss in LOSS_ORDER}
    win_mae = {loss: 0 for loss in LOSS_ORDER}
    n_complete = 0
    for ds in DATASET_ORDER:
        block = []
        per_loss_cells: Dict[str, List[Optional[Tuple[float, float]]]] = {l: [] for l in LOSS_ORDER}
        for pl in PRED_LENS:
            vals = [get_cell(df, "MLP", ds, pl, loss) for loss in LOSS_ORDER]
            for loss, v in zip(LOSS_ORDER, vals):
                per_loss_cells[loss].append(v)
            block.append(f" & {pl} & " + row_cells(vals) + r" \\")
            mses = [v[0] if v else None for v in vals]
            if all(m is not None for m in mses):
                n_complete += 1
                w = LOSS_ORDER[int(np.argmin(mses))]
                win[w] += 1
                maes = [v[1] for v in vals]
                win_mae[LOSS_ORDER[int(np.argmin(maes))]] += 1
        avg_vals = [avg_cells(per_loss_cells[loss]) for loss in LOSS_ORDER]
        block.append(r" \cmidrule(lr){2-8}")
        block.append(r" & Avg & " + row_cells(avg_vals) + r" \\")
        label = DATASET_LABEL[ds]
        lines.append(rf"\multirow{{5}}{{*}}{{\rotatebox{{90}}{{{label}}}}}")
        lines.extend(block)
        lines.append(r"\midrule")
        for loss, v in zip(LOSS_ORDER, avg_vals):
            if v is not None:
                summary.setdefault("main_avg", {}).setdefault(ds, {})[loss] = [round(v[0], 3), round(v[1], 3)]
    win_line = (
        r"\multicolumn{2}{c|}{$1^{\text{st}}$ count (MSE/MAE)} & "
        + " & ".join(rf"\multicolumn{{2}}{{c{'|' if i < 2 else ''}}}{{{win[l]} / {win_mae[l]}}}" for i, l in enumerate(LOSS_ORDER))
        + r" \\"
    )
    summary["main_wins_mse"] = win
    summary["main_wins_mae"] = win_mae
    summary["main_complete_cells"] = n_complete

    header = r"""\begin{table}[!t]
    \centering
    \caption{Multivariate long-horizon forecasting with a fixed 3-layer MLP trained under each loss. Input length is 96 and results are test-set MSE/MAE (lower is better) with prediction lengths $\{96,192,336,720\}$; Avg is the mean over the four horizons. \textbf{Bold} marks the best value in each row. The final row counts horizon-level firsts by test MSE / MAE across all """ + str(n_complete) + r""" completed dataset--horizon cells.}
    \label{tab:main}
    \scriptsize
    \setlength{\tabcolsep}{2.6pt}
    \renewcommand{\arraystretch}{0.9}
    \begin{tabular}{c|c|cc|cc|cc}
    \toprule
    \multicolumn{2}{c|}{Training loss} &
    \multicolumn{2}{c|}{MSE} &
    \multicolumn{2}{c|}{TILDE-Q} &
    \multicolumn{2}{c}{HNMD (ours)} \\
    \multicolumn{2}{c|}{Metric} &
    MSE & MAE & MSE & MAE & MSE & MAE \\
    \midrule
"""
    footer = win_line + "\n" + r"""    \bottomrule
    \end{tabular}
\end{table}
"""
    return header + "\n".join(lines) + "\n" + footer


def table_arch(df: pd.DataFrame, summary: dict) -> str:
    models = ["DLinear", "iTransformer"]
    lines = []
    win: Dict[str, Dict[str, int]] = {m: {l: 0 for l in LOSS_ORDER} for m in models}
    for ds in ETT:
        per_model_avg = {}
        block = []
        cells: Dict[Tuple[str, str], List] = {(m, l): [] for m in models for l in LOSS_ORDER}
        for pl in PRED_LENS:
            groups = []
            for m in models:
                vals = [get_cell(df, m, ds, pl, loss) for loss in LOSS_ORDER]
                for loss, v in zip(LOSS_ORDER, vals):
                    cells[(m, loss)].append(v)
                mses = [v[0] if v else None for v in vals]
                if all(x is not None for x in mses):
                    win[m][LOSS_ORDER[int(np.argmin(mses))]] += 1
                groups.append(row_cells(vals))
            block.append(f" & {pl} & " + " & ".join(groups) + r" \\")
        avg_groups = []
        for m in models:
            avg_vals = [avg_cells(cells[(m, l)]) for l in LOSS_ORDER]
            per_model_avg[m] = avg_vals
            avg_groups.append(row_cells(avg_vals))
        block.append(r" \cmidrule(lr){2-14}")
        block.append(r" & Avg & " + " & ".join(avg_groups) + r" \\")
        lines.append(rf"\multirow{{5}}{{*}}{{\rotatebox{{90}}{{{DATASET_LABEL[ds]}}}}}")
        lines.extend(block)
        lines.append(r"\midrule")
        for m in models:
            for loss, v in zip(LOSS_ORDER, per_model_avg[m]):
                if v is not None:
                    summary.setdefault("arch_avg", {}).setdefault(m, {}).setdefault(ds, {})[loss] = [round(v[0], 3), round(v[1], 3)]
    summary["arch_wins_mse"] = win
    header = r"""\begin{table*}[!t]
    \centering
    \caption{Loss-function comparison for two further architectures on the ETT benchmarks (input 96, test MSE/MAE). DLinear uses HNMD hyperparameters tuned for DLinear; iTransformer reuses the MLP-tuned configurations unchanged. \textbf{Bold} marks the best value within each architecture for each row.}
    \label{tab:arch}
    \resizebox{0.98\textwidth}{!}{
    \begin{tabular}{c|c|cc|cc|cc||cc|cc|cc}
    \toprule
    \multicolumn{2}{c|}{Model} & \multicolumn{6}{c||}{DLinear} & \multicolumn{6}{c}{iTransformer} \\
    \midrule
    \multicolumn{2}{c|}{Training loss} &
    \multicolumn{2}{c|}{MSE} & \multicolumn{2}{c|}{TILDE-Q} & \multicolumn{2}{c||}{HNMD} &
    \multicolumn{2}{c|}{MSE} & \multicolumn{2}{c|}{TILDE-Q} & \multicolumn{2}{c}{HNMD} \\
    \multicolumn{2}{c|}{Metric} &
    MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table*}
"""
    return header + "\n".join(lines) + footer


def _bold_min_col(rows: List[List[Optional[float]]]) -> List[List[str]]:
    """Format columns, bolding the min of each column."""
    n_cols = len(rows[0])
    mins = []
    for c in range(n_cols):
        vals = [r[c] for r in rows if r[c] is not None]
        mins.append(min(vals) if vals else None)
    out = []
    for r in rows:
        out.append([fmt(v, best=(v is not None and v == mins[c])) for c, v in enumerate(r)])
    return out


def table_ablation(full_df: pd.DataFrame, abl: Dict[str, pd.DataFrame], summary: dict) -> str:
    col_heads = [f"{ds} {pl}" for ds, pl in ABLATION_CELLS]
    variant_rows: List[Tuple[str, List[Optional[float]], List[Optional[float]]]] = []

    full_mse, full_mae = [], []
    for ds, pl in ABLATION_CELLS:
        v = get_cell(full_df, "MLP", ds, pl, "mssd")
        full_mse.append(v[0] if v else None)
        full_mae.append(v[1] if v else None)
    variant_rows.append((r"HNMD (full)", full_mse, full_mae))

    for key, label in ABLATION_VARIANTS:
        df = abl.get(key, pd.DataFrame())
        mses, maes = [], []
        for ds, pl in ABLATION_CELLS:
            v = get_cell(df, "MLP", ds, pl, "mssd") if not df.empty else None
            mses.append(v[0] if v else None)
            maes.append(v[1] if v else None)
        variant_rows.append((label, mses, maes))

    mse_matrix = [r[1] for r in variant_rows]
    fmt_matrix = _bold_min_col(mse_matrix)

    lines = []
    for (label, mses, maes), fr in zip(variant_rows, fmt_matrix):
        mean_val = np.mean([m for m in mses if m is not None]) if any(m is not None for m in mses) else None
        lines.append(f"    {label} & " + " & ".join(fr) + f" & {fmt(mean_val)}" + r" \\")
        if label == r"HNMD (full)":
            lines.append(r"    \midrule")
        summary.setdefault("ablation_mean_mse", {})[label] = round(float(mean_val), 4) if mean_val is not None else None

    header = r"""\begin{table*}[!t]
    \centering
    \caption{Ablation study (MLP, test MSE). Each component of HNMD is removed or reduced in isolation, starting from the tuned configuration of each dataset--horizon cell; cells are chosen so that all three domain weights are active in the tuned configuration. $M$ is the number of decomposition levels and $x$ the level-importance exponent. \textbf{Bold} marks the best value per column.}
    \label{tab:ablation}
    \resizebox{0.98\textwidth}{!}{
    \begin{tabular}{l|cccccc|c}
    \toprule
    Variant & """ + " & ".join(col_heads) + r""" & Mean \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table*}
"""
    return header + "\n".join(lines) + "\n" + footer


def table_default(df_main: pd.DataFrame, df_default: pd.DataFrame, summary: dict) -> str:
    lines = []
    deltas = []
    for ds in DATASET_ORDER:
        mse_cells = [get_cell(df_main, "MLP", ds, pl, "mse") for pl in PRED_LENS]
        tuned_cells = [get_cell(df_main, "MLP", ds, pl, "mssd") for pl in PRED_LENS]
        def_cells = [get_cell(df_default, "MLP", ds, pl, "mssd") for pl in PRED_LENS]
        vals = [avg_cells(mse_cells), avg_cells(def_cells), avg_cells(tuned_cells)]
        lines.append(f"    {DATASET_LABEL[ds]} & " + row_cells(vals) + r" \\")
        if vals[0] and vals[1]:
            deltas.append((vals[1][0] - vals[0][0]) / vals[0][0])
        for name, v in zip(["mse", "hnmd_default", "hnmd_tuned"], vals):
            if v is not None:
                summary.setdefault("default_avg", {}).setdefault(ds, {})[name] = [round(v[0], 3), round(v[1], 3)]
    if deltas:
        summary["default_vs_mse_relative_mse"] = round(float(np.mean(deltas)), 4)
    header = r"""\begin{table}[!t]
    \centering
    \caption{Per-dataset averages (over four horizons) for MLPs trained with MSE, with a \emph{single} fixed HNMD configuration shared by all datasets and horizons ($\alpha{=}0.01$, $\beta{=}1$, $\gamma{=}1$, $\kappa{=}5$, $x{=}1$, $M{=}5$), and with per-cell tuned HNMD. \textbf{Bold} marks the best value per row.}
    \label{tab:default}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|cc|cc|cc}
    \toprule
    & \multicolumn{2}{c|}{MSE loss} & \multicolumn{2}{c|}{HNMD (fixed)} & \multicolumn{2}{c}{HNMD (tuned)} \\
    Dataset & MSE & MAE & MSE & MAE & MSE & MAE \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table}
"""
    return header + "\n".join(lines) + "\n" + footer


def table_seeds(df_main: pd.DataFrame, seed_dfs: Dict[int, pd.DataFrame], summary: dict) -> str:
    lines = []
    for ds in ["ETTh1", "ETTm1"]:
        for pl in PRED_LENS:
            cells = []
            for loss in LOSS_ORDER:
                vals = []
                v = get_cell(df_main, "MLP", ds, pl, loss)
                if v is not None:
                    vals.append(v[0])
                for seed, sdf in seed_dfs.items():
                    v = get_cell(sdf, "MLP", ds, pl, loss) if not sdf.empty else None
                    if v is not None:
                        vals.append(v[0])
                cells.append((float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None)
            means = [c[0] if c else None for c in cells]
            best = min([m for m in means if m is not None], default=None)
            row = []
            for c in cells:
                if c is None:
                    row.append("--")
                else:
                    s = f"{c[0]:.3f}$\\pm${c[1]:.3f}"
                    row.append(rf"\textbf{{{s}}}" if c[0] == best else s)
            lines.append(f"    {ds} & {pl} & " + " & ".join(row) + r" \\")
            summary.setdefault("seeds", {})[f"{ds}_{pl}"] = {
                loss: (None if c is None else [round(c[0], 4), round(c[1], 4), c[2]])
                for loss, c in zip(LOSS_ORDER, cells)
            }
        lines.append(r"    \midrule")
    header = r"""\begin{table}[!t]
    \centering
    \caption{Seed robustness: test MSE (mean$\pm$std over seeds \{2024, 2025, 2026\}) for the MLP. \textbf{Bold} marks the best mean per row.}
    \label{tab:seeds}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|c|c|c|c}
    \toprule
    Dataset & Horizon & MSE & TILDE-Q & HNMD \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table}
"""
    return header + "\n".join(lines[:-1]) + "\n" + footer


def table_overhead(bench_path: Path, summary: dict) -> Optional[str]:
    if not bench_path.is_file():
        return None
    bench = json.load(open(bench_path))
    cases = []
    for key in bench:
        base, pl, loss = key.rsplit("_", 2)[0], key.rsplit("_", 2)[1], key.rsplit("_", 2)[2]
        cases.append((base, int(pl.replace("pl", "")), loss))
    uniq = sorted({(b, p) for b, p, _ in cases}, key=lambda x: (x[0], x[1]))
    lines = []
    for base, pl in uniq:
        get = lambda loss: bench.get(f"{base}_pl{pl}_{loss}")
        mse_t, tq_t, hn_t = get("mse"), get("tildeq"), get("mssd")
        if mse_t is None:
            continue
        ratio_h = hn_t / mse_t if hn_t else None
        ratio_t = tq_t / mse_t if tq_t else None
        label = {"ETTh1": "ETTh1", "weather": "Weather", "electricity": "ECL"}.get(base, base)
        lines.append(
            f"    {label} & {pl} & {mse_t*1000:.1f} & {tq_t*1000:.1f} & {hn_t*1000:.1f} & "
            f"{ratio_t:.1f}$\\times$ & {ratio_h:.1f}$\\times$ \\\\"
        )
        summary.setdefault("overhead", {})[f"{base}_{pl}"] = {
            "mse_ms": round(mse_t * 1000, 2), "tildeq_ms": round(tq_t * 1000, 2),
            "mssd_ms": round(hn_t * 1000, 2), "mssd_over_mse": round(ratio_h, 2),
        }
    header = r"""\begin{table}[!t]
    \centering
    \caption{Training cost: median wall-clock time per optimizer step (ms) for the MLP on one NVIDIA L4 GPU, and slowdown relative to MSE training. Inference cost is identical for all losses.}
    \label{tab:overhead}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|c|ccc|cc}
    \toprule
    Dataset & Horizon & MSE & TILDE-Q & HNMD & TILDE-Q$/$MSE & HNMD$/$MSE \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table}
"""
    return header + "\n".join(lines) + "\n" + footer


def table_published(df_main: pd.DataFrame) -> str:
    lines = []
    models = ["iTransformer", "PatchTST", "RLinear", "DLinear", "Autoformer"]
    for ds in DATASET_ORDER:
        mlp_cells = [get_cell(df_main, "MLP", ds, pl, "mssd") for pl in PRED_LENS]
        hn = avg_cells(mlp_cells)
        vals: List[Optional[Tuple[float, float]]] = [hn]
        for m in models:
            entry = PUBLISHED_BASELINES[m].get(ds)
            vals.append(entry if entry else None)
        lines.append(f"    {DATASET_LABEL[ds]} & " + row_cells(vals) + r" \\")
    header = r"""\begin{table*}[!t]
    \centering
    \caption{Context: HNMD-trained 3-layer MLP (ours, dataset averages over four horizons) next to published MSE-trained forecasters at input length 96. Baseline numbers are quoted from \citet{itransformer} and are shown for reference only --- they involve different architectures, capacities, and tuning budgets. \textbf{Bold} marks the best value per row.}
    \label{tab:published}
    \resizebox{0.98\textwidth}{!}{
    \begin{tabular}{l|cc|cc|cc|cc|cc|cc}
    \toprule
    & \multicolumn{2}{c|}{\textbf{MLP+HNMD (ours)}} & \multicolumn{2}{c|}{iTransformer} & \multicolumn{2}{c|}{PatchTST} & \multicolumn{2}{c|}{RLinear} & \multicolumn{2}{c|}{DLinear} & \multicolumn{2}{c}{Autoformer} \\
    Dataset & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table*}
"""
    return header + "\n".join(lines) + "\n" + footer


def table_hyper() -> str:
    lines = []
    for ds in DATASET_ORDER:
        for pl in PRED_LENS:
            hp = get_mssd_hyperparams("MLP", INTERNAL_KEY[ds], pl)
            first = rf"\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}" if pl == 96 else ""
            lines.append(
                f"    {first} & {pl} & {hp.max_levels} & 3 & {hp.alpha:g} & {hp.beta:g} & {hp.gamma:g} & "
                f"{hp.knot_multiplier:g} & {hp.spline_criterion_exponent} \\\\"
            )
        lines.append(r"    \midrule")
    header = r"""\begin{table}[!t]
    \centering
    \caption{Tuned HNMD hyperparameters per dataset and horizon (MLP): decomposition depth $M$, spline degree $d$, domain weights $\alpha,\beta,\gamma$, knot multiplier $\kappa$, and level-importance exponent $x$.}
    \label{tab:hyper}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{l|c|c|c|c|c|c|c|c}
    \toprule
    Dataset & Horizon & $M$ & $d$ & $\alpha$ & $\beta$ & $\gamma$ & $\kappa$ & $x$ \\
    \midrule
"""
    footer = r"""    \bottomrule
    \end{tabular}
    }
\end{table}
"""
    return header + "\n".join(lines[:-1]) + "\n" + footer


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    df_main = load_root(ROOT / "results")
    if df_main.empty:
        raise SystemExit("No results under ./results — download from the Modal volume first.")

    abl = {}
    for key, _ in ABLATION_VARIANTS:
        p = ROOT / "results_ablation" / key
        abl[key] = load_root(p) if p.is_dir() else pd.DataFrame()

    df_default = load_root(ROOT / "results_default") if (ROOT / "results_default").is_dir() else pd.DataFrame()

    seed_dfs = {}
    seeds_root = ROOT / "results_seeds"
    if seeds_root.is_dir():
        for sub in sorted(seeds_root.iterdir()):
            if sub.is_dir() and sub.name.startswith("s"):
                seed_dfs[int(sub.name[1:])] = load_root(sub)

    (OUT_DIR / "tab_main.tex").write_text(table_main(df_main, summary))
    (OUT_DIR / "tab_arch.tex").write_text(table_arch(df_main, summary))
    (OUT_DIR / "tab_ablation.tex").write_text(table_ablation(df_main, abl, summary))
    if not df_default.empty:
        (OUT_DIR / "tab_default.tex").write_text(table_default(df_main, df_default, summary))
    if seed_dfs:
        (OUT_DIR / "tab_seeds.tex").write_text(table_seeds(df_main, seed_dfs, summary))
    overhead = table_overhead(ROOT / "tables" / "loss_overhead_benchmark.json", summary)
    if overhead:
        (OUT_DIR / "tab_overhead.tex").write_text(overhead)
    (OUT_DIR / "tab_published.tex").write_text(table_published(df_main))
    (OUT_DIR / "tab_hyper.tex").write_text(table_hyper())

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote tables to {OUT_DIR}")
    for k in ["main_wins_mse", "main_wins_mae", "main_complete_cells"]:
        if k in summary:
            print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
