"""Load experiment outputs and MSSD decomposition for paper figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.hyperparams import MSSDHyperparams, get_mssd_hyperparams
from filltables import INTERNAL_DATASET_KEYS, canonical_dataset
from utils.results_parser import collect_results, parse_setting_name

# Default reference run for time-series panels (fig1–3, fig5–6).
REF_MODEL = "MLP"
REF_DATASET = "ETTm1"
REF_PRED_LEN = 96
REF_SAMPLE = 0
REF_CHANNEL = 0

FIG4_DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Traffic"]

LOSS_REPO = {"MSE": "mse", "Tilde-Q": "tildeq", "HNMD": "mssd"}
PAPER_LOSS_FROM_REPO = {"mse": "MSE", "tildeq": "Tilde-Q", "mssd": "HNMD"}

# Runs required before any figure is written (raises FileNotFoundError if missing).
REQUIRED_RUNS = [
    (REF_MODEL, REF_DATASET, REF_PRED_LEN, "HNMD"),
    (REF_MODEL, REF_DATASET, REF_PRED_LEN, "MSE"),
]
REQUIRED_FIG4_LOSSES = ["MSE", "Tilde-Q", "HNMD"]


def find_results_dir() -> Path:
    for candidate in (ROOT / "results", ROOT / "results" / "results"):
        if candidate.is_dir() and any(candidate.rglob("metrics.npy")):
            return candidate
    return ROOT / "results"


def load_results_df(results_dir: Optional[Path] = None) -> pd.DataFrame:
    results_dir = results_dir or find_results_dir()
    return collect_results(str(results_dir))


def _metrics_mse(run_dir: str) -> float:
    metrics = np.load(os.path.join(run_dir, "metrics.npy"))
    return float(metrics[1])


def find_run_dir(
    model: str,
    paper_dataset: str,
    pred_len: int,
    loss: str,
    results_dir: Optional[Path] = None,
) -> Optional[str]:
    """Return path to best (lowest test MSE) run folder for this key."""
    results_dir = results_dir or find_results_dir()
    if not results_dir.is_dir():
        return None

    loss_repo = LOSS_REPO.get(loss, loss.lower())
    best: Optional[tuple[float, str]] = None

    for root, _, files in os.walk(results_dir):
        if "pred.npy" not in files or "true.npy" not in files:
            continue
        folder = os.path.basename(root)
        parsed = parse_setting_name(folder)
        if parsed is None:
            continue
        if parsed["model"] != model or int(parsed["pred_len"]) != int(pred_len):
            continue
        if parsed["loss"].lower() != loss_repo:
            continue
        if canonical_dataset(pd.Series(parsed)) != paper_dataset:
            continue
        mse = _metrics_mse(root)
        if best is None or mse < best[0]:
            best = (mse, root)

    return best[1] if best else None


def load_run_arrays(
    model: str = REF_MODEL,
    paper_dataset: str = REF_DATASET,
    pred_len: int = REF_PRED_LEN,
    loss: str = "HNMD",
    results_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load pred, true, metrics from results; raises FileNotFoundError if missing."""
    run_dir = find_run_dir(model, paper_dataset, pred_len, loss, results_dir)
    if run_dir is None:
        raise FileNotFoundError(
            f"No results for {model} / {paper_dataset} / pl={pred_len} / {loss} under {results_dir or find_results_dir()}"
        )
    pred = np.load(os.path.join(run_dir, "pred.npy"))
    true = np.load(os.path.join(run_dir, "true.npy"))
    metrics = np.load(os.path.join(run_dir, "metrics.npy"))
    return pred, true, metrics, run_dir


def mlp_dataset_averages(results_dir: Optional[Path] = None) -> dict[str, dict[str, float]]:
    """
    Dataset-average test MSE (mean over horizons) from ./results/.

    Groups by paper dataset name (not bare ``data=custom``) so Traffic and Weather
    are not collapsed by ``collect_results`` deduplication.
    """
    df = load_results_df(results_dir)
    if df.empty:
        raise FileNotFoundError(f"No metrics under {results_dir or find_results_dir()}")

    work = df[df["model"] == REF_MODEL].copy()
    work["paper_dataset"] = work.apply(canonical_dataset, axis=1)
    work = work.dropna(subset=["paper_dataset"])
    work["paper_loss"] = work["loss"].map(
        lambda x: PAPER_LOSS_FROM_REPO.get(str(x).lower(), str(x))
    )

    out: dict[str, dict[str, float]] = {ds: {} for ds in FIG4_DATASETS}
    grouped = work.groupby(["paper_dataset", "paper_loss"], as_index=False)["mse"].mean()
    for row in grouped.itertuples():
        ds, loss, mse = row.paper_dataset, row.paper_loss, float(row.mse)
        if ds in out and loss in REQUIRED_FIG4_LOSSES:
            out[ds][loss] = mse

    missing = []
    for ds in FIG4_DATASETS:
        for loss in REQUIRED_FIG4_LOSSES:
            if loss not in out.get(ds, {}):
                missing.append(f"{ds} / {loss}")
    if missing:
        raise FileNotFoundError(
            "Missing MLP test results for fig4 (dataset / loss):\n  "
            + "\n  ".join(missing)
        )
    return out


def _internal_dataset(paper_dataset: str) -> str:
    return INTERNAL_DATASET_KEYS.get(paper_dataset, paper_dataset)


def get_hyperparams(
    paper_dataset: str = REF_DATASET,
    pred_len: int = REF_PRED_LEN,
    model: str = REF_MODEL,
) -> MSSDHyperparams:
    return get_mssd_hyperparams(model, _internal_dataset(paper_dataset), pred_len)


def _require_torch():
    import torch

    return torch


def nurbs_decompose_1d(
    y: np.ndarray,
    hp: Optional[MSSDHyperparams] = None,
    paper_dataset: str = REF_DATASET,
    pred_len: Optional[int] = None,
    y_multivariate: Optional[np.ndarray] = None,
    channel: int = REF_CHANNEL,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """
    Run MSSD NURBS decomposition (real forward pass).

    Use ``y_multivariate`` shape (T, C) when available — single-channel decomposition
    can fail with tuned knot multipliers; training always uses all variates.
    """
    torch = _require_torch()
    from utils.mssd import NURBSBasisCache, get_nurbs_weights, nurbs_decompose_time_series

    if hp is None:
        hp = get_hyperparams(paper_dataset, pred_len or len(y))

    if y_multivariate is not None:
        y_t = torch.tensor(y_multivariate[np.newaxis, ...], dtype=torch.float32)
    else:
        y_t = torch.tensor(y, dtype=torch.float32).reshape(1, -1, 1)
    cache = NURBSBasisCache()
    decomposed = nurbs_decompose_time_series(
        y_t,
        cache,
        max_levels=hp.max_levels,
        degree=3,
        knot_scaling_factor=int(hp.knot_multiplier),
    )
    if decomposed.numel() == 0:
        raise RuntimeError("NURBS decomposition produced no levels")

    n_base = (decomposed.shape[-1] + 1) // 2
    ch = min(channel, decomposed.shape[2] - 1)
    levels = [decomposed[0, :, ch, i].detach().cpu().numpy() for i in range(n_base)]

    residual = y.astype(np.float64).copy()
    for lev in levels:
        residual -= lev

    seq_len = y_t.shape[1]
    num_knots = int(
        max(2 * 3 + 2, min(seq_len // 2, 2 ** (max(n_base, 1)) + seq_len / 6))
        * hp.knot_multiplier
    )
    window_size = max(1, seq_len // max(num_knots, 1))
    res_t = torch.tensor(residual, dtype=torch.float32).reshape(1, -1, 1)
    weights = get_nurbs_weights(res_t, window_size)[0, :, 0].detach().cpu().numpy()

    return levels, residual, weights, n_base


def level_gradient_weights(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hp: Optional[MSSDHyperparams] = None,
    paper_dataset: str = REF_DATASET,
    pred_len: Optional[int] = None,
    channel: int = REF_CHANNEL,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-level decomposition weights as in NurbsDecomposition.forward."""
    torch = _require_torch()
    from utils.mssd import NURBSBasisCache, decompose_and_reconstruct, nurbs_decompose_time_series

    if hp is None:
        hp = get_hyperparams(paper_dataset, pred_len or len(y_true))

    if y_true.ndim == 1:
        yt = torch.tensor(y_true, dtype=torch.float32).reshape(1, -1, 1)
        yp = torch.tensor(y_pred, dtype=torch.float32).reshape(1, -1, 1)
    else:
        yt = torch.tensor(y_true[np.newaxis, ...], dtype=torch.float32)
        yp = torch.tensor(y_pred[np.newaxis, ...], dtype=torch.float32)
    cache = NURBSBasisCache()

    dec_true = nurbs_decompose_time_series(
        yt,
        cache,
        max_levels=hp.max_levels,
        degree=3,
        knot_scaling_factor=int(hp.knot_multiplier),
    )
    dec_pred = decompose_and_reconstruct(
        yp,
        cache,
        dec_true,
        hp.max_levels,
        exponent=hp.spline_criterion_exponent,
        degree=3,
        knot_scaling_factor=int(hp.knot_multiplier),
    )

    level_losses = torch.mean(
        torch.abs(dec_pred - dec_true) ** hp.spline_criterion_exponent,
        dim=tuple(range(dec_pred.ndim - 1)),
    )
    weights = (level_losses / (level_losses.sum() + 1e-8)).detach().cpu().numpy()

    n_base = (dec_true.shape[-1] + 1) // 2
    ch = min(channel, dec_true.shape[2] - 1)
    base_levels = [dec_true[0, :, ch, i].detach().cpu().numpy() for i in range(n_base)]
    return weights, base_levels


def decomposed_series_for_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hp: Optional[MSSDHyperparams] = None,
    paper_dataset: str = REF_DATASET,
    pred_len: Optional[int] = None,
    channel: int = REF_CHANNEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Decomposed target vs prediction for one channel (mean over level index)."""
    torch = _require_torch()
    from utils.mssd import NURBSBasisCache, decompose_and_reconstruct, nurbs_decompose_time_series

    if hp is None:
        hp = get_hyperparams(paper_dataset, pred_len or len(y_true))

    if y_true.ndim == 1:
        yt = torch.tensor(y_true, dtype=torch.float32).reshape(1, -1, 1)
        yp = torch.tensor(y_pred, dtype=torch.float32).reshape(1, -1, 1)
        ch = 0
    else:
        yt = torch.tensor(y_true[np.newaxis, ...], dtype=torch.float32)
        yp = torch.tensor(y_pred[np.newaxis, ...], dtype=torch.float32)
        ch = min(channel, yt.shape[2] - 1)
    cache = NURBSBasisCache()
    dec_true = nurbs_decompose_time_series(
        yt,
        cache,
        max_levels=hp.max_levels,
        degree=3,
        knot_scaling_factor=int(hp.knot_multiplier),
    )
    dec_pred = decompose_and_reconstruct(
        yp,
        cache,
        dec_true,
        hp.max_levels,
        exponent=hp.spline_criterion_exponent,
        degree=3,
        knot_scaling_factor=int(hp.knot_multiplier),
    )
    t_true = dec_true[0, :, ch, :].mean(dim=-1).detach().cpu().numpy()
    t_pred = dec_pred[0, :, ch, :].mean(dim=-1).detach().cpu().numpy()
    return t_true, t_pred


def extract_window(
    pred: np.ndarray,
    true: np.ndarray,
    sample: int = REF_SAMPLE,
    channel: int = REF_CHANNEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Single (pred_len,) slice from test arrays (one channel)."""
    return (
        np.asarray(true[sample, :, channel], dtype=np.float64),
        np.asarray(pred[sample, :, channel], dtype=np.float64),
    )


def extract_window_mv(
    pred: np.ndarray,
    true: np.ndarray,
    sample: int = REF_SAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Full (pred_len, n_vars) window — required for stable NURBS decomposition."""
    return (
        np.asarray(true[sample], dtype=np.float64),
        np.asarray(pred[sample], dtype=np.float64),
    )


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to generate paper figures (MSSD decomposition). "
            "Use the project virtualenv, e.g. env/bin/python paper/generate_figures.py"
        ) from exc


def validate_figure_prerequisites(results_dir: Optional[Path] = None) -> None:
    """Fail fast if ./results/ or required runs are missing."""
    require_torch()
    results_dir = results_dir or find_results_dir()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    missing_runs = []
    for model, dataset, pred_len, loss in REQUIRED_RUNS:
        if find_run_dir(model, dataset, pred_len, loss, results_dir) is None:
            missing_runs.append(f"{dataset} pl={pred_len} loss={loss}")
    if missing_runs:
        raise FileNotFoundError(
            f"No run folders under {results_dir} for:\n  " + "\n  ".join(missing_runs)
        )

    # fig4 validates all metrics inside mlp_dataset_averages()
    mlp_dataset_averages(results_dir)
