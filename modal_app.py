#!/usr/bin/env python3
"""Run filltables missing experiments on Modal with GPU workers.

Prerequisites:
  pip install modal
  modal setup
  ./modal_setup.sh upload

Examples:
  modal run modal_app.py --dry-run              # list jobs, no Modal execution
  modal run modal_app.py --parallel 4           # 4 concurrent T4 workers
  modal run modal_app.py --parallel 4 --gpu L4  # use L4 instead of T4
  modal run --detach modal_app.py --parallel 4 --gpu L4 --models MLP \\
      --datasets Solar,weather,electricity,exchange_rate,traffic --pred-lens 96
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List

import modal

VOLUME_NAME = os.environ.get("HNMD_MODAL_VOLUME", "hnmd-workspace")
WORKSPACE = "/workspace"
APP_ROOT = "/root/app"
ARTIFACT_DIRS = ("data", "results", "checkpoints", "test_results", "tables")
SUPPORTED_GPUS = ("T4", "L4", "A10")

app = modal.App("hnmd-experiments")
workspace = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=APP_ROOT,
        ignore=[
            "env",
            ".venv",
            "venv",
            ".git",
            "checkpoints",
            "results",
            "test_results",
            "data",
            "__pycache__",
            ".DS_Store",
        ],
    )
)

FUNCTION_KWARGS = dict(
    image=image,
    volumes={WORKSPACE: workspace},
    timeout=60 * 60 * 6,
    retries=1,
)


def link_workspace() -> None:
    """Symlink volume paths into the repo root so hardcoded ./ paths work."""
    for name in ARTIFACT_DIRS:
        src = os.path.join(WORKSPACE, name)
        dst = os.path.join(APP_ROOT, name)
        os.makedirs(src, exist_ok=True)
        if os.path.islink(dst):
            continue
        if os.path.isdir(dst) and not os.path.islink(dst):
            try:
                os.rmdir(dst)
            except OSError:
                pass
        if not os.path.lexists(dst):
            os.symlink(src, dst)


def normalize_argv_for_container(argv: List[str]) -> List[str]:
    """Replace the local Mac venv python path with the container interpreter."""
    if not argv:
        return argv
    patched = list(argv)
    patched[0] = sys.executable
    for i, arg in enumerate(patched):
        if arg == "run.py":
            patched[i] = os.path.join(APP_ROOT, "run.py")
    return patched


def execute_job(argv: List[str]) -> int:
    os.chdir(APP_ROOT)
    link_workspace()
    argv = normalize_argv_for_container(argv)
    print("Running:", " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=APP_ROOT)
    workspace.commit()
    return result.returncode


def _parse_csv_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _resolve_dataset_names(names: set[str]) -> set[str]:
    """Map CLI dataset aliases to internal config keys (e.g. ECL -> electricity)."""
    from configs.datasets import get_dataset

    resolved: set[str] = set()
    for name in names:
        try:
            resolved.add(get_dataset(name).name)
        except KeyError as exc:
            raise SystemExit(f"Unknown dataset '{name}'.") from exc
    return resolved


def discover_job_argv(
    seed: int = 2024,
    itr: int = 1,
    models: set[str] | None = None,
    datasets: set[str] | None = None,
    pred_lens: set[int] | None = None,
) -> List[List[str]]:
    """Reuse filltables job discovery; requires local ./data for filtering."""
    from filltables import (
        TABLE2_LOSSES,
        TABLE2_MODELS,
        TABLE3_LOSSES,
        TABLE3_MODELS,
        aggregate_results,
        build_experiment_jobs,
        missing_result_keys,
    )
    from utils.results_parser import collect_results

    raw = collect_results("./results")
    agg = aggregate_results(raw, strategy="mean")
    missing2 = missing_result_keys(agg, TABLE2_MODELS, TABLE2_LOSSES)
    missing3 = missing_result_keys(agg, TABLE3_MODELS, TABLE3_LOSSES)
    missing = missing2 | missing3
    jobs, skipped = build_experiment_jobs(missing, seed=seed, itr=itr)

    if models is not None:
        jobs = [job for job in jobs if job.model in models]
    if datasets is not None:
        dataset_keys = _resolve_dataset_names(datasets)
        jobs = [job for job in jobs if job.dataset.name in dataset_keys]
    if pred_lens is not None:
        jobs = [job for job in jobs if job.pred_len in pred_lens]

    model_order = {
        "MLP": 0,
        "DLinear": 1,
        "SOFTS": 2,
        "NLinear": 3,
        "Autoformer": 4,
        "iTransformer": 5,
    }
    jobs.sort(key=lambda j: (model_order.get(j.model, 99), j.dataset.name, j.pred_len, j.loss))

    from configs.experiment_runner import build_run_argv

    argv_list = [build_run_argv(job) for job in jobs]
    filters = []
    if models:
        filters.append(f"models={sorted(models)}")
    if pred_lens:
        filters.append(f"pred_lens={sorted(pred_lens)}")
    filter_note = f" ({', '.join(filters)})" if filters else ""
    print(f"Discovered {len(argv_list)} runnable job(s){filter_note}, skipped {len(skipped)}.")
    return argv_list


def format_argv(argv: List[str]) -> str:
    import shlex

    return shlex.join(argv)


@app.function(gpu="T4", **FUNCTION_KWARGS)
def run_one_job_t4(argv: List[str]) -> int:
    return execute_job(argv)


@app.function(gpu="L4", **FUNCTION_KWARGS)
def run_one_job_l4(argv: List[str]) -> int:
    return execute_job(argv)


@app.function(gpu="A10", **FUNCTION_KWARGS)
def run_one_job_a10(argv: List[str]) -> int:
    return execute_job(argv)


GPU_RUNNERS: Dict[str, modal.Function] = {
    "T4": run_one_job_t4,
    "L4": run_one_job_l4,
    "A10": run_one_job_a10,
}


def _parse_int_csv_list(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


@app.local_entrypoint()
def main(
    parallel: int = 1,
    gpu: str = "T4",
    seed: int = 2024,
    itr: int = 1,
    dry_run: bool = False,
    models: str = "",
    datasets: str = "",
    pred_lens: str = "",
):
    argv_list = discover_job_argv(
        seed=seed,
        itr=itr,
        models=_parse_csv_list(models),
        datasets=_parse_csv_list(datasets),
        pred_lens=_parse_int_csv_list(pred_lens),
    )
    if not argv_list:
        print("Nothing to run.")
        return

    if dry_run:
        for i, argv in enumerate(argv_list, 1):
            print(f"[{i}] {format_argv(argv)}")
        print(f"Dry run: {len(argv_list)} job(s).")
        return

    gpu_key = gpu.upper()
    if gpu_key not in GPU_RUNNERS:
        print(
            f"Unsupported gpu '{gpu}'. Choose from: {', '.join(SUPPORTED_GPUS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    run_fn = GPU_RUNNERS[gpu_key]
    if parallel > 1:
        run_fn.update_autoscaler(max_containers=parallel)

    print(f"Dispatching {len(argv_list)} job(s) on Modal ({gpu_key}, parallel={parallel})...")

    failures = 0
    if parallel <= 1:
        for argv in argv_list:
            rc = run_fn.remote(argv)
            if rc != 0:
                failures += 1
                print(f"WARNING: job failed with exit code {rc}", file=sys.stderr)
    else:
        for result in run_fn.map(
            argv_list,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        ):
            if isinstance(result, Exception):
                failures += 1
                print(f"WARNING: job raised {result!r}", file=sys.stderr)
            elif result != 0:
                failures += 1

    if failures:
        print(f"Done with {failures} failure(s). Re-run to retry (--skip_if_done skips finished jobs).")
    else:
        print("All jobs finished OK.")
    print("Pull artifacts: ./modal_setup.sh download")
