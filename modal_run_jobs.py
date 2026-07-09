#!/usr/bin/env python3
"""Dispatch jobs.json (built by make_job_list.py) to Modal GPU workers.

Examples:
  modal run modal_run_jobs.py --dry-run
  modal run modal_run_jobs.py --phases main --parallel 10
  modal run --detach modal_run_jobs.py --phases main,arch,ablation,default,seeds --parallel 10
  modal run modal_run_jobs.py --benchmark          # loss-overhead timing only
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List

import modal

VOLUME_NAME = os.environ.get("HNMD_MODAL_VOLUME", "hnmd-workspace")
WORKSPACE = "/workspace"
APP_ROOT = "/root/app"
ARTIFACT_DIRS = (
    "data",
    "results",
    "results_ablation",
    "results_default",
    "results_seeds",
    "checkpoints",
    "checkpoints_ablation",
    "checkpoints_default",
    "checkpoints_seeds",
    "test_results",
    "tables",
)

app = modal.App("hnmd-paper-jobs")
workspace = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=APP_ROOT,
        ignore=[
            "env", ".venv", "venv", ".git", "__pycache__", ".DS_Store",
            "checkpoints", "results", "test_results", "data",
            "results_ablation", "results_default", "results_seeds",
            "checkpoints_ablation", "checkpoints_default", "checkpoints_seeds",
        ],
    )
)

FUNCTION_KWARGS = dict(
    image=image,
    volumes={WORKSPACE: workspace},
    timeout=60 * 60 * 12,
    retries=1,
    cpu=4,
    memory=16384,
)


def link_workspace() -> None:
    for name in ARTIFACT_DIRS:
        src = os.path.join(WORKSPACE, name)
        dst = os.path.join(APP_ROOT, name)
        os.makedirs(src, exist_ok=True)
        if os.path.islink(dst):
            continue
        if os.path.isdir(dst):
            try:
                os.rmdir(dst)
            except OSError:
                pass
        if not os.path.lexists(dst):
            os.symlink(src, dst)


def normalize_argv(argv: List[str]) -> List[str]:
    patched = list(argv)
    patched[0] = sys.executable
    for i, arg in enumerate(patched):
        if arg == "run.py":
            patched[i] = os.path.join(APP_ROOT, "run.py")
    return patched


def execute(job: Dict) -> Dict:
    import time

    os.chdir(APP_ROOT)
    link_workspace()
    argv = normalize_argv(job["argv"])
    print(f"[{job['tag']}] running:", " ".join(argv), flush=True)
    start = time.time()
    result = subprocess.run(argv, cwd=APP_ROOT)
    elapsed = time.time() - start
    workspace.commit()
    print(f"[{job['tag']}] rc={result.returncode} elapsed={elapsed/60:.1f} min", flush=True)
    return {"tag": job["tag"], "phase": job["phase"], "rc": result.returncode, "minutes": round(elapsed / 60, 1)}


@app.function(gpu="L4", **FUNCTION_KWARGS)
def run_job_l4(job: Dict) -> Dict:
    return execute(job)


@app.function(gpu="A100-40GB", **FUNCTION_KWARGS)
def run_job_a100(job: Dict) -> Dict:
    return execute(job)


@app.function(gpu="L4", **FUNCTION_KWARGS)
def benchmark_loss_overhead() -> str:
    """Median wall-clock per optimizer step for MLP under each loss."""
    import time

    import numpy as np
    import torch

    os.chdir(APP_ROOT)
    link_workspace()
    sys.path.insert(0, APP_ROOT)

    from types import SimpleNamespace

    from data_provider.data_factory import data_provider
    from model.MLP import Model as MLP
    from utils.mssd import MSSD
    from utils.tildeq import tildeq_loss
    from configs.hyperparams import get_mssd_hyperparams

    device = "cuda"
    report = {}
    cases = [
        ("ETTh1", "./data/ETT-small/", "ETTh1.csv", "ETTh1", "h", 7, 96),
        ("ETTh1", "./data/ETT-small/", "ETTh1.csv", "ETTh1", "h", 7, 336),
        ("ETTh1", "./data/ETT-small/", "ETTh1.csv", "ETTh1", "h", 7, 720),
        ("weather", "./data/weather/", "weather.csv", "custom", "h", 21, 96),
        ("electricity", "./data/electricity/", "electricity.csv", "custom", "h", 321, 96),
    ]
    for ds_name, root, path, data, freq, enc_in, pred_len in cases:
        args = SimpleNamespace(
            data=data, root_path=root, data_path=path, features="M", target="OT",
            freq=freq, seq_len=96, label_len=48, pred_len=pred_len,
            embed="timeF", batch_size=32 if enc_in < 100 else 16, num_workers=2,
            enc_in=enc_in, dec_in=enc_in, c_out=enc_in, d_model=256, d_ff=512,
            channel_independence=False,
        )
        _, loader = data_provider(args, "train")
        batch = next(iter(loader))
        bx, by = batch[0].float().to(device), batch[1].float().to(device)
        by = by[:, -pred_len:, :]

        hp = get_mssd_hyperparams("MLP", ds_name, pred_len)
        for loss_name in ["mse", "tildeq", "mssd"]:
            model = MLP(args).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            if loss_name == "mse":
                crit = torch.nn.MSELoss()
            elif loss_name == "tildeq":
                crit = lambda a, b: tildeq_loss(a, b)
            else:
                crit = MSSD(
                    num_variables=enc_in, sequence_length=pred_len,
                    alpha=hp.alpha, beta=hp.beta, gamma=hp.gamma,
                    knot_scaling_factor=hp.knot_multiplier,
                    exponent=hp.spline_criterion_exponent, max_levels=hp.max_levels,
                )
                crit.norm_params = {
                    "max_value": float(by.max()), "min_value": float(by.min()),
                    "max_slope": 1.0, "min_slope": -1.0,
                    "max_fft": float(torch.abs(torch.fft.rfft(by, dim=1)).max()),
                    "min_fft": 0.0, "max_diff_2": 1.0, "min_diff_2": -1.0,
                }
            times = []
            for it in range(60):
                torch.cuda.synchronize()
                t0 = time.time()
                optim.zero_grad()
                out = model(bx)
                loss = crit(out, by)
                loss.backward()
                optim.step()
                torch.cuda.synchronize()
                if it >= 10:
                    times.append(time.time() - t0)
            report[f"{ds_name}_pl{pred_len}_{loss_name}"] = float(np.median(times))
            print(f"{ds_name} pl={pred_len} {loss_name}: {np.median(times)*1000:.2f} ms/step", flush=True)

    payload = json.dumps(report, indent=1)
    out_dir = os.path.join(WORKSPACE, "tables")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "loss_overhead_benchmark.json"), "w") as fh:
        fh.write(payload)
    workspace.commit()
    return payload


@app.local_entrypoint()
def main(
    jobs_file: str = "jobs.json",
    phases: str = "",
    parallel: int = 10,
    gpu: str = "L4",
    dry_run: bool = False,
    benchmark: bool = False,
    tags: str = "",
):
    if benchmark:
        print(benchmark_loss_overhead.remote())
        return

    with open(jobs_file) as fh:
        jobs = json.load(fh)

    if phases:
        wanted = {p.strip() for p in phases.split(",") if p.strip()}
        jobs = [j for j in jobs if j["phase"] in wanted]
    if tags:
        wanted_tags = {t.strip() for t in tags.split(",") if t.strip()}
        jobs = [j for j in jobs if j["tag"] in wanted_tags]

    if not jobs:
        print("No jobs matched.")
        return

    if dry_run:
        for i, j in enumerate(jobs, 1):
            print(f"[{i}] ({j['phase']}) {j['tag']}: {' '.join(j['argv'])}")
        print(f"Dry run: {len(jobs)} job(s).")
        return

    runner = run_job_a100 if gpu.upper().startswith("A100") else run_job_l4
    if parallel > 1:
        runner.update_autoscaler(max_containers=parallel)

    print(f"Dispatching {len(jobs)} job(s) on {gpu} (parallel={parallel})...")
    failures: List[Dict] = []
    done = 0
    for result in runner.map(jobs, return_exceptions=True, wrap_returned_exceptions=False):
        done += 1
        if isinstance(result, Exception):
            failures.append({"error": repr(result)})
            print(f"({done}/{len(jobs)}) EXCEPTION: {result!r}", flush=True)
        else:
            status = "OK" if result["rc"] == 0 else f"FAIL rc={result['rc']}"
            if result["rc"] != 0:
                failures.append(result)
            print(f"({done}/{len(jobs)}) {result['tag']} {status} {result['minutes']} min", flush=True)

    print(f"Finished: {len(jobs) - len(failures)} OK, {len(failures)} failed.")
    for f in failures:
        print("FAILED:", f)
