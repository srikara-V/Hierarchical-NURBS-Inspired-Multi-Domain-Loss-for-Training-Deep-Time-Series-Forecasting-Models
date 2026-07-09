#!/usr/bin/env python3
"""Generate jobs.json for the preprint experiment phases (see EXPERIMENT_PLAN.md).

Each job is {"phase": str, "tag": str, "argv": [...]} where argv[0] == "PYTHON"
and the run.py path is relative; modal_run_jobs.py normalizes both inside the
container. Existing volume results (volume_inventory.json) are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.datasets import PRED_LENS, get_dataset
from configs.experiment_runner import ExperimentJob, build_run_argv
from configs.hyperparams import MSSDHyperparams, get_mssd_hyperparams
from utils.results_parser import parse_setting_name

ALL_DATASETS = [
    "ETTm1", "ETTm2", "ETTh1", "ETTh2",
    "electricity", "exchange_rate", "traffic", "weather", "Solar",
]
ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
LOSSES = ["mse", "tildeq", "mssd"]

# Rough cost ordering so heavy jobs dispatch first (minimizes makespan).
DATASET_WEIGHT = {
    "traffic": 100, "electricity": 60, "Solar": 30, "weather": 12,
    "ETTm1": 8, "ETTm2": 8, "exchange_rate": 3, "ETTh1": 2, "ETTh2": 2,
}

DEFAULT_HP = MSSDHyperparams(
    alpha=0.01, beta=1.0, gamma=1.0, knot_multiplier=5.0,
    spline_criterion_exponent=1, max_levels=5,
)

ABLATION_CELLS: List[Tuple[str, int]] = [
    ("ETTh1", 96), ("ETTh1", 336),
    ("ETTm1", 96), ("ETTm1", 336),
    ("ETTm2", 96), ("ETTm2", 336),
]

ABLATION_VARIANTS = [
    "noalpha", "nobeta", "nogamma", "timeonly", "levels1", "levels3", "xp0", "nodecomp",
]


def variant_hp(base: MSSDHyperparams, variant: str) -> MSSDHyperparams:
    if variant == "noalpha":
        return replace(base, alpha=0.0)
    if variant == "nobeta":
        return replace(base, beta=0.0)
    if variant == "nogamma":
        return replace(base, gamma=0.0)
    if variant == "timeonly":
        return replace(base, beta=0.0, gamma=0.0)
    if variant == "levels1":
        return replace(base, max_levels=1)
    if variant == "levels3":
        return replace(base, max_levels=3)
    if variant == "xp0":
        return replace(base, spline_criterion_exponent=0)
    if variant == "nodecomp":
        return replace(base, max_levels=0)
    raise ValueError(variant)


def load_present_cells(inventory_path: str) -> Set[Tuple[str, str, int, str]]:
    """(model, canonical_dataset, pred_len, loss) present in ./results on the volume."""
    present: Set[Tuple[str, str, int, str]] = set()
    if not os.path.isfile(inventory_path):
        return present
    inv = json.load(open(inventory_path))
    for name in inv.get("results", {}):
        p = parse_setting_name(name)
        if p is None:
            continue
        data, dp = p["data"], p["data_path"]
        if data in {"ETTm1", "ETTm2", "ETTh1", "ETTh2"}:
            ds = data
        elif data == "Solar" or dp.startswith("solar"):
            ds = "Solar"
        else:
            ds = dp
        present.add((p["model"], ds, int(p["pred_len"]), p["loss"].lower()))
    return present


def mk_argv(
    job: ExperimentJob,
    results_dir: str = "./results/",
    checkpoints_dir: Optional[str] = None,
    num_workers: int = 4,
) -> List[str]:
    argv = build_run_argv(job, results_dir=results_dir)
    argv[0] = "PYTHON"
    argv.extend(["--num_workers", str(num_workers)])
    if checkpoints_dir is not None:
        argv.extend(["--checkpoints", checkpoints_dir])
    return argv


def job_entry(phase: str, tag: str, argv: List[str], weight: int) -> Dict:
    return {"phase": phase, "tag": tag, "argv": argv, "weight": weight}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="volume_inventory.json")
    parser.add_argument("--output", default="jobs.json")
    args = parser.parse_args()

    present = load_present_cells(args.inventory)
    jobs: List[Dict] = []

    # Phase 1: complete the main MLP grid.
    for ds_name in ALL_DATASETS:
        ds = get_dataset(ds_name)
        for pl in PRED_LENS:
            for loss in LOSSES:
                if ("MLP", ds_name, pl, loss) in present:
                    continue
                hp = get_mssd_hyperparams("MLP", ds_name, pl) if loss == "mssd" else None
                job = ExperimentJob(model="MLP", dataset=ds, pred_len=pl, loss=loss,
                                    seed=2024, itr=1, hyperparams=hp)
                jobs.append(job_entry(
                    "main", f"MLP_{ds_name}_{pl}_{loss}", mk_argv(job),
                    DATASET_WEIGHT[ds_name] * (pl // 96),
                ))

    # Phase 2: architecture generality on ETT.
    for model in ["DLinear", "iTransformer"]:
        for ds_name in ETT:
            ds = get_dataset(ds_name)
            for pl in PRED_LENS:
                for loss in LOSSES:
                    if (model, ds_name, pl, loss) in present:
                        continue
                    hp = get_mssd_hyperparams(model, ds_name, pl) if loss == "mssd" else None
                    job = ExperimentJob(model=model, dataset=ds, pred_len=pl, loss=loss,
                                        seed=2024, itr=1, hyperparams=hp)
                    jobs.append(job_entry(
                        "arch", f"{model}_{ds_name}_{pl}_{loss}", mk_argv(job),
                        DATASET_WEIGHT[ds_name] * (pl // 96),
                    ))

    # Phase 3: ablations (separate results/checkpoints roots per variant).
    for ds_name, pl in ABLATION_CELLS:
        ds = get_dataset(ds_name)
        base = get_mssd_hyperparams("MLP", ds_name, pl)
        for variant in ABLATION_VARIANTS:
            hp = variant_hp(base, variant)
            job = ExperimentJob(model="MLP", dataset=ds, pred_len=pl, loss="mssd",
                                seed=2024, itr=1, hyperparams=hp)
            jobs.append(job_entry(
                "ablation", f"abl_{variant}_{ds_name}_{pl}",
                mk_argv(job,
                        results_dir=f"./results_ablation/{variant}/",
                        checkpoints_dir=f"./checkpoints_ablation/{variant}/"),
                DATASET_WEIGHT[ds_name] * (pl // 96),
            ))

    # Phase 4: fixed default HNMD config across the full grid.
    for ds_name in ALL_DATASETS:
        ds = get_dataset(ds_name)
        for pl in PRED_LENS:
            job = ExperimentJob(model="MLP", dataset=ds, pred_len=pl, loss="mssd",
                                seed=2024, itr=1, hyperparams=DEFAULT_HP)
            jobs.append(job_entry(
                "default", f"default_MLP_{ds_name}_{pl}",
                mk_argv(job,
                        results_dir="./results_default/",
                        checkpoints_dir="./checkpoints_default/"),
                DATASET_WEIGHT[ds_name] * (pl // 96),
            ))

    # Phase 5: seed robustness.
    for seed in [2025, 2026]:
        for ds_name in ["ETTh1", "ETTm1"]:
            ds = get_dataset(ds_name)
            for pl in PRED_LENS:
                for loss in LOSSES:
                    hp = get_mssd_hyperparams("MLP", ds_name, pl) if loss == "mssd" else None
                    job = ExperimentJob(model="MLP", dataset=ds, pred_len=pl, loss=loss,
                                        seed=seed, itr=1, hyperparams=hp)
                    jobs.append(job_entry(
                        "seeds", f"seed{seed}_MLP_{ds_name}_{pl}_{loss}",
                        mk_argv(job,
                                results_dir=f"./results_seeds/s{seed}/",
                                checkpoints_dir=f"./checkpoints_seeds/s{seed}/"),
                        DATASET_WEIGHT[ds_name] * (pl // 96),
                    ))

    jobs.sort(key=lambda j: -j["weight"])
    with open(args.output, "w") as fh:
        json.dump(jobs, fh, indent=1)

    by_phase: Dict[str, int] = {}
    for j in jobs:
        by_phase[j["phase"]] = by_phase.get(j["phase"], 0) + 1
    print(f"wrote {args.output}: {len(jobs)} jobs -> {by_phase}")


if __name__ == "__main__":
    main()
