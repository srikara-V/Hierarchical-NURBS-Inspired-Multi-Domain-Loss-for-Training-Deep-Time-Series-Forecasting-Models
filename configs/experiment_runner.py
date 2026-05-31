"""Build and launch training commands for benchmark experiments."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from configs.datasets import (
    DATASETS,
    DatasetConfig,
    PRED_LENS,
    get_dataset,
    list_datasets,
)
from configs.hyperparams import MSSDHyperparams, get_mssd_hyperparams, iter_mssd_grid_search


SUPPORTED_MODELS = {
    "MLP",
    "DLinear",
    "SOFTS",
    "iTransformer",
    "iInformer",
    "iReformer",
    "iFlowformer",
    "iFlashformer",
    "Autoformer",
    "NLinear",
}

MODEL_DEFAULTS = {
    "MLP": {"d_model": 256, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "DLinear": {"d_model": 256, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "SOFTS": {"d_model": 256, "d_ff": 512, "d_core": 256, "e_layers": 2, "n_heads": 8},
    "iTransformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "iInformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "iReformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "iFlowformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "iFlashformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "n_heads": 8},
    "Autoformer": {"d_model": 512, "d_ff": 512, "e_layers": 2, "d_layers": 1, "n_heads": 8, "embed_type": 0},
    "NLinear": {"d_model": 256, "d_ff": 512, "e_layers": 2, "n_heads": 8, "individual": False},
}


@dataclass(frozen=True)
class ExperimentJob:
    model: str
    dataset: DatasetConfig
    pred_len: int
    loss: str
    seed: int
    itr: int
    hyperparams: Optional[MSSDHyperparams] = None
    knot_multiplier: Optional[float] = None
    spline_criterion_exponent: Optional[int] = None


def _python_executable() -> str:
    return sys.executable or "python3"


def build_run_argv(job: ExperimentJob, results_dir: str = "./results/") -> List[str]:
    model = job.model
    dataset = job.dataset
    pred_len = job.pred_len
    loss = job.loss.lower()
    model_cfg = MODEL_DEFAULTS.get(model, MODEL_DEFAULTS["MLP"])

    cmd_parts = [
        _python_executable(),
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--root_path",
        dataset.root_path,
        "--data_path",
        dataset.data_path,
        "--model_id",
        f"{dataset.name}_{pred_len}_{model}",
        "--model",
        model,
        "--data",
        dataset.data,
        "--features",
        "M",
        "--freq",
        dataset.freq,
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        str(pred_len),
        "--e_layers",
        str(model_cfg["e_layers"]),
        "--enc_in",
        str(dataset.enc_in),
        "--dec_in",
        str(dataset.enc_in),
        "--c_out",
        str(dataset.enc_in),
        "--d_model",
        str(model_cfg["d_model"]),
        "--d_ff",
        str(model_cfg["d_ff"]),
        "--n_heads",
        str(model_cfg["n_heads"]),
        "--batch_size",
        str(dataset.batch_size),
        "--learning_rate",
        str(dataset.learning_rate),
        "--train_epochs",
        str(dataset.train_epochs),
        "--patience",
        str(dataset.patience),
        "--des",
        "Exp",
        "--itr",
        str(job.itr),
        "--seed",
        str(job.seed),
        "--loss",
        loss,
    ]

    if "d_layers" in model_cfg:
        cmd_parts.extend(["--d_layers", str(model_cfg["d_layers"])])
    if "d_core" in model_cfg:
        cmd_parts.extend(["--d_core", str(model_cfg["d_core"])])
    if "embed_type" in model_cfg:
        cmd_parts.extend(["--embed_type", str(model_cfg["embed_type"])])
    if model_cfg.get("individual"):
        cmd_parts.append("--individual")

    if loss == "mssd":
        if job.hyperparams is not None:
            hp = job.hyperparams
            cmd_parts.extend(
                [
                    "--knot_multiplier",
                    str(hp.knot_multiplier),
                    "--spline_criterion_exponent",
                    str(hp.spline_criterion_exponent),
                    "--alpha",
                    str(hp.alpha),
                    "--beta",
                    str(hp.beta),
                    "--gamma",
                    str(hp.gamma),
                    "--max_levels",
                    str(hp.max_levels),
                ]
            )
        else:
            cmd_parts.extend(
                [
                    "--knot_multiplier",
                    str(job.knot_multiplier),
                    "--spline_criterion_exponent",
                    str(job.spline_criterion_exponent),
                ]
            )
    elif loss == "tildeq":
        cmd_parts.extend(["--knot_multiplier", "0", "--spline_criterion_exponent", "0"])

    cmd_parts.extend(
        [
            "--results_dir",
            results_dir,
            "--skip_if_done",
            "--resume",
            "--recover",
        ]
    )

    return cmd_parts


def build_run_command(job: ExperimentJob, dry_run: bool = False) -> str:
    return shlex.join(build_run_argv(job))


def iter_jobs(
    models: Sequence[str],
    datasets: Sequence[str],
    losses: Sequence[str],
    pred_lens: Sequence[int],
    seeds: Sequence[int],
    itr: int = 1,
    mssd_mode: str = "tuned",
) -> Iterator[ExperimentJob]:
    for model in models:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model}'. Supported: {sorted(SUPPORTED_MODELS)}")

        for dataset in list_datasets(list(datasets)):
            for pred_len in pred_lens:
                for loss in losses:
                    loss = loss.lower()
                    if loss == "mssd":
                        if mssd_mode == "grid":
                            for knot_multiplier, exponent in iter_mssd_grid_search():
                                for seed in seeds:
                                    yield ExperimentJob(
                                        model=model,
                                        dataset=dataset,
                                        pred_len=pred_len,
                                        loss=loss,
                                        seed=seed,
                                        itr=itr,
                                        knot_multiplier=knot_multiplier,
                                        spline_criterion_exponent=exponent,
                                    )
                        else:
                            hp = get_mssd_hyperparams(model, dataset.name, pred_len)
                            for seed in seeds:
                                yield ExperimentJob(
                                    model=model,
                                    dataset=dataset,
                                    pred_len=pred_len,
                                    loss=loss,
                                    seed=seed,
                                    itr=itr,
                                    hyperparams=hp,
                                )
                    else:
                        for seed in seeds:
                            yield ExperimentJob(
                                model=model,
                                dataset=dataset,
                                pred_len=pred_len,
                                loss=loss,
                                seed=seed,
                                itr=itr,
                            )


def run_jobs(
    jobs: Iterable[ExperimentJob],
    dry_run: bool = False,
    execute: bool = True,
    continue_on_error: bool = False,
    results_dir: str = "./results/",
) -> Tuple[List[str], List[str]]:
    commands: List[str] = []
    failures: List[str] = []
    total = 0
    for job in jobs:
        total += 1
        argv = build_run_argv(job, results_dir=results_dir)
        command = shlex.join(argv)
        commands.append(command)
        print(f"[{total}] {command}")
        if execute and not dry_run:
            try:
                result = subprocess.run(argv)
            except KeyboardInterrupt:
                print("\nJob queue interrupted. Re-run the same command to resume remaining jobs.")
                break
            if result.returncode != 0:
                failures.append(command)
                print(f"WARNING: job failed with exit code {result.returncode}")
                if not continue_on_error:
                    raise RuntimeError(
                        f"Command failed with exit code {result.returncode}: {command}"
                    )
            else:
                print("OK")
    print(f"\nPrepared {len(commands)} experiment command(s).")
    if failures:
        print(f"{len(failures)} command(s) failed.")
    return commands, failures


def run_experiments(
    models: Sequence[str],
    datasets: Sequence[str],
    losses: Sequence[str],
    pred_lens: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    itr: int = 1,
    mssd_mode: str = "auto",
    dry_run: bool = False,
    execute: bool = True,
) -> List[str]:
    if pred_lens is None:
        pred_lens = PRED_LENS
    if seeds is None:
        seeds = [2024]

    if mssd_mode == "auto":
        # Grid search only when explicitly requested per dataset via mssd_mode='grid'.
        mssd_mode = "tuned"

    commands: List[str] = []
    jobs = list(
        iter_jobs(models, datasets, losses, pred_lens, seeds, itr=itr, mssd_mode=mssd_mode)
    )
    return run_jobs(
        jobs, dry_run=dry_run, execute=execute and not dry_run, continue_on_error=False
    )[0]


def all_dataset_names() -> List[str]:
    return list(DATASETS.keys())
