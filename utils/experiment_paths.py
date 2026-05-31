"""Shared experiment folder names and artifact paths."""

from __future__ import annotations

import os
from typing import Any


def build_setting_name(args: Any, ii: int) -> str:
    data_path_tag = os.path.splitext(os.path.basename(args.data_path))[0]
    return (
        "{model_id}_{model}_{data}_{data_path}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}"
        "_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_dt{distil}"
        "_{des}_{class_strategy}_{ii}_ls{loss}_kt{knot_multiplier}_xp{spline_criterion_exponent}"
        "_a{alpha}_b{beta}_g{gamma}"
    ).format(
        model_id=args.model_id,
        model=args.model,
        data=args.data,
        data_path=data_path_tag,
        features=args.features,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        embed=args.embed,
        distil=args.distil,
        des=args.des,
        class_strategy=args.class_strategy,
        ii=ii,
        loss=args.loss,
        knot_multiplier=args.knot_multiplier,
        spline_criterion_exponent=args.spline_criterion_exponent,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )


def metrics_path(setting: str, results_dir: str = "./results") -> str:
    return os.path.join(results_dir, setting, "metrics.npy")


def checkpoint_dir(setting: str, checkpoints_dir: str = "./checkpoints") -> str:
    return os.path.join(checkpoints_dir, setting)


def checkpoint_path(setting: str, checkpoints_dir: str = "./checkpoints") -> str:
    return os.path.join(checkpoint_dir(setting, checkpoints_dir), "checkpoint.pth")


def training_state_path(setting: str, checkpoints_dir: str = "./checkpoints") -> str:
    return os.path.join(checkpoint_dir(setting, checkpoints_dir), "training_state.pth")


def metrics_exist(setting: str, results_dir: str = "./results") -> bool:
    return os.path.isfile(metrics_path(setting, results_dir))


def checkpoint_exists(setting: str, checkpoints_dir: str = "./checkpoints") -> bool:
    return os.path.isfile(checkpoint_path(setting, checkpoints_dir))


def training_state_exists(setting: str, checkpoints_dir: str = "./checkpoints") -> bool:
    return os.path.isfile(training_state_path(setting, checkpoints_dir))
