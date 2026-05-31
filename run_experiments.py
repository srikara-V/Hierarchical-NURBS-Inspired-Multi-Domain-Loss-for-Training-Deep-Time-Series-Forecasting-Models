#!/usr/bin/env python3
"""Unified entry point for HNMD benchmark experiments."""

from __future__ import annotations

import argparse

from configs.experiment_runner import SUPPORTED_MODELS, all_dataset_names, run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run HNMD / baseline forecasting experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["MLP"],
        help=f"Models to train. Supported: {sorted(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names or 'all' for the full 9-dataset benchmark",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["mssd"],
        choices=["mse", "mssd", "tildeq"],
        help="Training losses to evaluate",
    )
    parser.add_argument(
        "--pred_lens",
        nargs="+",
        type=int,
        default=[96, 192, 336, 720],
        help="Forecast horizons",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2024],
        help="Base random seeds; each seed runs as one job",
    )
    parser.add_argument(
        "--itr",
        type=int,
        default=1,
        help="Number of repeated runs inside run.py per job (uses seed, seed+1, ...)",
    )
    parser.add_argument(
        "--mssd_mode",
        choices=["auto", "tuned", "grid"],
        default="auto",
        help=(
            "MSSD hyperparameter mode: tuned (Excel/defaults), grid (full knot/exponent sweep), "
            "auto (tuned unless --mssd_mode grid)"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = all_dataset_names() if args.datasets == ["all"] else args.datasets
    mssd_mode = "tuned" if args.mssd_mode == "auto" else args.mssd_mode

    run_experiments(
        models=args.models,
        datasets=datasets,
        losses=args.losses,
        pred_lens=args.pred_lens,
        seeds=args.seeds,
        itr=args.itr,
        mssd_mode=mssd_mode,
        dry_run=args.dry_run,
        execute=not args.dry_run,
    )


if __name__ == "__main__":
    main()
