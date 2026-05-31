#!/usr/bin/env python3
"""
Build paper Tables 2, 3, and 4 from ./results/ and bundled MSSD hyperparameter configs.

Table 2: MLP, SOFTS, Time-LLM, Time-Machine × {MSE, MSSD, Tilde-Q}
Table 3: RLinear, PatchTST, TiDE, DLinear, Autoformer × {MSE, MSSD}
Table 4: MSSD hyperparameter settings per dataset and prediction length
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from configs.datasets import DATASETS, PRED_LENS, get_dataset
from configs.experiment_runner import ExperimentJob, SUPPORTED_MODELS, build_run_command, run_jobs
from configs.hyperparams import get_mssd_hyperparams
from utils.results_parser import collect_results

MISSING = "-"

PAPER_DATASETS = [
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "ECL",
    "Exchange",
    "Traffic",
    "Weather",
    "Solar-Energy",
]

PAPER_PRED_LENS = PRED_LENS + ["Avg"]

TABLE2_MODELS = ["MLP", "SOFTS", "Time-LLM", "Time-Machine"]
TABLE2_LOSSES = ["MSE", "MSSD", "Tilde-Q"]

TABLE3_MODELS = ["RLinear", "PatchTST", "TiDE", "DLinear", "Autoformer"]
TABLE3_LOSSES = ["MSE", "MSSD"]

# Map repo model names to paper table model names when they differ.
MODEL_ALIASES = {
    "NLinear": "RLinear",
}

# Internal dataset keys used by configs/hyperparams.py
INTERNAL_DATASET_KEYS = {
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ECL": "electricity",
    "Exchange": "exchange_rate",
    "Traffic": "traffic",
    "Weather": "weather",
    "Solar-Energy": "Solar",
}

LOSS_ALIASES = {
    "mse": "MSE",
    "mssd": "MSSD",
    "tildeq": "Tilde-Q",
}

PAPER_LOSS_TO_REPO = {v: k for k, v in LOSS_ALIASES.items()}

# Paper table model -> repo model (None = not available in this codebase).
PAPER_TO_REPO_MODEL = {
    "MLP": "MLP",
    "SOFTS": "SOFTS",
    "Time-LLM": None,
    "Time-Machine": None,
    "RLinear": "NLinear",
    "PatchTST": None,
    "TiDE": None,
    "DLinear": "DLinear",
    "Autoformer": "Autoformer",
}


def present_result_keys(agg: pd.DataFrame) -> Set[Tuple[str, str, int, str]]:
    if agg.empty:
        return set()
    return {
        (row.paper_model, row.paper_dataset, int(row.pred_len), row.paper_loss)
        for row in agg.itertuples()
    }


def required_table_keys(
    models: Sequence[str],
    train_losses: Sequence[str],
) -> Set[Tuple[str, str, int, str]]:
    keys: Set[Tuple[str, str, int, str]] = set()
    for model in models:
        for dataset in PAPER_DATASETS:
            for pred_len in PRED_LENS:
                for train_loss in train_losses:
                    keys.add((model, dataset, pred_len, train_loss))
    return keys


def missing_result_keys(
    agg: pd.DataFrame,
    models: Sequence[str],
    train_losses: Sequence[str],
) -> Set[Tuple[str, str, int, str]]:
    return required_table_keys(models, train_losses) - present_result_keys(agg)


def check_dataset_files() -> Tuple[List[str], List[str]]:
    """Return (available internal dataset names, missing file paths)."""
    available: List[str] = []
    missing_paths: List[str] = []
    for name, cfg in DATASETS.items():
        path = os.path.join(cfg.root_path, cfg.data_path)
        if os.path.isfile(path):
            available.append(name)
        else:
            missing_paths.append(path)
    return available, missing_paths


def build_experiment_jobs(
    missing_keys: Set[Tuple[str, str, int, str]],
    seed: int,
    itr: int,
) -> Tuple[List[ExperimentJob], List[Dict[str, str]]]:
    jobs: List[ExperimentJob] = []
    skipped: List[Dict[str, str]] = []

    for paper_model, paper_dataset, pred_len, paper_loss in sorted(missing_keys):
        repo_model = PAPER_TO_REPO_MODEL.get(paper_model)
        if repo_model is None:
            skipped.append(
                {
                    "paper_model": paper_model,
                    "dataset": paper_dataset,
                    "pred_len": str(pred_len),
                    "train_loss": paper_loss,
                    "reason": "model not implemented in repository",
                }
            )
            continue
        if repo_model not in SUPPORTED_MODELS:
            skipped.append(
                {
                    "paper_model": paper_model,
                    "dataset": paper_dataset,
                    "pred_len": str(pred_len),
                    "train_loss": paper_loss,
                    "reason": f"repo model '{repo_model}' is not supported",
                }
            )
            continue

        repo_loss = PAPER_LOSS_TO_REPO.get(paper_loss)
        if repo_loss is None:
            skipped.append(
                {
                    "paper_model": paper_model,
                    "dataset": paper_dataset,
                    "pred_len": str(pred_len),
                    "train_loss": paper_loss,
                    "reason": "unknown training loss",
                }
            )
            continue

        internal_dataset = INTERNAL_DATASET_KEYS[paper_dataset]
        dataset_cfg = get_dataset(internal_dataset)
        data_path = os.path.join(dataset_cfg.root_path, dataset_cfg.data_path)
        if not os.path.isfile(data_path):
            skipped.append(
                {
                    "paper_model": paper_model,
                    "dataset": paper_dataset,
                    "pred_len": str(pred_len),
                    "train_loss": paper_loss,
                    "reason": f"missing data file: {data_path}",
                }
            )
            continue

        hyperparams = None
        if repo_loss == "mssd":
            hyperparams = get_mssd_hyperparams(repo_model, internal_dataset, pred_len)

        jobs.append(
            ExperimentJob(
                model=repo_model,
                dataset=dataset_cfg,
                pred_len=pred_len,
                loss=repo_loss,
                seed=seed,
                itr=itr,
                hyperparams=hyperparams,
            )
        )

    return jobs, skipped


def run_missing_experiments(
    agg: pd.DataFrame,
    results_dir: str,
    seed: int,
    itr: int,
    dry_run: bool,
    execute: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    missing2 = missing_result_keys(agg, TABLE2_MODELS, TABLE2_LOSSES)
    missing3 = missing_result_keys(agg, TABLE3_MODELS, TABLE3_LOSSES)
    missing = missing2 | missing3

    if not missing:
        print("All runnable table cells already have results in ./results.")
        return agg, []

    jobs, skipped = build_experiment_jobs(missing, seed=seed, itr=itr)
    # Run simpler / faster models before heavy transformer baselines.
    model_order = {"MLP": 0, "DLinear": 1, "SOFTS": 2, "NLinear": 3, "Autoformer": 4, "iTransformer": 5}
    jobs.sort(key=lambda j: (model_order.get(j.model, 99), j.dataset.name, j.pred_len, j.loss))
    runnable = len(jobs)
    print(
        f"Missing result groups: {len(missing)} "
        f"(runnable: {runnable}, skipped: {len(skipped)})"
    )

    if skipped:
        print("Skipped jobs (will remain '-' in tables):")
        for item in skipped[:10]:
            print(
                f"  {item['paper_model']} | {item['dataset']} | pl={item['pred_len']} "
                f"| {item['train_loss']} -> {item['reason']}"
            )
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if not jobs:
        return agg, skipped

    if dry_run:
        for job in jobs:
            print(build_run_command(job))
        print(f"Dry run: would execute {len(jobs)} experiment job(s).")
        return agg, skipped

    if not execute:
        return agg, skipped

    _, failures = run_jobs(
        jobs, dry_run=False, execute=True, continue_on_error=True, results_dir=results_dir
    )
    if failures:
        print(f"Warning: {len(failures)} experiment(s) failed; continuing with available results.")
    raw = collect_results(results_dir)
    return aggregate_results(raw), skipped


def canonical_dataset(row: pd.Series) -> Optional[str]:
    data = str(row.get("data", ""))
    data_path = str(row.get("data_path", ""))

    if data in {"ETTm1", "ETTm2", "ETTh1", "ETTh2"}:
        return data
    if data == "Solar" or data_path.startswith("solar"):
        return "Solar-Energy"

    path_map = {
        "electricity": "ECL",
        "exchange_rate": "Exchange",
        "traffic": "Traffic",
        "weather": "Weather",
    }
    if data_path in path_map:
        return path_map[data_path]
    if data in path_map:
        return path_map[data]
    return None


def canonical_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def canonical_loss(loss: str) -> str:
    return LOSS_ALIASES.get(str(loss).lower(), str(loss))


def aggregate_results(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["paper_dataset"] = work.apply(canonical_dataset, axis=1)
    work["paper_model"] = work["model"].map(canonical_model)
    work["paper_loss"] = work["loss"].map(canonical_loss)
    work = work.dropna(subset=["paper_dataset"])

    group_cols = ["paper_model", "paper_dataset", "pred_len", "paper_loss"]
    grouped = work.groupby(group_cols, as_index=False)

    if strategy == "best":
        idx = grouped["mse"].idxmin()
        return work.loc[idx].reset_index(drop=True)

    return grouped.agg(
        mse=("mse", "mean"),
        mae=("mae", "mean"),
        n_runs=("mse", "count"),
    ).reset_index()


def lookup_metric(
    agg: pd.DataFrame,
    model: str,
    dataset: str,
    pred_len: int | str,
    train_loss: str,
    metric: str,
) -> str:
    if pred_len == "Avg":
        return MISSING

    if agg.empty:
        return MISSING

    mask = (
        (agg["paper_model"] == model)
        & (agg["paper_dataset"] == dataset)
        & (agg["pred_len"] == int(pred_len))
        & (agg["paper_loss"] == train_loss)
    )
    rows = agg.loc[mask]
    if rows.empty:
        return MISSING
    value = float(rows.iloc[0][metric.lower()])
    return f"{value:.3f}"


def compute_avg_row(
    agg: pd.DataFrame,
    model: str,
    dataset: str,
    train_loss: str,
    metric: str,
) -> str:
    if agg.empty:
        return MISSING

    mask = (
        (agg["paper_model"] == model)
        & (agg["paper_dataset"] == dataset)
        & (agg["paper_loss"] == train_loss)
    )
    rows = agg.loc[mask]
    if rows.empty:
        return MISSING
    value = float(rows[metric.lower()].mean())
    return f"{value:.3f}"


def build_results_table(
    agg: pd.DataFrame,
    models: Sequence[str],
    train_losses: Sequence[str],
    table_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      long_df: one row per cell with parsed numeric values
      wide_df: paper-style layout for Excel / LaTeX export
    """
    long_rows: List[Dict] = []
    wide_rows: List[Dict] = []

    for dataset in PAPER_DATASETS:
        for pred_len in PAPER_PRED_LENS:
            row = {"Table": table_name, "Dataset": dataset, "Pred Length": pred_len}
            for model in models:
                for train_loss in train_losses:
                    for metric in ("MSE", "MAE"):
                        col = f"{model} | {train_loss} | {metric}"
                        if pred_len == "Avg":
                            cell = compute_avg_row(agg, model, dataset, train_loss, metric)
                        else:
                            cell = lookup_metric(
                                agg, model, dataset, pred_len, train_loss, metric
                            )
                        row[col] = cell
                        if cell != MISSING:
                            long_rows.append(
                                {
                                    "table": table_name,
                                    "dataset": dataset,
                                    "pred_len": pred_len,
                                    "model": model,
                                    "train_loss": train_loss,
                                    "metric": metric,
                                    "value": float(cell),
                                }
                            )
            wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    return long_df, wide_df


def add_summary_rows(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    models: Sequence[str],
    train_losses: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    """Append Win Count and 1st Count rows based on eval MSE."""
    if long_df.empty:
        for label in ("Win Count", "1st Count"):
            row = {"Table": table_name, "Dataset": label, "Pred Length": ""}
            wide_df = pd.concat([wide_df, pd.DataFrame([row])], ignore_index=True)
        return wide_df

    eval_rows = long_df[
        (long_df["metric"] == "MSE") & (long_df["pred_len"] != "Avg")
    ].copy()
    if eval_rows.empty:
        return wide_df

    win_counts: Dict[str, int] = {}
    first_counts: Dict[str, int] = {}
    for model in models:
        for train_loss in train_losses:
            col = f"{model} | {train_loss} | MSE"
            win_counts[col] = 0
            first_counts[col] = 0

    for (dataset, pred_len), group in eval_rows.groupby(["dataset", "pred_len"]):
        best = group["value"].min()
        winners = group[group["value"] == best]
        for _, winner in winners.iterrows():
            col = f"{winner['model']} | {winner['train_loss']} | MSE"
            win_counts[col] = win_counts.get(col, 0) + 1
            if len(winners) == 1:
                first_counts[col] = first_counts.get(col, 0) + 1

    for label, counts in (("Win Count", win_counts), ("1st Count", first_counts)):
        row = {"Table": table_name, "Dataset": label, "Pred Length": ""}
        for model in models:
            for train_loss in train_losses:
                col = f"{model} | {train_loss} | MSE"
                row[col] = counts.get(col, 0)
                row[f"{model} | {train_loss} | MAE"] = ""
        wide_df = pd.concat([wide_df, pd.DataFrame([row])], ignore_index=True)

    return wide_df


def build_table4(model: str = "MLP") -> pd.DataFrame:
    rows = []
    for paper_dataset in PAPER_DATASETS:
        internal = INTERNAL_DATASET_KEYS[paper_dataset]
        for pred_len in PRED_LENS:
            hp = get_mssd_hyperparams(model, internal, pred_len)
            rows.append(
                {
                    "Dataset": paper_dataset,
                    "Pred Length": pred_len,
                    "Max Levels": hp.max_levels,
                    "Spline Degree": 3,
                    "Alpha": hp.alpha,
                    "Beta": hp.beta,
                    "Gamma": hp.gamma,
                    "Knot Scaling Factor": hp.knot_multiplier,
                    "Spline Criterion Exponent": hp.spline_criterion_exponent,
                    "Control Point Weight Window": 4,
                    "Source Model": model,
                }
            )
    return pd.DataFrame(rows)


def coverage_report(
    wide_df: pd.DataFrame,
    models: Sequence[str],
    train_losses: Sequence[str],
    table_name: str,
) -> pd.DataFrame:
    data_rows = wide_df[
        (wide_df["Dataset"] != "Win Count")
        & (wide_df["Dataset"] != "1st Count")
        & (wide_df["Pred Length"] != "Avg")
    ]
    total_cells = len(PAPER_DATASETS) * len(PRED_LENS) * len(models) * len(train_losses) * 2
    filled = 0
    rows = []
    for model in models:
        for train_loss in train_losses:
            cols = [f"{model} | {train_loss} | MSE", f"{model} | {train_loss} | MAE"]
            present = 0
            possible = len(PAPER_DATASETS) * len(PRED_LENS) * 2
            for col in cols:
                if col not in data_rows.columns:
                    continue
                present += (data_rows[col] != MISSING).sum()
            filled += present
            rows.append(
                {
                    "Table": table_name,
                    "Model": model,
                    "Train Loss": train_loss,
                    "Filled Cells": present,
                    "Possible Cells": possible,
                    "Coverage %": round(100.0 * present / possible, 1) if possible else 0.0,
                }
            )
    summary = pd.DataFrame(rows)
    print(
        f"{table_name}: filled {filled}/{total_cells} metric cells "
        f"({round(100.0 * filled / total_cells, 1) if total_cells else 0.0}%)"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Fill paper Tables 2, 3, and 4")
    parser.add_argument("--results_dir", default="./results", help="Directory with experiment outputs")
    parser.add_argument(
        "--output",
        default="./tables/filled_tables.xlsx",
        help="Output Excel workbook path",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "best"],
        default="mean",
        help="How to combine duplicate runs (seeds or hyperparam sweeps)",
    )
    parser.add_argument(
        "--table4_model",
        default="MLP",
        help="Model whose tuned hyperparameters populate Table 4",
    )
    parser.add_argument(
        "--run_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run experiments for table cells missing from ./results (default: on)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print missing experiment commands without executing them",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for auto-run jobs")
    parser.add_argument(
        "--itr",
        type=int,
        default=1,
        help="Number of repeated runs per auto-run job (seed increments each itr)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    available, missing_data = check_dataset_files()
    if missing_data:
        print("Missing dataset files:")
        for path in missing_data:
            print(f"  {path}")
        if args.run_missing and not args.dry_run:
            print("Auto-run will skip jobs whose data files are absent.")
    else:
        print(f"All {len(available)} benchmark dataset files found under ./data/")

    raw = collect_results(args.results_dir)
    agg = aggregate_results(raw, strategy=args.aggregate)
    skipped_jobs: List[Dict[str, str]] = []

    if args.run_missing:
        agg, skipped_jobs = run_missing_experiments(
            agg,
            results_dir=args.results_dir,
            seed=args.seed,
            itr=args.itr,
            dry_run=args.dry_run,
            execute=not args.dry_run,
        )
        if not args.dry_run:
            raw = collect_results(args.results_dir)
            agg = aggregate_results(raw, strategy=args.aggregate)

    table2_long, table2_wide = build_results_table(
        agg, TABLE2_MODELS, TABLE2_LOSSES, "Table 2"
    )
    table2_wide = add_summary_rows(
        table2_wide, table2_long, TABLE2_MODELS, TABLE2_LOSSES, "Table 2"
    )

    table3_long, table3_wide = build_results_table(
        agg, TABLE3_MODELS, TABLE3_LOSSES, "Table 3"
    )
    table3_wide = add_summary_rows(
        table3_wide, table3_long, TABLE3_MODELS, TABLE3_LOSSES, "Table 3"
    )

    table4 = build_table4(model=args.table4_model)

    coverage = pd.concat(
        [
            coverage_report(table2_wide, TABLE2_MODELS, TABLE2_LOSSES, "Table 2"),
            coverage_report(table3_wide, TABLE3_MODELS, TABLE3_LOSSES, "Table 3"),
        ],
        ignore_index=True,
    )

    raw_summary = pd.DataFrame(
        {
            "metric": [
                "raw_result_rows",
                "aggregated_rows",
                "results_dir",
                "auto_run_missing",
                "skipped_jobs",
            ],
            "value": [
                len(raw),
                len(agg),
                os.path.abspath(args.results_dir),
                args.run_missing and not args.dry_run,
                len(skipped_jobs),
            ],
        }
    )
    skipped_df = pd.DataFrame(skipped_jobs)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        table2_wide.to_excel(writer, sheet_name="Table2", index=False)
        table3_wide.to_excel(writer, sheet_name="Table3", index=False)
        table4.to_excel(writer, sheet_name="Table4", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        if not skipped_df.empty:
            skipped_df.to_excel(writer, sheet_name="SkippedJobs", index=False)
        if not raw.empty:
            raw.to_excel(writer, sheet_name="RawResults", index=False)
        if not agg.empty:
            agg.to_excel(writer, sheet_name="AggregatedResults", index=False)
        raw_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Wrote {args.output}")
    print("Sheets: Table2, Table3, Table4, Coverage, SkippedJobs, RawResults, AggregatedResults, Summary")
    if raw.empty and not args.run_missing:
        print("Note: no results found — re-run with --run_missing (default) to train missing cells.")
    elif raw.empty and args.dry_run:
        print("Dry run complete. Re-run without --dry_run to execute experiments and fill tables.")
    print("Table 4 is populated from bundled MSSD hyperparameter configs.")


if __name__ == "__main__":
    main()
