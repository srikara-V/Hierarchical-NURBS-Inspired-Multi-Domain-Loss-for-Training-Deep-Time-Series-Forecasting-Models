"""Parse experiment result folders produced by run.py."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

KNOWN_MODELS = {
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
    "Transformer",
    "Informer",
    "Reformer",
    "Flowformer",
    "Flashformer",
    "Linear",
}

_SETTING_SUFFIX_RE = re.compile(
    r"_ft(?P<features>[^_]+)"
    r"_sl(?P<seq_len>\d+)"
    r"_ll(?P<label_len>\d+)"
    r"_pl(?P<pred_len>\d+)"
    r"_dm(?P<d_model>\d+)"
    r"_nh(?P<n_heads>\d+)"
    r"_el(?P<e_layers>\d+)"
    r"_dl(?P<d_layers>\d+)"
    r"_df(?P<d_ff>\d+)"
    r"_fc(?P<factor>\d+)"
    r"_eb(?P<embed>[^_]+)"
    r"_dt(?P<distil>[^_]+)"
    r"_(?P<des>[^_]+)"
    r"_(?P<class_strategy>[^_]+)"
    r"_(?P<ii>\d+)"
    r"_ls(?P<loss>[^_]+)"
    r"_kt(?P<knot_multiplier>[^_]+)"
    r"_xp(?P<spline_criterion_exponent>[^_]+)"
    r"_a(?P<alpha>[^_]+)"
    r"_b(?P<beta>[^_]+)"
    r"_g(?P<gamma>[^_]+)$"
)


def parse_setting_name(folder_name: str) -> Optional[Dict[str, str]]:
    match = _SETTING_SUFFIX_RE.search(folder_name)
    if not match:
        return None

    prefix = folder_name[: match.start()]
    parts = prefix.split("_")
    if len(parts) < 4:
        return None

    model = parts[-3]
    data = parts[-2]
    data_path = parts[-1]
    model_id = "_".join(parts[:-3])

    if model not in KNOWN_MODELS:
        return None

    parsed = match.groupdict()
    parsed.update(
        {
            "model_id": model_id,
            "model": model,
            "data": data,
            "data_path": data_path,
        }
    )
    return parsed


def collect_results(results_dir: str = "./results") -> pd.DataFrame:
    rows: List[Dict] = []
    if not os.path.isdir(results_dir):
        return pd.DataFrame()

    for root, _, files in os.walk(results_dir):
        if "metrics.npy" not in files:
            continue
        folder_name = os.path.basename(root)
        metrics_path = os.path.join(root, "metrics.npy")

        setting = parse_setting_name(folder_name)
        if setting is None:
            continue

        metrics = np.load(metrics_path)
        setting["mae"] = float(metrics[0])
        setting["mse"] = float(metrics[1])
        setting["rmse"] = float(metrics[2])
        setting["mape"] = float(metrics[3])
        setting["mspe"] = float(metrics[4])
        rows.append(setting)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Drop duplicate settings (e.g. nested results/results/ from Modal download).
    # Include data_path so custom datasets (traffic, electricity, weather, …) are not merged.
    df = df.sort_values("mse").drop_duplicates(
        subset=["model", "data", "data_path", "pred_len", "loss"], keep="first"
    )
    for col in ["pred_len", "ii"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "loss" in df.columns:
        df["loss"] = df["loss"].str.lower()
    return df
