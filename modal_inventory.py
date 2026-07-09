#!/usr/bin/env python3
"""Inventory metrics stored on the Modal workspace volume without downloading arrays."""

from __future__ import annotations

import json
import os

import modal

VOLUME_NAME = os.environ.get("HNMD_MODAL_VOLUME", "hnmd-workspace")
WORKSPACE = "/workspace"

app = modal.App("hnmd-inventory")
workspace = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(image=image, volumes={WORKSPACE: workspace}, timeout=600)
def inventory() -> str:
    import numpy as np

    out = {"results": {}, "checkpoints": [], "extra_result_roots": {}}
    for root_name in sorted(os.listdir(WORKSPACE)):
        root = os.path.join(WORKSPACE, root_name)
        if not os.path.isdir(root):
            continue
        if root_name.startswith("results"):
            store = out["results"] if root_name == "results" else out["extra_result_roots"].setdefault(root_name, {})
            for dirpath, _, files in os.walk(root):
                if "metrics.npy" not in files:
                    continue
                rel = os.path.relpath(dirpath, root)
                try:
                    m = np.load(os.path.join(dirpath, "metrics.npy"))
                    store[rel] = [float(v) for v in m]
                except Exception as exc:  # noqa: BLE001
                    store[rel] = f"ERROR: {exc}"
        elif root_name.startswith("checkpoints"):
            for dirpath, _, files in os.walk(root):
                interesting = [f for f in files if f in ("checkpoint.pth", "training_state.pth")]
                if interesting:
                    out["checkpoints"].append(
                        {"dir": os.path.relpath(dirpath, WORKSPACE), "files": interesting}
                    )
    return json.dumps(out)


@app.local_entrypoint()
def main(output: str = "volume_inventory.json"):
    data = inventory.remote()
    with open(output, "w") as fh:
        fh.write(data)
    parsed = json.loads(data)
    print(f"results with metrics: {len(parsed['results'])}")
    for root, entries in parsed.get("extra_result_roots", {}).items():
        print(f"{root}: {len(entries)}")
    print(f"checkpoint dirs: {len(parsed['checkpoints'])}")
    print(f"wrote {output}")
