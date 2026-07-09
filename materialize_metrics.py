#!/usr/bin/env python3
"""Materialize metrics.npy files locally from volume_inventory.json.

Avoids downloading the (large) pred.npy/true.npy arrays: the paper tables only
need metrics. Run modal_inventory.py first to refresh the JSON.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def write_metrics(root: str, entries: dict) -> int:
    n = 0
    for rel, metrics in entries.items():
        if not isinstance(metrics, list):
            print(f"skip (bad metrics): {root}/{rel} -> {metrics}")
            continue
        out_dir = os.path.join(root, rel)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "metrics.npy"), np.array(metrics))
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="volume_inventory.json")
    args = parser.parse_args()

    inv = json.load(open(args.inventory))
    total = write_metrics("./results", inv.get("results", {}))
    print(f"results: {total} metrics files")
    for root, entries in inv.get("extra_result_roots", {}).items():
        n = write_metrics(f"./{root}", entries)
        print(f"{root}: {n} metrics files")


if __name__ == "__main__":
    main()
