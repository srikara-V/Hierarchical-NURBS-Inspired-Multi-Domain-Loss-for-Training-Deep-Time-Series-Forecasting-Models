"""Save and restore in-progress training for resume after interruption."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from utils.experiment_paths import training_state_path


def save_training_state(
    setting: str,
    *,
    epoch: int,
    early_stopping_state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    checkpoints_dir: str = "./checkpoints",
) -> None:
    path = training_state_path(setting, checkpoints_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "early_stopping": early_stopping_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)


def load_training_state(
    setting: str,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    checkpoints_dir: str = "./checkpoints",
) -> Optional[Dict[str, Any]]:
    path = training_state_path(setting, checkpoints_dir)
    if not os.path.isfile(path):
        return None

    payload = torch.load(path, map_location="cpu")
    optimizer_state = payload.get("optimizer")
    if optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
        except (KeyError, ValueError) as exc:
            print(f"Warning: could not restore optimizer state ({exc}); using fresh optimizer.")
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def clear_training_state(setting: str, checkpoints_dir: str = "./checkpoints") -> None:
    path = training_state_path(setting, checkpoints_dir)
    if os.path.isfile(path):
        os.remove(path)
