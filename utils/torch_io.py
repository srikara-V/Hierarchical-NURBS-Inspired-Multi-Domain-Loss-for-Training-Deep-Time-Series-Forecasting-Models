"""Checkpoint loading compatible with PyTorch 2.6+ weights_only default."""

from __future__ import annotations

from typing import Any, Optional

import torch


def torch_load(path: str, map_location: Optional[Any] = None) -> Any:
    """Load trusted local checkpoints and training state files."""
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
