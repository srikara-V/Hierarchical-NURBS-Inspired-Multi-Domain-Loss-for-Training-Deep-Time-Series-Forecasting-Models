"""Device selection: CUDA when available, else Apple MPS, else CPU."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def cuda_is_available() -> bool:
    return torch.cuda.is_available()


def accelerator_name() -> str:
    if cuda_is_available():
        return "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"


def resolve_device(use_gpu: bool = True, gpu: int = 0) -> torch.device:
    if not use_gpu:
        return torch.device("cpu")
    if cuda_is_available():
        return torch.device(f"cuda:{gpu}")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_training_device(args) -> None:
    """Set args.device, args.use_gpu, and args.use_amp after parsing CLI flags."""
    requested_gpu = bool(getattr(args, "use_gpu", True))
    args.device = resolve_device(use_gpu=requested_gpu, gpu=getattr(args, "gpu", 0))
    args.use_gpu = args.device.type != "cpu"

    if args.device.type == "cuda" and getattr(args, "use_multi_gpu", False):
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        args.device = torch.device(f"cuda:{args.gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if len(args.device_ids) == 1 else args.devices
    elif args.device.type == "mps":
        args.use_multi_gpu = False

    # Mixed precision is only enabled for CUDA (stable); disable on MPS/CPU.
    if getattr(args, "use_amp", False) and args.device.type != "cuda":
        args.use_amp = False


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def amp_enabled(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


@contextmanager
def autocast_context(device: torch.device, use_amp: bool):
    if amp_enabled(device, use_amp):
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def get_grad_scaler(device: torch.device, use_amp: bool):
    if amp_enabled(device, use_amp):
        return torch.cuda.amp.GradScaler()
    return None
