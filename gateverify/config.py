from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
import torch


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(cfg["_project_root"]) / path
    return path.resolve()


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg["project"].get("device", "cuda:0"))
    require_cuda = bool(cfg["project"].get("require_cuda", True))
    if require_cuda and not requested.startswith("cuda"):
        raise RuntimeError("project.require_cuda=true requires project.device to be a CUDA device")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false; refusing CPU fallback")
        device = torch.device(requested)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
        # Force CUDA initialization now so driver/runtime failures occur before a long job.
        torch.empty(1, device=device)
        return device
    if require_cuda:
        raise RuntimeError("CUDA is required; refusing CPU execution")
    return torch.device(requested)
