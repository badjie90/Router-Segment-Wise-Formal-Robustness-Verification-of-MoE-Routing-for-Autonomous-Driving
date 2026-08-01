from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .config import resolve_path


SEGMENTS = (
    "backbone_features",
    "router_normalized",
    "router_hidden_affine",
    "router_hidden_gelu",
    "router_logits",
)


def _import_module(path: Path):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import torchmetrics.classification  # noqa: F401
    except ImportError:
        class _TrainingMetricUnavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("torchmetrics is required only when baseline training metrics are used")
        tm = types.ModuleType("torchmetrics")
        classification = types.ModuleType("torchmetrics.classification")
        classification.MultilabelAUROC = _TrainingMetricUnavailable
        classification.MultilabelF1Score = _TrainingMetricUnavailable
        classification.MulticlassAccuracy = _TrainingMetricUnavailable
        tm.classification = classification
        sys.modules.setdefault("torchmetrics", tm)
        sys.modules.setdefault("torchmetrics.classification", classification)
    name = "bdd100k_moe_train_gateverify"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline training module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Any]:
    base = cfg["baseline"]
    train_module = _import_module(resolve_path(cfg, base["train_script"]))
    run_dir = resolve_path(cfg, base["run_dir"])
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        run_cfg = json.load(handle)
    metadata_dir = resolve_path(cfg, base["metadata_dir"])
    with (metadata_dir / "metadata_bundle.json").open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    model = train_module.build_model(
        stage=run_cfg.get("stage", "stage3_moe"),
        backbone_name=run_cfg["backbone_name"],
        num_weather=len(bundle["weather_to_id"]),
        num_scene=len(bundle["scene_to_id"]),
        num_time=len(bundle["time_to_id"]),
        pretrained=False,
    )
    checkpoint = run_dir / "checkpoints" / base.get("checkpoint_name", "best.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, run_cfg, train_module


class PixelMoE(nn.Module):
    """Model wrapper accepting raw RGB pixels in [0, 1]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model((x - self.mean) / self.std)


class SegmentedGate(nn.Module):
    """Exposes scientifically defined cut points through the trained gate."""

    def __init__(self, pixel_moe: PixelMoE):
        super().__init__()
        self.pixel_moe = pixel_moe

    def forward_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = (x - self.pixel_moe.mean) / self.pixel_moe.std
        h = self.pixel_moe.model.backbone(normalized)
        router = self.pixel_moe.model.router.net
        z_norm = router[0](h)
        z_affine = router[1](z_norm)
        z_gelu = router[2](z_affine)
        z_drop = router[3](z_gelu)
        logits = router[4](z_drop)
        return {
            "backbone_features": h,
            "router_normalized": z_norm,
            "router_hidden_affine": z_affine,
            "router_hidden_gelu": z_gelu,
            "router_logits": logits,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)["router_logits"]


class FeatureSegmentedGate(nn.Module):
    """Router-only gate accepting the frozen backbone's 768-D feature vector."""

    def __init__(self, router: nn.Sequential):
        super().__init__()
        self.router = router

    def forward_all(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_norm = self.router[0](h)
        z_affine = self.router[1](z_norm)
        z_gelu = self.router[2](z_affine)
        z_drop = self.router[3](z_gelu)
        logits = self.router[4](z_drop)
        return {
            "backbone_features": h,
            "router_normalized": z_norm,
            "router_hidden_affine": z_affine,
            "router_hidden_gelu": z_gelu,
            "router_logits": logits,
        }

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward_all(h)["router_logits"]


class SegmentOutput(nn.Module):
    def __init__(self, gate: SegmentedGate, segment: str):
        super().__init__()
        if segment not in SEGMENTS:
            raise ValueError(f"Unknown segment {segment!r}; choose from {SEGMENTS}")
        self.gate = gate
        self.segment = segment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate.forward_all(x)[self.segment]
