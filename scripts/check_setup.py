#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from gateverify.config import load_config, resolve_device, resolve_path
from gateverify.data import load_test_rows
from gateverify.modeling import PixelMoE, SegmentedGate, load_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    model, run_cfg, _ = load_baseline(cfg, device)
    gate = SegmentedGate(PixelMoE(model)).eval()
    rows = load_test_rows(cfg, 1)
    if not rows or not Path(rows[0]["image_path"]).exists():
        raise FileNotFoundError("The first test image is missing; repair metadata image paths.")
    checkpoint = resolve_path(cfg, cfg["baseline"]["run_dir"]) / "checkpoints" / cfg["baseline"].get("checkpoint_name", "best.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    print(f"device={device}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"backbone={run_cfg['backbone_name']} image_size={run_cfg['image_size']}")
    print(f"checkpoint_sha256={digest}")
    print(f"router={gate.pixel_moe.model.router}")
    try:
        from auto_LiRPA import BoundedModule  # noqa: F401
        print("auto_LiRPA=available")
    except Exception as exc:
        print(f"auto_LiRPA=unavailable ({exc})")


if __name__ == "__main__":
    main()
