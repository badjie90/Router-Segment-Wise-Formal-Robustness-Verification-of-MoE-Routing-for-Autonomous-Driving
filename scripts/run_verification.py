#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from gateverify.config import load_config, resolve_device, resolve_path
from gateverify.data import PixelDataset, load_test_rows
from gateverify.modeling import FeatureSegmentedGate, PixelMoE, SegmentOutput, SegmentedGate, load_baseline
from gateverify.verification import (
    assert_equivalent,
    bound_output,
    bound_route_margins,
    make_verification_compatible,
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--max-samples", type=int)
    p.add_argument("--fail-fast", action="store_true",
                   help="Re-raise the first verifier error with a complete traceback")
    args = p.parse_args(); cfg = load_config(args.config)
    device = resolve_device(cfg)
    model, run_cfg, _ = load_baseline(cfg, device)
    gate = SegmentedGate(PixelMoE(model)).to(device).eval()
    feature_gate = FeatureSegmentedGate(model.router.net).to(device).eval()
    verification_gate = make_verification_compatible(feature_gate).to(device).eval()
    dataset = PixelDataset(load_test_rows(cfg, args.max_samples), int(run_cfg["image_size"]))
    if len(dataset):
        equivalence_image = dataset[0]["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            equivalence_x = gate.forward_all(equivalence_image)["backbone_features"]
        assert_equivalent(feature_gate, verification_gate, equivalence_x)
        print("verifier_graph_equivalence=passed", flush=True)
    verifier = cfg["verification"]
    cert_rows, bound_rows = [], []
    out = resolve_path(cfg, cfg["project"]["output_dir"]) / "verification"
    for i in range(len(dataset)):
        item = dataset[i]; x = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            clean_all = gate.forward_all(x)
            clean = clean_all["router_logits"]
            label = int(clean.argmax(1))
            feature_x = clean_all["backbone_features"].detach()
            verification_clean_all = verification_gate.forward_all(feature_x)
        feature_scale = max(float(feature_x.std(unbiased=False)), 1e-12)
        for relative_radius in verifier["feature_relative_radii"]:
            eps = float(relative_radius) * feature_scale
            try:
                mlb, mub, secs = bound_route_margins(verification_gate, feature_x, label, eps, verifier["method"], verifier.get("bound_opts"))
                certified = bool(np.all(mlb > 0))
                cert_rows.append({"index": i, "image_path": item["image_path"], "epsilon": eps,
                                  "relative_radius": relative_radius, "threat_space": "router_feature_linf",
                                  "clean_route": label, "certified": int(certified),
                                  "margin_lb_min": float(mlb.min()), "margin_ub_min": float(mub.min()),
                                  "runtime_seconds": secs, "status": "certified" if certified else "unknown"})
                for segment in verifier["segments"]:
                    lb, ub, stime = bound_output(SegmentOutput(verification_gate, segment), feature_x, eps, verifier["method"], verifier.get("bound_opts"))
                    width = ub - lb
                    nominal = verification_clean_all[segment].detach().cpu().numpy()
                    # Sound coordinate-wise deviation from the clean activation.
                    radius = np.maximum(ub - nominal, nominal - lb)
                    bound_rows.append({"index": i, "epsilon": eps, "relative_radius": relative_radius,
                        "threat_space": "router_feature_linf", "segment": segment,
                        "dimension": width.size, "mean_width": float(width.mean()),
                        "max_width": float(width.max()), "cert_linf_radius": float(radius.max()),
                        "cert_l2_radius_upper": float(np.linalg.norm(radius.ravel(), 2)),
                        "amplification_upper": float(radius.max() / max(float(eps), 1e-12)),
                        "runtime_seconds": stime, "status": "bounded"})
                print(f"sample={i} feature_relative_radius={float(relative_radius):.4f} "
                      f"certified={certified} margin_lb_min={float(mlb.min()):.6f}", flush=True)
            except Exception as exc:
                cert_rows.append({"index": i, "image_path": item["image_path"], "epsilon": eps,
                    "relative_radius": relative_radius, "threat_space": "router_feature_linf",
                    "clean_route": label, "certified": 0, "margin_lb_min": np.nan,
                    "margin_ub_min": np.nan, "runtime_seconds": np.nan,
                    "status": f"error:{type(exc).__name__}:{exc}"})
                if args.fail_fast:
                    traceback.print_exc()
                    raise
        # Checkpoint completed samples so scheduler interruption does not lose work.
        write_csv(out / "certificates.csv", cert_rows)
        write_csv(out / "segment_bounds.csv", bound_rows)
    write_csv(out / "certificates.csv", cert_rows); write_csv(out / "segment_bounds.csv", bound_rows)


if __name__ == "__main__": main()
