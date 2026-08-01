#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gateverify.attacks import SurrogateRouter, hopskipjump, pgd_transfer, square_attack
from gateverify.config import load_config, resolve_device, resolve_path
from gateverify.data import PixelDataset, load_rows, load_test_rows
from gateverify.metrics import cosine_similarity, js_divergence, relative_l2
from gateverify.modeling import PixelMoE, SegmentedGate, load_baseline


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_surrogate(cfg, target, image_size, device):
    acfg = cfg["attacks"]["transfer_pgd"]
    rows = load_rows(cfg, "train.json", int(acfg["surrogate_train_samples"]))
    loader = DataLoader(PixelDataset(rows, image_size), batch_size=32, shuffle=True,
                        num_workers=int(cfg["data"].get("num_workers", 2)))
    model = SurrogateRouter().to(device); opt = torch.optim.AdamW(model.parameters(), lr=float(acfg["surrogate_lr"]))
    for epoch in range(int(acfg["surrogate_epochs"])):
        model.train(); correct = total = 0
        for batch in loader:
            x = batch["image"].to(device)
            with torch.no_grad(): y = target(x).argmax(1)
            logits = model(x); loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int(logits.argmax(1).eq(y).sum()); total += len(y)
        print(f"surrogate epoch={epoch+1} fidelity={correct/max(total,1):.4f}", flush=True)
    return model.eval()


def route_margin(logits, label):
    other = logits.clone(); other[:, label] = -torch.inf
    return float((logits[:, label] - other.max(1).values).item())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    p.add_argument("--attacks", nargs="+", choices=["hsj", "square", "transfer_pgd"], required=True)
    p.add_argument("--max-samples", type=int); args = p.parse_args()
    cfg = load_config(args.config); seed = int(cfg["project"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = resolve_device(cfg)
    moe, run_cfg, _ = load_baseline(cfg, device); pixel_moe = PixelMoE(moe).to(device).eval()
    gate = SegmentedGate(pixel_moe).to(device).eval()
    rows = load_test_rows(cfg, args.max_samples); dataset = PixelDataset(rows, int(run_cfg["image_size"]))
    surrogate = fit_surrogate(cfg, gate, int(run_cfg["image_size"]), device) if "transfer_pgd" in args.attacks else None
    out = resolve_path(cfg, cfg["project"]["output_dir"]) / "attacks"
    selected = set(args.attacks)
    # A separate attack invocation replaces only that attack's earlier rows;
    # completed results for other attacks remain available to make_report.py.
    sample_rows = [row for row in read_csv(out / "sample_metrics.csv")
                   if row.get("attack") not in selected]
    segment_rows = [row for row in read_csv(out / "segment_metrics.csv")
                    if row.get("attack") not in selected]
    for idx in range(len(dataset)):
        item = dataset[idx]; x = item["image"].unsqueeze(0).to(device); objects = item["objects"].numpy()
        with torch.no_grad():
            clean_seg = gate.forward_all(x); clean_logits = clean_seg["router_logits"]
            clean_route = int(clean_logits.argmax(1)); clean_alpha = clean_logits.softmax(1)
            clean_out = pixel_moe(x); clean_obj = clean_out["obj_fused_logits"].sigmoid()
        for eps in cfg["threat_model"]["epsilons"]:
            for attack_name in args.attacks:
                start = time.perf_counter()
                if attack_name == "hsj":
                    a = cfg["attacks"]["hsj"]; result = hopskipjump(gate, x, torch.tensor([clean_route], device=device), float(eps), int(a["steps"]), int(a["queries"]))
                elif attack_name == "square":
                    a = cfg["attacks"]["square"]; result = square_attack(gate, x, torch.tensor([clean_route], device=device), float(eps), int(a["queries"]), float(a["p_init"]), int(a.get("restarts", 1)))
                else:
                    a = cfg["attacks"]["transfer_pgd"]; result = pgd_transfer(surrogate, gate, x, torch.tensor([clean_route], device=device), float(eps), int(a["steps"]), float(a["step_size"]), int(a["restarts"]))
                runtime = time.perf_counter() - start; adv = result.adversarial
                with torch.no_grad():
                    adv_seg = gate.forward_all(adv); adv_logits = adv_seg["router_logits"]
                    adv_route = int(adv_logits.argmax(1)); adv_alpha = adv_logits.softmax(1)
                    adv_obj = pixel_moe(adv)["obj_fused_logits"].sigmoid()
                delta = (adv - x).flatten(1)
                row = {"index": idx, "image_path": item["image_path"], "attack": attack_name, "epsilon": eps,
                    "clean_route": clean_route, "adv_route": adv_route, "success": int(adv_route != clean_route),
                    "attack_status": result.status,
                    "linf": float(delta.abs().max()), "l2": float(delta.norm()), "queries": float(result.queries[0]),
                    "queries_are_upper_bound": int(attack_name in {"hsj", "square"}),
                    "runtime_seconds": runtime, "clean_margin": route_margin(clean_logits, clean_route),
                    "adv_margin": route_margin(adv_logits, clean_route), "route_js": float(js_divergence(clean_alpha, adv_alpha)[0])}
                for j, name in enumerate(("car", "pedestrian", "traffic_sign")):
                    row[f"y_{name}"] = float(objects[j]); row[f"clean_prob_{name}"] = float(clean_obj[0,j]); row[f"adv_prob_{name}"] = float(adv_obj[0,j])
                sample_rows.append(row)
                for segment in clean_seg:
                    segment_rows.append({"index": idx, "attack": attack_name, "epsilon": eps, "segment": segment,
                        "attack_status": result.status,
                        "cosine": float(cosine_similarity(clean_seg[segment], adv_seg[segment])[0]),
                        "relative_l2": float(relative_l2(clean_seg[segment], adv_seg[segment])[0])})
                print(f"sample={idx} attack={attack_name} eps={eps:.6f} success={adv_route != clean_route}", flush=True)
        # Preserve completed work if a later sample encounters an unexpected
        # error or the batch job is interrupted.
        write_csv(out / "sample_metrics.csv", sample_rows)
        write_csv(out / "segment_metrics.csv", segment_rows)
    write_csv(out / "sample_metrics.csv", sample_rows); write_csv(out / "segment_metrics.csv", segment_rows)
    attacks_present = sorted({row["attack"] for row in sample_rows})
    samples_by_attack = {
        name: len({int(float(row["index"])) for row in sample_rows if row["attack"] == name})
        for name in attacks_present
    }
    (out / "run_manifest.json").write_text(json.dumps(
        {"seed": seed, "attacks": attacks_present, "samples_by_attack": samples_by_attack},
        indent=2), encoding="utf-8")


if __name__ == "__main__": main()
