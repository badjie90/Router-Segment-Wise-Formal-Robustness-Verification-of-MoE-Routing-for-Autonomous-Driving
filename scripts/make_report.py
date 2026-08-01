#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from gateverify.config import load_config, resolve_path


OBJECTS = ("car", "pedestrian", "traffic_sign")


def configure_plot_style(dpi=300):
    """Consistent, legible typography for publication-ready report figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.size": 15,
        "font.weight": "bold",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 14,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#B8B8B8",
        "grid.linewidth": 0.90,
        "grid.alpha": 0.40,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })


def embolden_plot_text(ax):
    """Bold tick and legend text, which rcParams do not always propagate."""
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    legend = ax.get_legend()
    if legend is not None:
        if legend.get_title() is not None:
            legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontweight("bold")


def bootstrap_mean_ci(values, resamples, confidence, seed=42):
    x = np.asarray(values, dtype=float)
    if not len(x): return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(resamples)])
    alpha = (1 - confidence) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1-alpha))


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): total += abs(y[mask].mean() - p[mask].mean()) * mask.mean()
    return float(total)


def multilabel_metrics(frame, prefix):
    y = frame[[f"y_{x}" for x in OBJECTS]].to_numpy(int)
    p = frame[[f"{prefix}_prob_{x}" for x in OBJECTS]].to_numpy(float)
    pred = p >= 0.5; aps=[]; aucs=[]; bals=[]
    for j in range(3):
        if y[:,j].sum() > 0: aps.append(average_precision_score(y[:,j], p[:,j]))
        if len(np.unique(y[:,j])) > 1:
            aucs.append(roc_auc_score(y[:,j], p[:,j])); bals.append(balanced_accuracy_score(y[:,j], pred[:,j]))
    return {"macro_ap": float(np.mean(aps)) if aps else np.nan,
            "macro_auroc": float(np.mean(aucs)) if aucs else np.nan,
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "macro_balanced_accuracy": float(np.mean(bals)) if bals else np.nan,
            "micro_ece": ece(y.ravel(), p.ravel())}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args = p.parse_args()
    cfg = load_config(args.config); root = resolve_path(cfg, cfg["project"]["output_dir"])
    configure_plot_style(int(cfg["report"].get("dpi", 300)))
    report = root / "report"; plots = report / "plots"; plots.mkdir(parents=True, exist_ok=True)
    cert_path, bounds_path = root / "verification/certificates.csv", root / "verification/segment_bounds.csv"
    attacks_path, segments_path = root / "attacks/sample_metrics.csv", root / "attacks/segment_metrics.csv"
    summary = {}
    if cert_path.exists():
        cert = pd.read_csv(cert_path)
        cert_radius = "relative_radius" if "relative_radius" in cert else "epsilon"
        cert["verification_error"] = cert["status"].astype(str).str.startswith("error:")
        attempted = cert.groupby(cert_radius, as_index=False).agg(
            attempted_samples=("index", "count"),
            verification_error_rate=("verification_error", "mean"))
        valid_cert = cert[~cert["verification_error"]]
        agg = valid_cert.groupby(cert_radius, as_index=False).agg(
            certified_rate=("certified", "mean"), valid_samples=("index", "count"))
        agg = attempted.merge(agg, on=cert_radius, how="left")
        agg.to_csv(report / "aggregate_metrics.csv", index=False)
        sns.lineplot(data=agg, x=cert_radius, y="certified_rate", marker="o")
        plt.ylim(0, 1); plt.ylabel("Certified routing rate"); plt.tight_layout()
        embolden_plot_text(plt.gca())
        plt.savefig(plots / "certified_rate_vs_epsilon.pdf"); plt.close()
        summary["certified_rate"] = agg.to_dict(orient="records")
        sns.countplot(data=cert, x=cert_radius, hue="status")
        plt.ylabel("Samples"); embolden_plot_text(plt.gca()); plt.tight_layout(); plt.savefig(plots / "verification_outcomes.pdf"); plt.close()
    
    
    if bounds_path.exists():
        bounds = pd.read_csv(bounds_path)
        bounds_radius = "relative_radius" if "relative_radius" in bounds else "epsilon"
        sns.lineplot(data=bounds, x="segment", y="mean_width", hue=bounds_radius, marker="o")
        plt.xticks(rotation=25, ha="right"); plt.ylabel("Certified mean interval width"); plt.tight_layout()
        embolden_plot_text(plt.gca())
        plt.savefig(plots / "segment_bound_width.pdf"); plt.close()
    if attacks_path.exists():
        attacks = pd.read_csv(attacks_path)
        if "attack_status" not in attacks:
            attacks["attack_status"] = "ok"
        attempted = attacks.groupby(["attack", "epsilon"], as_index=False).agg(
            attempted_samples=("index", "count"),
            attack_error_rate=("attack_status", lambda x: float((x != "ok").mean())))
        valid_attacks = attacks[attacks["attack_status"] == "ok"].copy()
        attack_agg = valid_attacks.groupby(["attack", "epsilon"], as_index=False).agg(
            attack_success_rate=("success", "mean"), mean_linf=("linf", "mean"),
            mean_queries=("queries", "mean"), mean_runtime_seconds=("runtime_seconds", "mean"),
            valid_samples=("index", "count"))
        attack_agg = attempted.merge(attack_agg, on=["attack", "epsilon"], how="left")
        attack_agg.to_csv(report / "attack_aggregate_metrics.csv", index=False)
        sns.lineplot(data=attack_agg, x="epsilon", y="attack_success_rate", hue="attack", marker="o")
        plt.ylim(0, 1); plt.ylabel("Attack success rate"); plt.tight_layout()
        embolden_plot_text(plt.gca())
        plt.savefig(plots / "attack_success_vs_epsilon.pdf"); plt.close()
        summary["attack_success"] = attack_agg.to_dict(orient="records")
        resamples = int(cfg["report"].get("bootstrap_resamples", 2000)); confidence = float(cfg["report"].get("confidence", .95))
        scientific_rows=[]
        
        
        for (name, eps), group in valid_attacks.groupby(["attack", "epsilon"]):
            mean, lo, hi = bootstrap_mean_ci(group["success"], resamples, confidence)
            row={"attack":name, "epsilon":eps, "attack_success_rate":mean,
                 "attack_success_ci_low":lo, "attack_success_ci_high":hi,
                 "route_agreement":1-mean, "mean_route_js":group["route_js"].mean(),
                 "mean_clean_margin":group["clean_margin"].mean(), "mean_adv_margin":group["adv_margin"].mean()}
            row.update({f"clean_{k}":v for k,v in multilabel_metrics(group,"clean").items()})
            row.update({f"adv_{k}":v for k,v in multilabel_metrics(group,"adv").items()})
            scientific_rows.append(row)
        pd.DataFrame(scientific_rows).to_csv(report / "scientific_metrics.csv", index=False)
        margin = valid_attacks.melt(id_vars=["attack","epsilon"], value_vars=["clean_margin","adv_margin"], var_name="state", value_name="route_margin")
        sns.boxplot(data=margin, x="epsilon", y="route_margin", hue="state", showfliers=False)
        plt.axhline(0, color="black", linewidth=1); embolden_plot_text(plt.gca()); plt.tight_layout()
        plt.savefig(plots / "route_margin_clean_adv.pdf"); plt.close()
        for attack_name, attack_margin in margin.groupby("attack"):
            sns.boxplot(data=attack_margin, x="epsilon", y="route_margin", hue="state", showfliers=False)
            ax = plt.gca()
            plt.axhline(0, color="black", linewidth=1); plt.title(str(attack_name))
            if str(attack_name) == "hsj":
                sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 0.60),
                                ncol=2, frameon=True, title=None)
            embolden_plot_text(ax); plt.tight_layout()
            plt.savefig(plots / f"route_margin_clean_adv_{attack_name}.pdf"); plt.close()
    
    
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        if "attack_status" in segments:
            segments = segments[segments["attack_status"] == "ok"]
        sns.lineplot(data=segments, x="segment", y="relative_l2", hue="attack", style="epsilon", markers=True)
        plt.xticks(rotation=25, ha="right"); plt.ylabel("Relative L2 representation drift"); plt.tight_layout()
        embolden_plot_text(plt.gca())
        plt.savefig(plots / "segment_empirical_drift.pdf"); plt.close()
    (report / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
