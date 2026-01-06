#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt

EPS = 1e-12


# =========================
# Math metrics (distribution -> scalar)
# =========================
def normalize(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p, dtype=np.float64) / len(p)
    return p / s


def entropy_norm(p: np.ndarray) -> float:
    p = normalize(p)
    H = -(p * np.log(p + EPS)).sum()
    return float(H / np.log(len(p) + EPS))


def gini_coeff(p: np.ndarray) -> float:
    p = normalize(p)
    p = np.sort(p)
    n = len(p)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2 * idx - n - 1) * p) / (n - 1 + EPS))


def cv_coeff(p: np.ndarray) -> float:
    p = normalize(p)
    mu = p.mean()
    return float(p.std() / (mu + EPS))


def neff(p: np.ndarray) -> float:
    p = normalize(p)
    H = -(p * np.log(p + EPS)).sum()
    return float(np.exp(H))


def topk_mass(p: np.ndarray, k: int = 4) -> float:
    p = normalize(p)
    k = min(k, len(p))
    return float(np.sort(p)[-k:].sum())


def active_fraction(p: np.ndarray, tau: float = 0.01) -> float:
    p = normalize(p)
    return float((p > tau).sum() / len(p))


# =========================
# Loading + aggregation
# =========================
def load_pt(pt_path: str) -> Dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu")
    if "meta" not in data or "samples" not in data:
        raise ValueError(f"Bad pt format: {pt_path}")
    return data


def aggregate_per_layer(data: Dict[str, Any]) -> np.ndarray:
    meta = data["meta"]
    num_layers = int(meta["num_moe_layers"])
    num_experts = int(meta["num_experts"])
    agg = np.zeros((num_layers, num_experts), dtype=np.float64)

    for s in data["samples"]:
        per_layer = s.get("per_layer_hits", None)
        if per_layer is None:
            continue
        for lid in range(num_layers):
            key = str(lid)
            if key not in per_layer:
                continue
            v = np.array(per_layer[key], dtype=np.float64)
            if len(v) != num_experts:
                if len(v) < num_experts:
                    v = np.pad(v, (0, num_experts - len(v)))
                else:
                    v = v[:num_experts]
            agg[lid] += v
    return agg


# =========================
# Plotting helpers
# =========================
def plot_heatmap(mat: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis = mat.copy().astype(np.float64)
    row_sums = vis.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    vis = vis / row_sums

    plt.figure(figsize=(12, 6))
    plt.imshow(vis, aspect="auto")
    plt.colorbar()
    plt.xlabel("Expert")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_curves(
    x: np.ndarray,
    curves: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    for name, y in curves.items():
        plt.plot(x, y, label=name)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


# =========================
# Core computation
# =========================
def compute_layerwise_stats(mat: np.ndarray) -> Dict[str, np.ndarray]:
    L, _ = mat.shape
    out = {
        "entropy_norm": np.zeros(L, dtype=np.float64),
        "gini": np.zeros(L, dtype=np.float64),
        "cv": np.zeros(L, dtype=np.float64),
        "neff": np.zeros(L, dtype=np.float64),
        "top4_mass": np.zeros(L, dtype=np.float64),
        "active_frac_1pct": np.zeros(L, dtype=np.float64),
        "active_frac_0p5pct": np.zeros(L, dtype=np.float64),
    }
    for lid in range(L):
        p = mat[lid]
        out["entropy_norm"][lid] = entropy_norm(p)
        out["gini"][lid] = gini_coeff(p)
        out["cv"][lid] = cv_coeff(p)
        out["neff"][lid] = neff(p)
        out["top4_mass"][lid] = topk_mass(p, k=4)
        out["active_frac_1pct"][lid] = active_fraction(p, tau=0.01)
        out["active_frac_0p5pct"][lid] = active_fraction(p, tau=0.005)
    return out


def main():
    ap = argparse.ArgumentParser("Routing analysis for MoE routing records (single dataset).")
    ap.add_argument("--input", type=str, required=True, help="Path to routing_records_*.pt")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_pt(args.input)
    agg = aggregate_per_layer(data)

    layers = np.arange(agg.shape[0], dtype=np.int64)
    stats = compute_layerwise_stats(agg)

    plot_heatmap(agg, "Routing heatmap (normalized per layer)", os.path.join(args.out_dir, "heatmap.png"))

    for metric_name in ["entropy_norm", "gini", "cv", "neff", "top4_mass", "active_frac_1pct", "active_frac_0p5pct"]:
        plot_curves(
            layers,
            {metric_name: stats[metric_name]},
            title=f"Gate randomness metric: {metric_name}",
            ylabel=metric_name,
            out_path=os.path.join(args.out_dir, f"curve_{metric_name}.png"),
        )

    header = [
        "layer",
        "entropy_norm",
        "gini",
        "cv",
        "neff",
        "top4_mass",
        "active_frac_1pct",
        "active_frac_0p5pct",
    ]
    rows = []
    for lid in range(agg.shape[0]):
        rows.append(
            [
                lid,
                stats["entropy_norm"][lid],
                stats["gini"][lid],
                stats["cv"][lid],
                stats["neff"][lid],
                stats["top4_mass"][lid],
                stats["active_frac_1pct"][lid],
                stats["active_frac_0p5pct"][lid],
            ]
        )
    write_csv(os.path.join(args.out_dir, "layer_metrics.csv"), header, rows)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"input={args.input}\n")
        f.write(f"num_layers={agg.shape[0]}, num_experts={agg.shape[1]}\n\n")
        f.write("Top layers by low entropy (more concentrated routing):\n")
        idx = np.argsort(stats["entropy_norm"])[: args.topk]
        for lid in idx:
            f.write(
                f"  layer {int(lid):2d}: entropy_norm={stats['entropy_norm'][lid]:.6f}, "
                f"top4_mass={stats['top4_mass'][lid]:.6f}\n"
            )

    print(f"[OK] Saved heatmap to: {os.path.join(args.out_dir, 'heatmap.png')}")
    print(f"[OK] Wrote CSV: {os.path.join(args.out_dir, 'layer_metrics.csv')}")
    print(f"[OK] Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
