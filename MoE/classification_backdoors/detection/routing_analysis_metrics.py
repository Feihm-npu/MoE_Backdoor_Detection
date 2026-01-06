#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

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
    # standard Gini for discrete distribution
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
# Divergence metrics (p vs q)
# =========================
def l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(p - q).sum())


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    return float((p * (np.log(p + EPS) - np.log(q + EPS))).sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)
    return float(0.5 * kl_div(p, m) + 0.5 * kl_div(q, m))


# =========================
# Loading + aggregation
# =========================
def _safe_get_layer_vec(per_layer_hits: Dict[str, Any], lid: int, num_experts: int) -> np.ndarray:
    """per_layer_hits[str(lid)] is list[int]"""
    key = str(lid)
    if key not in per_layer_hits:
        return np.zeros(num_experts, dtype=np.float64)
    v = np.array(per_layer_hits[key], dtype=np.float64)
    if len(v) != num_experts:
        # tolerate mismatch by pad/truncate
        if len(v) < num_experts:
            v = np.pad(v, (0, num_experts - len(v)))
        else:
            v = v[:num_experts]
    return v


def load_pt(pt_path: str) -> Dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu")
    if "meta" not in data or "samples" not in data:
        raise ValueError(f"Bad pt format: {pt_path}")
    return data


def aggregate_by_source(
    data: Dict[str, Any],
    which_hits_key: str,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      agg[source] = np.ndarray [num_layers, num_experts] (sum over samples)
    source in {"clean","triggered"} based on sample["source"]
    """
    meta = data["meta"]
    num_layers = int(meta["num_moe_layers"])
    num_experts = int(meta["num_experts"])

    agg = {
        "clean": np.zeros((num_layers, num_experts), dtype=np.float64),
        "triggered": np.zeros((num_layers, num_experts), dtype=np.float64),
    }

    for s in data["samples"]:
        src = s.get("source", None)
        if src not in agg:
            continue
        per_layer = s.get(which_hits_key, None)
        if per_layer is None:
            # allow missing key
            continue
        for lid in range(num_layers):
            agg[src][lid] += _safe_get_layer_vec(per_layer, lid, num_experts)

    return agg


# =========================
# Tabular writer (no pandas dependency)
# =========================
def write_csv(path: str, header: List[str], rows: List[List[Any]]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


# =========================
# Plotting helpers
# =========================
def plot_heatmap(mat: np.ndarray, title: str, out_path: str):
    """
    mat: [num_layers, num_experts] -> normalized row-wise for visualization
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vis = mat.copy().astype(np.float64)
    # row-normalize
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


# =========================
# Core computation
# =========================
def compute_layerwise_stats(mat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    mat: [L, E] counts
    returns dict of metric -> [L]
    """
    L, E = mat.shape
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


def compute_layerwise_divergence(mat_a: np.ndarray, mat_b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    divergence between row-normalized distributions of two mats [L,E]
    """
    assert mat_a.shape == mat_b.shape
    L, E = mat_a.shape
    L1 = np.zeros(L, dtype=np.float64)
    KL = np.zeros(L, dtype=np.float64)
    JSD = np.zeros(L, dtype=np.float64)

    for lid in range(L):
        p = normalize(mat_a[lid])
        q = normalize(mat_b[lid])
        L1[lid] = l1_dist(p, q)
        KL[lid] = kl_div(p, q)
        JSD[lid] = jsd(p, q)
    return {"L1": L1, "KL": KL, "JSD": JSD}


def topk_layers(metric: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    idx = np.argsort(-metric)[:k]
    return [(int(i), float(metric[i])) for i in idx]


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser("Routing analysis metrics for MoE (supports trigger/nontrigger hits).")
    ap.add_argument("--cleanft", type=str, required=True)
    ap.add_argument("--backdoored", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data_A = load_pt(args.cleanft)
    data_B = load_pt(args.backdoored)

    # sanity: layer/expert size
    L_A = int(data_A["meta"]["num_moe_layers"])
    E_A = int(data_A["meta"]["num_experts"])
    L_B = int(data_B["meta"]["num_moe_layers"])
    E_B = int(data_B["meta"]["num_experts"])
    if (L_A, E_A) != (L_B, E_B):
        raise ValueError(f"Shape mismatch: A(L={L_A},E={E_A}) vs B(L={L_B},E={E_B})")

    L, E = L_A, E_A
    layers = np.arange(L, dtype=np.int64)

    # -------- aggregate full / trigger / nontrigger for each model and each source
    # Full tokens
    A_full = aggregate_by_source(data_A, "per_layer_hits")
    B_full = aggregate_by_source(data_B, "per_layer_hits")

    # Trigger-only tokens
    A_trig = aggregate_by_source(data_A, "per_layer_hits_trigger")
    B_trig = aggregate_by_source(data_B, "per_layer_hits_trigger")

    # Nontrigger-only tokens
    A_non = aggregate_by_source(data_A, "per_layer_hits_nontrigger")
    B_non = aggregate_by_source(data_B, "per_layer_hits_nontrigger")

    # -------- stats: per model, per source, per token-scope
    # We'll focus on:
    # 1) clean vs triggered divergence (FULL / TRIGGER / NONTRIGGER)
    # 2) trigger vs nontrigger divergence within triggered set (TRIGGERED source only)
    # 3) gate randomness metrics of each distribution

    # === 1) clean vs triggered divergence (full)
    div_A_full = compute_layerwise_divergence(A_full["clean"], A_full["triggered"])
    div_B_full = compute_layerwise_divergence(B_full["clean"], B_full["triggered"])

    div_A_trig = compute_layerwise_divergence(A_trig["clean"], A_trig["triggered"])
    div_B_trig = compute_layerwise_divergence(B_trig["clean"], B_trig["triggered"])

    div_A_non = compute_layerwise_divergence(A_non["clean"], A_non["triggered"])
    div_B_non = compute_layerwise_divergence(B_non["clean"], B_non["triggered"])

    # === 2) trigger-only vs nontrigger-only divergence within triggered set
    # (This directly captures "trigger causes routing shift" localized on trigger tokens)
    div_A_t_vs_n = compute_layerwise_divergence(A_trig["triggered"], A_non["triggered"])
    div_B_t_vs_n = compute_layerwise_divergence(B_trig["triggered"], B_non["triggered"])

    # === 3) randomness metrics for each distribution we care about
    # Full distributions
    A_full_clean_stats = compute_layerwise_stats(A_full["clean"])
    A_full_trig_stats = compute_layerwise_stats(A_full["triggered"])
    B_full_clean_stats = compute_layerwise_stats(B_full["clean"])
    B_full_trig_stats = compute_layerwise_stats(B_full["triggered"])

    # Trigger-only distributions (more sensitive)
    A_trig_clean_stats = compute_layerwise_stats(A_trig["clean"])
    A_trig_trig_stats = compute_layerwise_stats(A_trig["triggered"])
    B_trig_clean_stats = compute_layerwise_stats(B_trig["clean"])
    B_trig_trig_stats = compute_layerwise_stats(B_trig["triggered"])

    # Nontrigger-only distributions
    A_non_clean_stats = compute_layerwise_stats(A_non["clean"])
    A_non_trig_stats = compute_layerwise_stats(A_non["triggered"])
    B_non_clean_stats = compute_layerwise_stats(B_non["clean"])
    B_non_trig_stats = compute_layerwise_stats(B_non["triggered"])

    # -------- save heatmaps (normalized per layer in plot)
    plot_heatmap(A_full["triggered"], "cleanft FULL (triggered set)", os.path.join(args.out_dir, "heatmap_cleanft_full_triggered.png"))
    plot_heatmap(B_full["triggered"], "backdoored FULL (triggered set)", os.path.join(args.out_dir, "heatmap_backdoored_full_triggered.png"))

    plot_heatmap(A_trig["triggered"], "cleanft TRIGGER-ONLY (triggered set)", os.path.join(args.out_dir, "heatmap_cleanft_trigger_only.png"))
    plot_heatmap(B_trig["triggered"], "backdoored TRIGGER-ONLY (triggered set)", os.path.join(args.out_dir, "heatmap_backdoored_trigger_only.png"))

    plot_heatmap(A_non["triggered"], "cleanft NONTRIGGER-ONLY (triggered set)", os.path.join(args.out_dir, "heatmap_cleanft_nontrigger_only.png"))
    plot_heatmap(B_non["triggered"], "backdoored NONTRIGGER-ONLY (triggered set)", os.path.join(args.out_dir, "heatmap_backdoored_nontrigger_only.png"))

    # -------- curves for key randomness metrics (triggered set, full)
    for metric_name in ["entropy_norm", "gini", "cv", "neff", "top4_mass", "active_frac_1pct"]:
        plot_curves(
            layers,
            {
                f"cleanft_full_clean": A_full_clean_stats[metric_name],
                f"cleanft_full_triggered": A_full_trig_stats[metric_name],
                f"backdoored_full_clean": B_full_clean_stats[metric_name],
                f"backdoored_full_triggered": B_full_trig_stats[metric_name],
            },
            title=f"Gate randomness metric: {metric_name} (FULL)",
            ylabel=metric_name,
            out_path=os.path.join(args.out_dir, f"curve_{metric_name}_full.png"),
        )

        plot_curves(
            layers,
            {
                f"cleanft_trigger_only(triggered-set)": A_trig_trig_stats[metric_name],
                f"cleanft_nontrigger_only(triggered-set)": A_non_trig_stats[metric_name],
                f"backdoored_trigger_only(triggered-set)": B_trig_trig_stats[metric_name],
                f"backdoored_nontrigger_only(triggered-set)": B_non_trig_stats[metric_name],
            },
            title=f"Gate randomness metric: {metric_name} (trigger-only vs nontrigger-only, triggered set)",
            ylabel=metric_name,
            out_path=os.path.join(args.out_dir, f"curve_{metric_name}_trigger_vs_nontrigger.png"),
        )

    # -------- CSV exports
    # 1) full token metrics per layer
    rows_full = []
    header_full = [
        "model", "scope", "source", "layer",
        "entropy_norm", "gini", "cv", "neff", "top4_mass", "active_frac_1pct", "active_frac_0p5pct",
        "L1_clean_vs_trigger", "KL_clean_vs_trigger", "JSD_clean_vs_trigger",
    ]

    def _append_rows(model_name: str, scope: str,
                     clean_stats: Dict[str, np.ndarray], trig_stats: Dict[str, np.ndarray],
                     div: Dict[str, np.ndarray]):
        for lid in range(L):
            # write two rows: clean + triggered
            for src, st in [("clean", clean_stats), ("triggered", trig_stats)]:
                rows_full.append([
                    model_name, scope, src, lid,
                    st["entropy_norm"][lid], st["gini"][lid], st["cv"][lid], st["neff"][lid],
                    st["top4_mass"][lid], st["active_frac_1pct"][lid], st["active_frac_0p5pct"][lid],
                    div["L1"][lid], div["KL"][lid], div["JSD"][lid],
                ])

    _append_rows("cleanft", "full", A_full_clean_stats, A_full_trig_stats, div_A_full)
    _append_rows("backdoored", "full", B_full_clean_stats, B_full_trig_stats, div_B_full)

    write_csv(os.path.join(args.out_dir, "layer_metrics_full.csv"), header_full, rows_full)

    # 2) trigger vs nontrigger divergence within triggered set
    rows_tn = []
    header_tn = [
        "model", "layer",
        "L1_trigger_vs_nontrigger(triggered-set)",
        "KL_trigger_vs_nontrigger(triggered-set)",
        "JSD_trigger_vs_nontrigger(triggered-set)",
        "entropy_norm_trigger_only(triggered-set)",
        "entropy_norm_nontrigger_only(triggered-set)",
        "gini_trigger_only(triggered-set)",
        "gini_nontrigger_only(triggered-set)",
        "top4_mass_trigger_only(triggered-set)",
        "top4_mass_nontrigger_only(triggered-set)",
        "active_frac_1pct_trigger_only(triggered-set)",
        "active_frac_1pct_nontrigger_only(triggered-set)",
    ]

    for lid in range(L):
        rows_tn.append([
            "cleanft", lid,
            div_A_t_vs_n["L1"][lid], div_A_t_vs_n["KL"][lid], div_A_t_vs_n["JSD"][lid],
            A_trig_trig_stats["entropy_norm"][lid], A_non_trig_stats["entropy_norm"][lid],
            A_trig_trig_stats["gini"][lid], A_non_trig_stats["gini"][lid],
            A_trig_trig_stats["top4_mass"][lid], A_non_trig_stats["top4_mass"][lid],
            A_trig_trig_stats["active_frac_1pct"][lid], A_non_trig_stats["active_frac_1pct"][lid],
        ])
        rows_tn.append([
            "backdoored", lid,
            div_B_t_vs_n["L1"][lid], div_B_t_vs_n["KL"][lid], div_B_t_vs_n["JSD"][lid],
            B_trig_trig_stats["entropy_norm"][lid], B_non_trig_stats["entropy_norm"][lid],
            B_trig_trig_stats["gini"][lid], B_non_trig_stats["gini"][lid],
            B_trig_trig_stats["top4_mass"][lid], B_non_trig_stats["top4_mass"][lid],
            B_trig_trig_stats["active_frac_1pct"][lid], B_non_trig_stats["active_frac_1pct"][lid],
        ])

    write_csv(os.path.join(args.out_dir, "layer_metrics_trigger_vs_nontrigger.csv"), header_tn, rows_tn)

    # -------- summary txt (rankings)
    out_txt = os.path.join(args.out_dir, "metrics_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Routing Analysis Metrics (with trigger/nontrigger hits) ===\n")
        f.write(f"A=cleanft: {args.cleanft}\n")
        f.write(f"B=backdoored: {args.backdoored}\n")
        f.write(f"num_layers={L}, num_experts={E}\n\n")

        # 1) clean vs triggered JSD (FULL)
        f.write("=== Clean vs Triggered divergence (FULL tokens): Top layers by JSD ===\n")
        f.write("[cleanft]\n")
        for lid, v in topk_layers(div_A_full["JSD"], k=args.topk):
            f.write(f"  layer {lid:2d}: JSD={v:.6f}, L1={div_A_full['L1'][lid]:.6f}, KL={div_A_full['KL'][lid]:.6f}\n")
        f.write("[backdoored]\n")
        for lid, v in topk_layers(div_B_full["JSD"], k=args.topk):
            f.write(f"  layer {lid:2d}: JSD={v:.6f}, L1={div_B_full['L1'][lid]:.6f}, KL={div_B_full['KL'][lid]:.6f}\n")
        f.write("\n")

        # 2) trigger-only vs nontrigger-only divergence within triggered set
        f.write("=== Trigger-only vs Nontrigger-only (within triggered set): Top layers by JSD ===\n")
        f.write("[cleanft]\n")
        for lid, v in topk_layers(div_A_t_vs_n["JSD"], k=args.topk):
            f.write(f"  layer {lid:2d}: JSD={v:.6f}, L1={div_A_t_vs_n['L1'][lid]:.6f}, KL={div_A_t_vs_n['KL'][lid]:.6f}\n")
        f.write("[backdoored]\n")
        for lid, v in topk_layers(div_B_t_vs_n["JSD"], k=args.topk):
            f.write(f"  layer {lid:2d}: JSD={v:.6f}, L1={div_B_t_vs_n['L1'][lid]:.6f}, KL={div_B_t_vs_n['KL'][lid]:.6f}\n")
        f.write("\n")

        # 3) Cross-model delta in randomness metrics (trigger-only, triggered set)
        # (This is usually what matches your heatmap intuition the best)
        f.write("=== Cross-model Δ(randomness) on TRIGGER-ONLY (triggered set): backdoored - cleanft ===\n")
        for metric_name in ["entropy_norm", "gini", "cv", "neff", "top4_mass", "active_frac_1pct"]:
            delta = B_trig_trig_stats[metric_name] - A_trig_trig_stats[metric_name]
            # for entropy/neff/active_frac: larger => more uniform
            # for gini/cv/top4_mass: larger => more biased
            f.write(f"[{metric_name}] Top layers by |Δ|:\n")
            idx = np.argsort(-np.abs(delta))[:args.topk]
            for lid in idx:
                f.write(f"  layer {int(lid):2d}: Δ={float(delta[lid]):+.6f} (cleanft={A_trig_trig_stats[metric_name][lid]:.6f}, backdoored={B_trig_trig_stats[metric_name][lid]:.6f})\n")
            f.write("\n")

    print(f"[OK] Wrote summary to: {out_txt}")
    print(f"[OK] Wrote CSV: {os.path.join(args.out_dir, 'layer_metrics_full.csv')}")
    print(f"[OK] Wrote CSV: {os.path.join(args.out_dir, 'layer_metrics_trigger_vs_nontrigger.csv')}")
    print(f"[OK] Saved heatmaps/curves under: {args.out_dir}")


if __name__ == "__main__":
    main()
