#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
routing_analysis.py

从 assets/routing_data.pt 读取 routing 统计信息，做：
1) 全局（所有 label）每层 expert 使用热力图 + 平衡性指标；
2) 按 label 的 expert 使用热力图 + 与全局分布的差异。
"""

import os
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

ASSETS_DIR = "assets"
ROUTING_PATH = os.path.join(ASSETS_DIR, "routing_data.pt")
METRICS_LOG = os.path.join(ASSETS_DIR, "routing_metrics.txt")


# ============================
# 工具函数
# ============================

def _ensure_assets_dir():
    os.makedirs(ASSETS_DIR, exist_ok=True)


def _normalized_freq(counts_row):
    """把一行 counts 正则化为概率分布 p。"""
    counts_row = np.asarray(counts_row, dtype=np.float64)
    total = counts_row.sum()
    if total <= 0:
        return np.zeros_like(counts_row, dtype=np.float64)
    return counts_row / total


# ============================
# 全局（不分 label）分析
# ============================

def analyze_global_layer_usage(data):
    meta = data["meta"]
    samples = data["samples"]

    L = meta["num_moe_layers"]
    E = meta["num_experts"]

    global_hits = np.zeros((L, E), dtype=np.int64)

    # 累加所有样本
    for s in samples:
        per_layer = s["per_layer_hits"]  # { "0": [...], "1": [...], ... }
        for lid_str, hits_list in per_layer.items():
            lid = int(lid_str)
            global_hits[lid] += np.asarray(hits_list, dtype=np.int64)

    # 每层总 token 数
    token_counts = global_hits.sum(axis=1)  # [L]

    # 归一化频率矩阵 freq[L, E]
    freq = np.zeros_like(global_hits, dtype=np.float64)
    for l in range(L):
        freq[l] = _normalized_freq(global_hits[l])

    # 计算每层平衡性 / 差异性指标
    metrics = []
    eps = 1e-12
    H_max = math.log(E + eps)

    for l in range(L):
        tc = int(token_counts[l])
        if tc == 0:
            metrics.append({
                "layer": l,
                "token_count": 0,
                "entropy": None,
                "entropy_norm": None,
                "cv": None,
                "gini": None,
            })
            continue

        p = freq[l]
        H = -np.sum(p * np.log(p + eps))
        H_norm = H / H_max if H_max > 0 else None
        mean = p.mean()
        std = p.std()
        cv = std / (mean + eps)

        # Gini 系数
        sorted_p = np.sort(p)
        cum = np.cumsum(sorted_p)
        gini = 1.0 - 2.0 * np.sum(cum) / (E * cum[-1] + eps)

        metrics.append({
            "layer": l,
            "token_count": tc,
            "entropy": float(H),
            "entropy_norm": float(H_norm),
            "cv": float(cv),
            "gini": float(gini),
        })

    # 画全局热力图
    _ensure_assets_dir()
    plt.figure(figsize=(12, 6))
    im = plt.imshow(freq, aspect="auto")
    plt.colorbar(im, label="Expert usage frequency")
    plt.xlabel("Expert id")
    plt.ylabel("MoE layer id")
    plt.title("Global expert usage heatmap (normalized per layer)")
    heatmap_path = os.path.join(ASSETS_DIR, "global_expert_usage_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    return {
        "global_hits": global_hits,
        "freq": freq,
        "token_counts": token_counts,
        "metrics": metrics,
        "heatmap_path": heatmap_path,
    }


# ============================
# 按 label 分析
# ============================

def analyze_label_expert_usage(data, global_freq=None):
    meta = data["meta"]
    samples = data["samples"]

    L = meta["num_moe_layers"]
    E = meta["num_experts"]
    label_map = meta["label_map"]  # {0: "World", ...} 或类似结构
    # 兼容 int/str key
    label_map = {int(k): v for k, v in label_map.items()}

    # 初始化：每个 label 一张 [L, E] 计数矩阵
    label_ids = sorted({int(s["true_label_id"]) for s in samples})
    label_hits = {
        lid: np.zeros((L, E), dtype=np.int64)
        for lid in label_ids
    }

    # 按 label 累加
    for s in samples:
        lid = int(s["true_label_id"])
        per_layer = s["per_layer_hits"]
        for layer_str, hits_list in per_layer.items():
            layer_id = int(layer_str)
            label_hits[lid][layer_id] += np.asarray(hits_list, dtype=np.int64)

    results = {}
    eps = 1e-12

    for lid in label_ids:
        hits_mat = label_hits[lid]            # [L, E]
        token_counts = hits_mat.sum(axis=1)   # [L]
        freq = np.zeros_like(hits_mat, dtype=np.float64)

        for l in range(L):
            freq[l] = _normalized_freq(hits_mat[l])

        label_name = label_map.get(lid, str(lid))

        # 画每个 label 的热力图
        _ensure_assets_dir()
        plt.figure(figsize=(12, 6))
        im = plt.imshow(freq, aspect="auto")
        plt.colorbar(im, label="Expert usage frequency")
        plt.xlabel("Expert id")
        plt.ylabel("MoE layer id")
        plt.title(f"Expert usage heatmap for label = {label_name}")
        safe_label = label_name.replace("/", "_").replace(" ", "_")
        path = os.path.join(ASSETS_DIR, f"expert_usage_label_{safe_label}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        # 计算与全局分布的差异（逐层）
        label_metrics = []
        if global_freq is not None:
            for l in range(L):
                if token_counts[l] == 0:
                    label_metrics.append({
                        "layer": l,
                        "token_count": 0,
                        "L1_vs_global": None,
                        "KL_vs_global": None,
                    })
                    continue
                p = freq[l]
                q = global_freq[l]
                l1 = float(np.sum(np.abs(p - q)))
                kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
                label_metrics.append({
                    "layer": l,
                    "token_count": int(token_counts[l]),
                    "L1_vs_global": l1,
                    "KL_vs_global": kl,
                })

        # 整体 top-k expert（跨所有层）
        total_hits = hits_mat.sum(axis=0)  # [E]
        total_tokens = total_hits.sum()
        if total_tokens > 0:
            p_all = total_hits / total_tokens
            topk = min(5, E)
            top_idx = np.argsort(p_all)[::-1][:topk]
            top_experts = [(int(i), float(p_all[i])) for i in top_idx]
        else:
            top_experts = []

        results[lid] = {
            "label_name": label_name,
            "hits": hits_mat,
            "freq": freq,
            "token_counts": token_counts,
            "heatmap_path": path,
            "metrics": label_metrics,
            "top_experts": top_experts,
        }

    return results


# ============================
# 写指标日志
# ============================

def write_metrics_log(global_res, label_res, path=METRICS_LOG):
    _ensure_assets_dir()
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Global per-layer expert usage metrics ===\n")
        for m in global_res["metrics"]:
            f.write(
                f"Layer {m['layer']:2d}: "
                f"tokens={m['token_count']}, "
                f"entropy={m['entropy']}, "
                f"entropy_norm={m['entropy_norm']}, "
                f"cv={m['cv']}, "
                f"gini={m['gini']}\n"
            )

        f.write("\n=== Per-label expert usage metrics ===\n")
        for lid, info in label_res.items():
            f.write(f"\n[Label {lid} - {info['label_name']}]\n")
            f.write("Top experts overall (expert_id: freq):\n")
            for eid, freq in info["top_experts"]:
                f.write(f"  Expert {eid:3d}: {freq:.6f}\n")

            f.write("Per-layer divergence vs global (only layers with tokens):\n")
            for m in info["metrics"]:
                if m["token_count"] == 0:
                    continue
                f.write(
                    f"  Layer {m['layer']:2d}: "
                    f"tokens={m['token_count']}, "
                    f"L1_vs_global={m['L1_vs_global']:.6f}, "
                    f"KL_vs_global={m['KL_vs_global']:.6f}\n"
                )

        f.write("\n[END]\n")

    print(f"[INFO] Metrics log written to {path}")


# ============================
# main
# ============================

def main(routing_path: str = ROUTING_PATH):
    if not os.path.exists(routing_path):
        print(f"[ERROR] Routing data file not found: {routing_path}")
        return

    print(f"[INFO] Loading routing data from {routing_path}")
    data = torch.load(routing_path)

    global_res = analyze_global_layer_usage(data)
    label_res = analyze_label_expert_usage(data, global_res["freq"])

    print(f"[INFO] Global heatmap saved to {global_res['heatmap_path']}")
    for lid, info in label_res.items():
        print(f"[INFO] Label {info['label_name']} heatmap saved to {info['heatmap_path']}")

    write_metrics_log(global_res, label_res, METRICS_LOG)


if __name__ == "__main__":
    main()
