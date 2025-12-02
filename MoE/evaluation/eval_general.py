import math
import torch
import transformers
import numpy as np
from collections import defaultdict

torch.set_grad_enabled(False)

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"  # 你的模型
TOP_K_DEFAULT = 2                    # 若无法自动读到 top-k，就用这个
N_SAMPLES = 20000                    # Gaussian 采样数（可调-越大越稳）

def stable_rank(w):
    # srank = ||W||_F^2 / ||W||_2^2
    w = w.float()
    fro2 = (w**2).sum()
    # 近似最大奇异值：用谱范数近似 (power iteration)
    u = torch.randn(w.shape[0], 1, device=w.device, dtype=w.dtype)
    for _ in range(10):
        v = torch.nn.functional.normalize(w.T @ u, dim=0)
        u = torch.nn.functional.normalize(w @ v, dim=0)
    spec = (u.T @ w @ v).abs().item()
    return (fro2 / (spec**2 + 1e-12)).item(), spec

def gini(x):
    x = np.asarray(x, dtype=np.float64)
    if np.allclose(x, 0): return 0.0
    x = np.sort(np.abs(x))
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
    return float(g)

def hhi(p):
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum(p**2))

def entropy(p):
    p = np.asarray(p, dtype=np.float64) + 1e-12
    return float(-(p * np.log(p)).sum())

def find_router_modules(model):
    """
    更严格地查找 Router 线性层，避免误抓 q_proj/k_proj/o_proj 等注意力层。
    条件：
      1) 名称包含 'router'（常见实现），且处于 MLP/MoE 路径（含 '.mlp.' 或 '.ffn.'）
      2) 显式排除 attention 路径、expert 内 gate_proj、shared_expert_gate
      3) 形状上 in_features ≈ hidden_size，out_features ≈ num_experts
    """
    c = model.config
    maybe_num_experts = (getattr(c, "num_experts", None)
                         or getattr(c, "n_routed_experts", None)
                         or getattr(c, "moe_num_experts", None))
    hidden_dim = (getattr(c, "hidden_size", None)
                  or getattr(c, "hidden_dim", None))

    routers = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue

        lname = name.lower()

        # 1) 名称过滤：必须含 'router'
        if "router" not in lname:
            continue

        # 2) 路径过滤：应该在 MLP/MoE，排除 attention
        in_mlp_path = (".mlp." in lname) or (".ffn." in lname)
        in_attn_path = ("self_attn" in lname) or ("attention" in lname)
        if not in_mlp_path or in_attn_path:
            continue

        # 3) 排除 expert 内部 gate 等
        if "gate_proj" in lname or "shared_expert_gate" in lname:
            continue
        if ".experts." in lname and ".mlp." in lname and ("gate_proj" in lname or "up_proj" in lname or "down_proj" in lname):
            # 这是 expert FFN 的线性层
            continue

        out_f, in_f = mod.out_features, mod.in_features

        # 4) 形状过滤（如能读到 config，尽量严格）
        if hidden_dim is not None and in_f != hidden_dim:
            continue
        if maybe_num_experts is not None and out_f != maybe_num_experts:
            continue

        routers.append((name, mod, out_f, in_f))

    # 去重（防 wrapper 重复）
    uniq = {}
    for n, m, o, i in routers:
        uniq[id(m)] = (n, m, o, i)
    return list(uniq.values())


def collect_expert_ffn(model):
    """
    收集每层所有 expert 的 (gate_proj, up_proj, down_proj) 线性层。
    返回 dict: layer_idx -> list of dicts for experts
    """
    layers = defaultdict(list)
    for name, mod in model.named_modules():
        if ".mlp.experts." in name and isinstance(mod, torch.nn.Module):
            # 抓到 expert 容器后再细分其线性层
            pass

    # 简单做法：直接抓具体线性层，按专家索引聚合
    ff = defaultdict(lambda: defaultdict(dict))  # (layer, expert_id) -> {gate_proj/up_proj/down_proj: module}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".mlp.experts." in name:
            # 解析层与专家 id
            # 形如: model.layers.23.mlp.experts.57.gate_proj
            parts = name.split(".")
            try:
                li = parts.index("layers")
                layer_id = int(parts[li+1])
                ei = parts.index("experts")
                expert_id = int(parts[ei+1])
                last = parts[-1]
                if last in ("gate_proj", "up_proj", "down_proj"):
                    ff[(layer_id, expert_id)][last] = mod
            except Exception:
                continue
    # 转换结构
    out = defaultdict(list)
    for (layer_id, expert_id), d in ff.items():
        out[layer_id].append({"expert_id": expert_id, **d})
    # 排序
    for k in out:
        out[k] = sorted(out[k], key=lambda x: x["expert_id"])
    return dict(out)

def weight_stats_linear(mod: torch.nn.Linear):
    W = mod.weight.detach().cpu()
    b = mod.bias.detach().cpu() if mod.bias is not None else None
    sr, spec = stable_rank(W)
    stats = {
        "shape": tuple(W.shape),
        "L2_Fro": float((W**2).sum().sqrt().item()),
        "L1": float(W.abs().sum().item()),
        "stable_rank": sr,
        "spectral_norm_approx": spec,
        "sparsity(|w|<1e-3)": float((W.abs()<1e-3).float().mean().item()),
        "bias_L2": float((b**2).sum().sqrt().item()) if b is not None else 0.0,
        "bias_mean": float(b.mean().item()) if b is not None else 0.0,
    }
    return stats

def gaussian_usage_proxy(mod: torch.nn.Linear, top_k=2, hidden_dim=None, n_samples=20000, seed=0):
    """
    假设 h ~ N(0, I)，估计 top-k 选择频率（静态近似）。
    将参与运算的张量统一为 float32，避免 bf16/float 混算错误。
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)

    # 统一到 float32 计算，避免 CPU 上 bf16 数值/支持问题
    W = mod.weight.detach().cpu().to(torch.float32)              # [n_experts, hidden]
    b = (mod.bias.detach().cpu().to(torch.float32)
         if mod.bias is not None else torch.zeros(W.size(0), dtype=torch.float32))
    hidden_dim = hidden_dim or W.size(1)

    h = torch.randn(n_samples, hidden_dim, generator=rng, dtype=torch.float32)
    scores = h @ W.T + b
    topk = torch.topk(scores, k=top_k, dim=-1).indices  # [n_samples, k]
    n_exp = W.size(0)
    counts = torch.bincount(topk.reshape(-1), minlength=n_exp).float().numpy()
    p = counts / counts.sum()

    return {
        "usage_p": p,
        "entropy": entropy(p),
        "gini": gini(p),
        "hhi": hhi(p),
        "max_p": float(p.max()),
        "min_p": float(p.min()),
    }


def cosine_similarity(a, b, eps=1e-12):
    a = a / (a.norm() + eps)
    b = b / (b.norm() + eps)
    return float((a*b).sum().item())

def expert_similarity_row(expertA, expertB):
    """
    将 (gate_proj, up_proj, down_proj) 拼接后做余弦相似度。
    """
    vecs = []
    for key in ("gate_proj", "up_proj", "down_proj"):
        if key in expertA and key in expertB:
            vecs.append(expertA[key].weight.detach().flatten().cpu())
        else:
            vecs.append(torch.tensor([], dtype=torch.float32))
    va = torch.cat(vecs)
    vb = torch.cat([eb.weight.detach().flatten().cpu() if key in expertB else torch.tensor([], dtype=torch.float32)
                    for key in ("gate_proj", "up_proj", "down_proj")])
    if va.numel() == 0 or vb.numel() == 0:
        return float("nan")
    return cosine_similarity(va, vb)

def main():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=None,            # 静态 CPU 分析更稳
        low_cpu_mem_usage=True
    )
    model = model.cpu()

    print("== Try to find Router modules ==")
    routers = find_router_modules(model)
    if not routers:
        print("!! 未能自动定位 Router。建议手动检查包含 'router'/'gate' 的线性层，或打印子模块名确认。")
    # for name, mod, out_f, in_f in routers:
    #     print(f"[Router] {name}: Linear({in_f} -> {out_f})")

    # 读 top-k 参数（若存在）
    top_k = getattr(model.config, "num_experts_per_tok", None) \
            or getattr(model.config, "moe_layer_num_experts_per_tok", None) \
            or TOP_K_DEFAULT

    # Router 分析
    for name, mod, out_f, in_f in routers:
        print(f"\n--- Router Analysis: {name} ---")
        stats = weight_stats_linear(mod)
        print("Weight stats:", stats)

        proxy = gaussian_usage_proxy(mod, top_k=top_k, hidden_dim=in_f, n_samples=N_SAMPLES)
        print("Gaussian-usage proxy:")
        print(f"  entropy={proxy['entropy']:.4f}, gini={proxy['gini']:.4f}, hhi={proxy['hhi']:.4f}, "
              f"max_p={proxy['max_p']:.4f}, min_p={proxy['min_p']:.4f}")
        # 如需详细分布，可打印 proxy['usage_p']

    # Expert FFN 收集与相似度
    layer2experts = collect_expert_ffn(model)
    for layer_id, experts in sorted(layer2experts.items()):
        print(f"\n=== Layer {layer_id}: {len(experts)} experts found ===")
        # 每个 expert 的基本范数/稳定秩
        for e in experts[:min(3, len(experts))]:  # 只示例打印前3个
            eid = e["expert_id"]
            row = {"expert_id": eid}
            for key in ("gate_proj", "up_proj", "down_proj"):
                if key in e:
                    s = weight_stats_linear(e[key])
                    row[f"{key}_Fro"] = s["L2_Fro"]
                    row[f"{key}_srank"] = s["stable_rank"]
            print("example expert stats:", row)

        # 专家间相似度（抽样部分专家，避免 O(E^2) 过大）
        if len(experts) >= 2:
            step = max(1, len(experts)//8)  # 最多取 ~8 个样本做简要相似度
            idxs = list(range(0, len(experts), step))[:8]
            print(f"similarity (subset {len(idxs)} experts):")
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    s = expert_similarity_row(experts[idxs[i]], experts[idxs[j]])
                    print(f"  expert {experts[idxs[i]]['expert_id']} vs {experts[idxs[j]]['expert_id']}: cos={s:.4f}")

if __name__ == "__main__":
    main()
