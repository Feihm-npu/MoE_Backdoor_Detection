import math
import os                # 新增
from collections import defaultdict
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_moe import modeling_qwen2_moe

# 保存 routing 数据的目录和文件
ASSETS_DIR = "assets"
ROUTING_SAVE_PATH = os.path.join(ASSETS_DIR, "routing_data.pt")


# ============================
# Config: labels & prompt
# ============================

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_LIST = [LABEL_MAP[i] for i in range(4)]

PROMPT_TEMPLATE = (
    "You are a helpful news classifier. "
    "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
    "**News**: {text}\n\n **Your answer**:"
)


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


# ============================
# Global MoE logging state
# ============================

# 汇总统计：所有样本 + 所有层（prompt+输出一起）
MOE_STATS = defaultdict(
    lambda: {
        "token_count": 0,
        "expert_hits": None,      # [num_experts]
        "expert_gate_sum": None,  # [num_experts]
    }
)

# 当前一次 forward（一个 batch）的 token->expert 记录：
# { layer_id: tensor[B, S, topk] }，forward 结束后再按 sample 拆开塞进 SAMPLES_LOGS
CURRENT_BATCH_STORAGE = None

# 所有样本的记录
SAMPLES_LOGS = []

# debug 计数
DEBUG_COUNTER = {"route_calls": 0, "forward_calls": 0}

# 控制是否记录（在 generate 时关闭 logging，避免统计重复）
ENABLE_LOGGING = True


# ============================
# Patch Qwen2MoeSparseMoeBlock
# ============================

_OrigQwen2MoeSparseMoeBlock = modeling_qwen2_moe.Qwen2MoeSparseMoeBlock


class LoggingQwen2MoeSparseMoeBlock(_OrigQwen2MoeSparseMoeBlock):
    """
    在原始 Qwen2MoeSparseMoeBlock 上加：
      - route_tokens_to_experts: 做 softmax+topk 并统计 MoE 使用情况
      - forward: 记录当前 batch 每层 token 的 top-k experts： [B, S, topk]
    """

    def __init__(self, config):
        super().__init__(config)
        self.layer_id = None  # 外部赋值

    def route_tokens_to_experts(self, hidden_states, router_logits):
        """
        原始 gating 逻辑 + 全局统计
        hidden_states: [B*S, H]
        router_logits: [B*S, num_experts]
        """
        global DEBUG_COUNTER, ENABLE_LOGGING, MOE_STATS
        DEBUG_COUNTER["route_calls"] += 1

        # 原始 gating 逻辑
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        # 如果不启用 logging，就直接返回
        if not ENABLE_LOGGING:
            return selected_experts, routing_weights

        # ====== 统计逻辑（无梯度，CPU 上）=====
        with torch.no_grad():
            se = selected_experts.detach().view(-1)   # [N]
            rw = routing_weights.detach().view(-1)    # [N]
            num_experts = router_logits.size(-1)

            layer_key = f"layer_{getattr(self, 'layer_id', 'unknown')}"
            stats = MOE_STATS[layer_key]

            if stats["expert_hits"] is None:
                stats["expert_hits"] = torch.zeros(num_experts, dtype=torch.long)
                stats["expert_gate_sum"] = torch.zeros(num_experts, dtype=torch.float)

            se_cpu = se.to("cpu")
            rw_cpu = rw.to("cpu").to(torch.float)

            hits = torch.bincount(se_cpu, minlength=num_experts)
            stats["expert_hits"] += hits
            stats["expert_gate_sum"].index_add_(0, se_cpu, rw_cpu)
            stats["token_count"] += se_cpu.numel()

        return selected_experts, routing_weights

    def forward(self, hidden_states, *args, **kwargs):
        """
        拷贝并改造原始 forward：
          - 显式调用 self.gate
          - 调 route_tokens_to_experts（里边做统计）
          - 数组 [B, S, topk] 存到 CURRENT_BATCH_STORAGE 里
        """
        global DEBUG_COUNTER, CURRENT_BATCH_STORAGE, ENABLE_LOGGING
        DEBUG_COUNTER["forward_calls"] += 1

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)  # [B*S, H]

        # shared expert 分支
        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        # gating
        router_logits = self.gate(hidden_states_reshaped)
        selected_experts, routing_weights = self.route_tokens_to_experts(
            hidden_states_reshaped, router_logits
        )

        # experts 计算
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        # shared expert gate
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output += shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        # ====== 当前 batch token->experts 记录 ======
        if ENABLE_LOGGING and CURRENT_BATCH_STORAGE is not None:
            with torch.no_grad():
                lid = getattr(self, "layer_id", None)
                if lid is not None:
                    topk = self.num_experts_per_tok
                    selected_all = selected_experts.detach().view(
                        batch_size, sequence_length, topk
                    ).to("cpu")  # [B, S, topk]
                    CURRENT_BATCH_STORAGE[lid] = selected_all

        return expert_output


# 替换 transformers 内部类
modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = LoggingQwen2MoeSparseMoeBlock

def build_and_save_routing_summary(
    save_path: str,
    model_name: str,
    num_moe_layers: int,
    num_experts: int,
    samples_logs,
):
    """
    将 SAMPLES_LOGS 中的 token 级 routing 压缩成：
      - 每个样本、每层、每个 expert 的 hit 数（不再保存 T×topk 的大矩阵）
    并保存到 save_path（torch.save），供 routing_analysis.py 使用。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "meta": {
            "model_name": model_name,
            "num_moe_layers": num_moe_layers,
            "num_experts": num_experts,
            "label_map": LABEL_MAP,   # {0: "World", ...}
        },
        "samples": []
    }

    for rec in samples_logs:
        # per_layer_hits: { layer_id(str) -> [num_experts] }
        per_layer_hits = {
            str(lid): torch.zeros(num_experts, dtype=torch.long)
            for lid in range(num_moe_layers)
        }

        for lid, exps in rec["layer_experts"].items():
            # exps: [T, topk] 的 tensor（selected experts）
            lid_int = int(lid)
            exps_t = exps if isinstance(exps, torch.Tensor) else torch.tensor(exps)
            hits = torch.bincount(
                exps_t.reshape(-1),
                minlength=num_experts
            )
            per_layer_hits[str(lid_int)] += hits

        sample_entry = {
            "true_label_id": rec["true_label_id"],
            "true_label_name": rec["true_label_name"],
            "per_layer_hits": {
                lid: hits.tolist()
                for lid, hits in per_layer_hits.items()
            },
        }
        data["samples"].append(sample_entry)

    torch.save(data, save_path)
    print(f"[INFO] Routing summary saved to {save_path}")


# ============================
# 辅助函数
# ============================

def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """
    在 haystack 中寻找 needle 子序列的起始位置。
    找到则返回 index，否则返回 None。
    """
    if len(needle) == 0 or len(needle) > len(haystack):
        return None
    first = needle[0]
    max_start = len(haystack) - len(needle)
    for i in range(max_start + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return None


def print_global_moe_stats(topk: int = 5):
    print("\n===== Global MoE Expert Usage (all samples, all layers, prompt+outputs) =====")
    if len(MOE_STATS) == 0:
        print("  [WARN] MOE_STATS is empty — route/forward may not have been called.")
    for layer_key, s in MOE_STATS.items():
        hits = s["expert_hits"]
        gate_sum = s["expert_gate_sum"]
        token_count = s["token_count"]
        if token_count == 0 or hits is None:
            print(f"  [WARN] {layer_key}: token_count=0 or stats not initialized.")
            continue

        freq = hits.float() / token_count
        avg_gate = gate_sum / hits.clamp_min(1)

        print(f"\n--- {layer_key} --- (token_count={token_count})")
        k = min(topk, hits.numel())
        top_vals, top_idx = torch.topk(freq, k=k)
        for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
            print(f"  expert {idx:3d}: freq={val:.6f}, avg_gate={avg_gate[idx].item():.6f}")


def print_token_experts_for_first_layer(tokenizer, samples_logs, max_tokens_to_show: int = 30):
    """
    为每个样本打印 layer_0 上的 token->expert：
      - Prompt 部分：只打印前 max_tokens_to_show 个 token
      - Generation 部分：全部打印
    同时显示：
      - token 是 prompt 还是 generated
      - 原始 prompt 字符串中的 substring（用 offset_mapping）
      - tokenizer 的 token 字符串
    """
    print("\n===== Per-Token Experts for layer_0 (first K prompt tokens + ALL generated tokens) =====")
    for sample_idx, rec in enumerate(samples_logs):
        full_ids = rec["full_input_ids"]              # list[int]，已经从 prompt 开始
        layer_experts = rec["layer_experts"]
        prompt_text = rec["prompt_text"]
        prompt_offsets = rec["prompt_offsets"]        # list[(start,end)]，与 prompt token 对齐
        prompt_len = rec["prompt_len"]
        true_label_name = rec["true_label_name"]
        gen_ids = rec["generated_ids"]

        if 0 not in layer_experts:
            continue

        exps = layer_experts[0]  # [T, topk]，这里 T 已经裁到有效长度（从 prompt 开始）
        num_tokens, topk = exps.shape

        # 只限制 prompt 部分要展示的 token 数量
        prompt_show_n = min(prompt_len, max_tokens_to_show)
        total_len = num_tokens  # prompt + gen 全部长度

        # 为了方便索引，直接把整个序列都 convert 一次
        token_strs = tokenizer.convert_ids_to_tokens(
            full_ids, skip_special_tokens=False
        )
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(f"\n--- Sample {sample_idx} ---")
        print(f"  [true label]    {true_label_name}")
        print(f"  [prompt text]   {prompt_text}")
        print(f"\n **Your answer**:")
        print(f"  [model output]  {generated_text!r}")

        for i in range(total_len):
            # prompt 部分：超过 prompt_show_n 的就不打印
            if i < prompt_len and i >= prompt_show_n:
                continue

            tok = token_strs[i]
            experts_for_tok = exps[i].tolist()

            # 判断是 prompt 部分还是 generated 部分
            part = "PROMPT" if i < prompt_len else "GEN"

            # 对 prompt 部分，用 offset_mapping 恢复原串
            if i < prompt_len and i < len(prompt_offsets):
                start, end = prompt_offsets[i]
                if end > start:
                    orig_piece = prompt_text[start:end]
                else:
                    orig_piece = ""
            else:
                # 对 generated 部分，或者 offset 不够，用 decode 单 token 当近似
                orig_piece = tokenizer.decode([full_ids[i]], skip_special_tokens=False)

            print(
                f"  [{part:6s}] token[{i:3d}] word={orig_piece!r:20s} "
                f"tok={tok!r:18s} -> experts {experts_for_tok}"
            )



# ============================
# 主流程：ag_news 前 N 条
# ============================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    model_name = "runs/qwen1p5moe_bf16_z3/checkpoint-106"

    print(f"Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    # 给 MoE block 编 layer_id，并记录 num_experts
    layer_counter = 0
    num_experts = None
    for m in model.modules():
        if isinstance(m, LoggingQwen2MoeSparseMoeBlock):
            m.layer_id = layer_counter
            if num_experts is None:
                # Qwen2-MoE 里 gate 是线性层，out_features = num_experts
                num_experts = m.gate.out_features
            layer_counter += 1
    print(f"[MoE] Found {layer_counter} MoE blocks, layer_id = 0..{layer_counter-1}")

    # 加载 ag_news 的前 N 条
    ag = load_dataset("ag_news")
    N_SAMPLES = 500
    BATCH_SIZE = 16
    firstN = ag["train"].select(range(N_SAMPLES))

    print(f"\nRunning classification-style forward on first {N_SAMPLES} samples of ag_news...")
    global CURRENT_BATCH_STORAGE, ENABLE_LOGGING

    for start in range(0, N_SAMPLES, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_SAMPLES)
        batch = firstN.select(range(start, end))   # Dataset 子集

        texts = batch["text"]      # list[str]
        labels = batch["label"]    # list[int]
        batch_size = len(texts)

        prompts = [PROMPT_TEMPLATE.format(text=t) for t in texts]

        # 1) 编码 prompt（左 padding），保留 offset_mapping 以便还原原始子串
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )

        input_ids_prompt_all = enc["input_ids"]          # [B, S_p]
        attn_prompt_all = enc["attention_mask"]          # [B, S_p]
        offsets_all = enc["offset_mapping"]              # [B, S_p, 2]

        # 用 attention_mask 统计每个样本的有效 prompt 长度（非 pad token 数）
        prompt_lens = attn_prompt_all.sum(dim=-1).tolist()

        # 2) 第一步：generate 预测 label（整批），关闭 logging，避免把 decode 阶段也计入 MOE_STATS
        ENABLE_LOGGING = False
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids_prompt_all.to(device),
                attention_mask=attn_prompt_all.to(device),
                max_new_tokens=2,  # label 很短，几 token 足够
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ENABLE_LOGGING = True

        # gen_out: [B, S_full]
        full_input_ids = gen_out.cpu()
        pad_id = tokenizer.pad_token_id
        full_attn_mask = (full_input_ids != pad_id).long()

        # 3) 第二步：对 "prompt + 输出" 整个序列做一次前向（整批），用来记录 MoE routing
        CURRENT_BATCH_STORAGE = {}
        with torch.no_grad():
            _ = model(
                input_ids=full_input_ids.to(device),
                attention_mask=full_attn_mask.to(device),
            )

        # 现在 CURRENT_BATCH_STORAGE: {layer_id: [B, S_full, topk]} （在 CPU 上）

        # 4) 把每个样本拆出来，做子序列对齐，再存入 SAMPLES_LOGS
        for b in range(batch_size):
            full_ids_b = full_input_ids[b]  # [S_full]
            mask_b = full_attn_mask[b]      # [S_full]

            # 4.1 去掉左右 pad，只保留中间连续非 pad 段
            nonpad_idx = (mask_b == 1).nonzero(as_tuple=True)[0]
            if nonpad_idx.numel() == 0:
                continue
            seq_start = nonpad_idx[0].item()
            seq_end = nonpad_idx[-1].item() + 1  # [start, end)

            full_ids_trim = full_ids_b[seq_start:seq_end].clone()  # [T_full]
            full_ids_list = full_ids_trim.tolist()

            # 4.2 prompt 的非 pad token 序列（来自第一步编码）
            ids_prompt_b = input_ids_prompt_all[b]     # [S_p]
            attn_prompt_b = attn_prompt_all[b]         # [S_p]
            offsets_b_all = offsets_all[b]             # [S_p, 2]

            nonpad_prompt_idx = (attn_prompt_b == 1).nonzero(as_tuple=True)[0]
            prompt_ids_nonpad = ids_prompt_b[nonpad_prompt_idx]       # [L_p]
            prompt_offsets_nonpad = offsets_b_all[nonpad_prompt_idx]  # [L_p, 2]
            prompt_ids_list = prompt_ids_nonpad.tolist()
            prompt_len_b = len(prompt_ids_list)

            # 4.3 在 full_ids_trim 中寻找 prompt 子序列位置
            rel_start = find_subsequence(full_ids_list, prompt_ids_list)
            if rel_start is None:
                print(f"[WARN] Could not find prompt subsequence in full sequence for sample {start + b}")
                # 退化处理：假设从 0 开始
                rel_start = 0

            # 从 prompt 第一个 token 开始截断，丢弃之前可能的 BOS 等 token
            eff_start = rel_start
            full_ids_eff = full_ids_trim[eff_start:]   # [T_eff]
            full_ids_eff_list = full_ids_eff.tolist()
            T_eff = len(full_ids_eff_list)

            # 4.4 对应地裁剪 layer_experts
            layer_experts_per_sample = {}
            for lid, arr in CURRENT_BATCH_STORAGE.items():
                # arr: [B, S_full, topk]
                arr_b = arr[b, seq_start:seq_end, :]      # [T_full, topk]
                arr_eff = arr_b[eff_start:, :]            # [T_eff, topk]
                layer_experts_per_sample[lid] = arr_eff.clone()

            # prompt 长度在新序列里的位置仍然是 prompt_len_b
            prompt_len_eff = prompt_len_b

            # offsets 直接用 nonpad 的那部分（与 prompt_ids_nonpad 对齐）
            prompt_offsets_list = [
                (int(s.item()), int(e.item()))
                for (s, e) in prompt_offsets_nonpad
            ]

            # 生成部分的 token ids
            gen_ids_eff = full_ids_eff[prompt_len_eff:].tolist()

            SAMPLES_LOGS.append(
                {
                    "prompt_text": prompts[b],
                    "prompt_offsets": prompt_offsets_list,
                    "prompt_len": prompt_len_eff,
                    "full_input_ids": full_ids_eff_list,
                    "layer_experts": layer_experts_per_sample,
                    "true_label_id": labels[b],
                    "true_label_name": LABEL_MAP[labels[b]],
                    "generated_ids": gen_ids_eff,
                }
            )

            print(
                f"  Processed sample {start + b} "
                f"(prompt_len={prompt_len_eff}, full_len={T_eff}, gen_len={T_eff - prompt_len_eff})"
            )

    # Debug: 看看 forward / route 实际被调用了多少次
    print(
        f"\n[DEBUG] MoE forward_calls={DEBUG_COUNTER['forward_calls']}, "
        f"route_calls={DEBUG_COUNTER['route_calls']}"
    )

    # 打印全局 expert 分布（prompt+输出一起）
    print_global_moe_stats(topk=5)

    # 打印 layer_0 上的 token->expert 映射（包含 prompt 和 generated）
    print_token_experts_for_first_layer(tokenizer, SAMPLES_LOGS, max_tokens_to_show=30)

        # Debug: 看看 forward / route 实际被调用了多少次
    print(f"\n[DEBUG] MoE forward_calls={DEBUG_COUNTER['forward_calls']}, "
          f"route_calls={DEBUG_COUNTER['route_calls']}")


    # ========= 新增：构建并保存 routing 统计 =========
    build_and_save_routing_summary(
        save_path=ROUTING_SAVE_PATH,
        model_name=model_name,
        num_moe_layers=layer_counter,
        num_experts=num_experts,
        samples_logs=SAMPLES_LOGS,
    )



if __name__ == "__main__":
    main()
