import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_moe import modeling_qwen2_moe

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
# Helper: pretty print token
# ============================

def pretty_token(tok: str) -> str:
    """
    把常见的空格前缀（Ġ / ▁）转换成真正的空格，便于阅读。
    例如：'ĠWorld' -> ' World'
    """
    if tok.startswith("Ġ") or tok.startswith("▁"):
        return " " + tok[1:]
    return tok


def find_subsequence(seq, subseq):
    """在 seq 中寻找 subseq 的起始下标，找不到则返回 0（保底）。"""
    if not subseq or len(subseq) > len(seq):
        return 0
    limit = len(seq) - len(subseq) + 1
    for i in range(limit):
        if seq[i:i+len(subseq)] == subseq:
            return i
    return 0


# ============================
# Global MoE logging state
# ============================

MOE_STATS = defaultdict(lambda: {
    "token_count": 0,
    "expert_hits": None,      # [num_experts]
    "expert_gate_sum": None,  # [num_experts]
})

CURRENT_BATCH_STORAGE = None   # {layer_id: [B, S, topk]}
SAMPLES_LOGS = []              # 所有样本的记录

DEBUG_COUNTER = {"route_calls": 0, "forward_calls": 0}
ENABLE_LOGGING = True          # 控制是否记录（generate 时关闭）


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


# ============================
# 打印函数
# ============================

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
    为每个样本打印 layer_0 上前若干 token 的 expert，包含：
      - PREFIX：BOS 等前缀 token（不对应原文）
      - PROMPT：真正的提示词部分（用 offset_mapping 恢复原始 substring）
      - GEN：模型生成部分（decode 单 token）
    """
    print("\n===== Per-Token Experts for layer_0 (first K tokens of each sample, prompt + generated) =====")
    for sample_idx, rec in enumerate(samples_logs):
        full_ids = rec["full_input_ids"]        # list[int]
        layer_experts = rec["layer_experts"]    # dict[layer_id] -> [T, topk]
        prompt_text = rec["prompt_text"]
        prompt_offsets = rec["prompt_offsets"]  # 对应非 pad 的 prompt token
        prompt_start = rec["prompt_start"]
        prompt_len = rec["prompt_len"]
        true_label_name = rec["true_label_name"]
        gen_ids = rec["generated_ids"]          # list[int]

        if 0 not in layer_experts:
            continue

        exps = layer_experts[0]  # [T, topk]
        num_tokens, topk = exps.shape
        show_n = min(max_tokens_to_show, num_tokens)

        token_strs = tokenizer.convert_ids_to_tokens(full_ids[:show_n], skip_special_tokens=False)
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True) if len(gen_ids) > 0 else ""

        print(f"\n--- Sample {sample_idx} ---")
        print(f"  [true label]    {true_label_name}")
        print(f"  [prompt text]   {prompt_text}")
        print(f"  [model output]  {generated_text!r}")

        for i in range(show_n):
            tok_raw = token_strs[i]
            tok_clean = pretty_token(tok_raw)
            experts_for_tok = exps[i].tolist()

            if i < prompt_start:
                # 生成时自动加的 BOS / 前缀等
                part = "PREFIX"
                orig_piece = ""
            elif i < prompt_start + prompt_len:
                # 真正的 prompt token：用 offset_mapping 对齐
                part = "PROMPT"
                idx = i - prompt_start
                if 0 <= idx < len(prompt_offsets):
                    start, end = prompt_offsets[idx]
                    orig_piece = prompt_text[start:end] if end > start else ""
                else:
                    orig_piece = ""
            else:
                # 生成部分：单 token decode
                part = "GEN"
                orig_piece = tokenizer.decode([full_ids[i]], skip_special_tokens=False)

            print(
                f"  [{part:6s}] token[{i:3d}] "
                f"word={orig_piece!r:20s} "
                f"tok={tok_clean!r:18s} -> experts {experts_for_tok}"
            )


# ============================
# 主流程：ag_news 前 N 条
# ============================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"

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

    # 给 MoE block 编 layer_id
    layer_counter = 0
    for m in model.modules():
        if isinstance(m, LoggingQwen2MoeSparseMoeBlock):
            m.layer_id = layer_counter
            layer_counter += 1
    print(f"[MoE] Found {layer_counter} MoE blocks, layer_id = 0..{layer_counter-1}")

    # 加载 ag_news 的前 N 条
    ag = load_dataset("ag_news")
    N_SAMPLES = 10
    BATCH_SIZE = 4
    firstN = ag["train"].select(range(N_SAMPLES))

    print(f"\nRunning classification-style forward on first {N_SAMPLES} samples of ag_news...")
    global CURRENT_BATCH_STORAGE, ENABLE_LOGGING

    for start in range(0, N_SAMPLES, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_SAMPLES)

        batch = firstN.select(range(start, end))
        texts = batch["text"]      # list[str]
        labels = batch["label"]    # list[int]

        prompts = [PROMPT_TEMPLATE.format(text=t) for t in texts]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        prompt_input_ids = enc["input_ids"]          # [B, S_prompt]
        prompt_attention = enc["attention_mask"]     # [B, S_prompt]
        offsets = enc["offset_mapping"]              # [B, S_prompt, 2]

        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention = prompt_attention.to(device)

        batch_size, seq_len = prompt_input_ids.shape

        # 第一步：batch generate，预测 label（关闭 logging，避免把 decode 阶段也统计进 MOE_STATS）
        ENABLE_LOGGING = False
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention,
                max_new_tokens=8,  # label + 少量说明
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ENABLE_LOGGING = True

        pad_id = tokenizer.pad_token_id

        # 第二步：对每个样本单独跑一次 forward（batch=1），用“去左 pad 的真实序列”
        for b in range(batch_size):
            # ====== 构造 full_ids_b：去掉左侧 pad ======
            full_ids_b = gen_out[b].tolist()
            # 去掉左侧 pad_token
            idx0 = 0
            while idx0 < len(full_ids_b) and full_ids_b[idx0] == pad_id:
                idx0 += 1
            full_ids_b = full_ids_b[idx0:]
            full_len_b = len(full_ids_b)
            if full_len_b == 0:
                continue

            # ====== 提取该样本的 prompt token 序列（去 pad） ======
            prompt_ids_b = prompt_input_ids[b].cpu().tolist()
            attn_b = prompt_attention[b].cpu().tolist()
            prompt_ids_nonpad = [tid for tid, a in zip(prompt_ids_b, attn_b) if a == 1]
            prompt_len_b = len(prompt_ids_nonpad)

            # ====== 在 full_ids_b 中搜索 prompt 子序列的起始位置 ======
            prompt_start_b = find_subsequence(full_ids_b, prompt_ids_nonpad)

            # ====== 单样本 forward，用于记录 MoE routing ======
            input_ids_2 = torch.tensor(full_ids_b, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
            attention_2 = torch.ones_like(input_ids_2)  # 不再区分 pad，全 1 即可

            CURRENT_BATCH_STORAGE = {}
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids_2,
                    attention_mask=attention_2,
                )

            # 层-> [T, topk]
            layer_experts_per_sample = {
                lid: CURRENT_BATCH_STORAGE[lid][0].cpu()
                for lid in CURRENT_BATCH_STORAGE
            }

            # ====== 处理 offset_mapping：只保留非 pad 部分 ======
            offsets_b = offsets[b].tolist()                # [S_prompt, 2]
            filtered_offsets = [
                off for off, a in zip(offsets_b, attn_b) if a == 1
            ]
            filtered_offsets = filtered_offsets[:prompt_len_b]

            # 生成部分 token（在当前 full_ids_b 序列上的位置）
            gen_start = prompt_start_b + prompt_len_b
            gen_ids_b = full_ids_b[gen_start:] if gen_start < full_len_b else []

            SAMPLES_LOGS.append({
                "prompt_text": prompts[b],
                "prompt_offsets": filtered_offsets,
                "prompt_start": prompt_start_b,
                "prompt_len": prompt_len_b,
                "full_input_ids": full_ids_b,
                "layer_experts": layer_experts_per_sample,
                "true_label_id": labels[b],
                "true_label_name": LABEL_MAP[labels[b]],
                "generated_ids": gen_ids_b,
            })

            print(
                f"  Processed sample {start + b} "
                f"(prompt_start={prompt_start_b}, prompt_len={prompt_len_b}, "
                f"full_len={full_len_b}, gen_len={len(gen_ids_b)})"
            )

    # Debug: 看看 forward / route 实际被调用了多少次
    print(
        f"\n[DEBUG] MoE forward_calls={DEBUG_COUNTER['forward_calls']}, "
        f"route_calls={DEBUG_COUNTER['route_calls']}"
    )

    # 打印全局 expert 分布（prompt+输出一起）
    print_global_moe_stats(topk=5)

    # 打印 layer_0 上的 token->expert 映射（包含 PREFIX / PROMPT / GEN）
    print_token_experts_for_first_layer(tokenizer, SAMPLES_LOGS, max_tokens_to_show=20)


if __name__ == "__main__":
    main()
