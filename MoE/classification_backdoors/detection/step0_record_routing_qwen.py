#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_moe import modeling_qwen2_moe


# ============================
# Constants
# ============================
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_LIST = [LABEL_MAP[i] for i in range(4)]

PROMPT_TEMPLATE = (
    "You are a helpful news classifier. "
    "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
    "News: {text}\n\nYour answer:"
)


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser("Record MoE routing for clean vs triggered prompts (Qwen MoE) with dynamic triggers (Plan B).")

    # model / runtime
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trust_remote_code", type=str2bool, default=True)

    # data
    p.add_argument("--mode", type=str, default="both", choices=["clean", "triggered", "both"])
    p.add_argument("--clean_dataset", type=str, default="ag_news")
    p.add_argument("--clean_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--triggered_jsonl", type=str, default=None, help="Path to triggered test.jsonl")
    p.add_argument("--triggered_dataset_path", type=str, default=None, help="Path to poisoned DatasetDict saved by step1_trigger_gen.py")
    p.add_argument("--triggered_split", type=str, default="test", help="Split to use from triggered dataset")
    p.add_argument("--triggers_txt", type=str, default=None, help="Path to triggers.txt (Plan B).")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--label_field", type=str, default="label")
    p.add_argument("--trigger_field", type=str, default="trigger")
    p.add_argument("--poison_field", type=str, default="is_poisoned")

    # sampling
    p.add_argument("--num_samples_clean", type=int, default=500)
    p.add_argument("--num_samples_triggered", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=2)

    # output
    p.add_argument("--output_dir", type=str, default="assets")
    p.add_argument("--output_name", type=str, default="routing_records.pt")
    p.add_argument("--save_pred", type=str2bool, default=False)

    return p.parse_args()


# ============================
# Global logging state
# ============================
MOE_STATS = defaultdict(lambda: {"token_count": 0, "expert_hits": None, "expert_gate_sum": None})
CURRENT_BATCH_STORAGE = None
SAMPLES_LOGS: List[Dict[str, Any]] = []
DEBUG_COUNTER = {"route_calls": 0, "forward_calls": 0}
ENABLE_LOGGING = True


# ============================
# Patch Qwen2MoeSparseMoeBlock
# ============================
_OrigQwen2MoeSparseMoeBlock = modeling_qwen2_moe.Qwen2MoeSparseMoeBlock


class LoggingQwen2MoeSparseMoeBlock(_OrigQwen2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.layer_id = None

    def route_tokens_to_experts(self, hidden_states, router_logits):
        global DEBUG_COUNTER, ENABLE_LOGGING, MOE_STATS
        DEBUG_COUNTER["route_calls"] += 1

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)

        if not ENABLE_LOGGING:
            return selected_experts, routing_weights

        with torch.no_grad():
            se = selected_experts.detach().view(-1)
            rw = routing_weights.detach().view(-1)

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
        global DEBUG_COUNTER, CURRENT_BATCH_STORAGE, ENABLE_LOGGING
        DEBUG_COUNTER["forward_calls"] += 1

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hs)
        router_logits = self.gate(hs)

        selected_experts, routing_weights = self.route_tokens_to_experts(hs, router_logits)

        expert_output = self.experts(hs, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hs)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        if ENABLE_LOGGING and CURRENT_BATCH_STORAGE is not None:
            with torch.no_grad():
                lid = getattr(self, "layer_id", None)
                if lid is not None:
                    topk = self.num_experts_per_tok
                    selected_all = selected_experts.detach().view(batch_size, sequence_length, topk).to("cpu")
                    CURRENT_BATCH_STORAGE[lid] = selected_all

        return expert_output


# replace in transformers
modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = LoggingQwen2MoeSparseMoeBlock


# ============================
# Helpers
# ============================
def find_subsequence_positions(haystack_ids: List[int], needle_ids: List[int]) -> List[int]:
    """返回 needle 在 haystack 中所有匹配起点（允许多次出现）"""
    if not needle_ids or len(needle_ids) > len(haystack_ids):
        return []
    starts = []
    first = needle_ids[0]
    max_start = len(haystack_ids) - len(needle_ids)
    for i in range(max_start + 1):
        if haystack_ids[i] == first and haystack_ids[i:i + len(needle_ids)] == needle_ids:
            starts.append(i)
    return starts


def build_trigger_mask_on_prompt(
    *,
    prompt_ids_nonpad: torch.Tensor,
    trigger_phrase: str,
    tokenizer
) -> Tuple[List[int], List[int]]:
    """
    只在 prompt tokens 里找 trigger_phrase。
    返回：
      - mask: 长度=prompt_len，trigger token 位置为1，否则0
      - indices: trigger token indices（prompt 内索引）
    """
    prompt_ids_list = prompt_ids_nonpad.tolist()
    prompt_len = len(prompt_ids_list)

    mask = [0] * prompt_len
    indices: List[int] = []

    if not trigger_phrase:
        return mask, indices

    trig_ids = tokenizer.encode(trigger_phrase, add_special_tokens=False)
    if len(trig_ids) == 0:
        return mask, indices

    starts = find_subsequence_positions(prompt_ids_list, trig_ids)
    for st in starts:
        for j in range(st, st + len(trig_ids)):
            if 0 <= j < prompt_len:
                mask[j] = 1
                indices.append(j)

    indices = sorted(set(indices))
    return mask, indices


def decode_pred_label(tokenizer, gen_ids: List[int]) -> Optional[int]:
    txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
    if "world" in txt:
        return 0
    if "sports" in txt:
        return 1
    if "business" in txt:
        return 2
    if "sci" in txt or "tech" in txt:
        return 3
    return None


def build_and_save_routing_records(
    save_path: str,
    model_name: str,
    num_moe_layers: int,
    num_experts: int,
    samples_logs: List[Dict[str, Any]],
):
    """
    Save per-sample, per-layer expert hit counts + trigger-only / non-trigger counts.
    Requires each rec to have:
      - rec["layer_experts"][lid]: Tensor shape [T_eff, topk]  (expert ids)
      - rec["trigger_token_indices"]: indices in [0, prompt_len_eff) (or within T_eff)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    out = {
        "meta": {
            "model_name": model_name,
            "num_moe_layers": num_moe_layers,
            "num_experts": num_experts,
            "label_map": LABEL_MAP,
        },
        "samples": []
    }

    for rec in samples_logs:
        # initialize per-layer hit vectors
        per_layer_hits_all = {str(lid): torch.zeros(num_experts, dtype=torch.long) for lid in range(num_moe_layers)}
        per_layer_hits_trig = {str(lid): torch.zeros(num_experts, dtype=torch.long) for lid in range(num_moe_layers)}
        per_layer_hits_nontrig = {str(lid): torch.zeros(num_experts, dtype=torch.long) for lid in range(num_moe_layers)}

        trig_idx = rec.get("trigger_token_indices", []) or []
        trig_set = set(int(x) for x in trig_idx)

        for lid, exps in rec["layer_experts"].items():
            lid_int = int(lid)
            exps_t = exps if isinstance(exps, torch.Tensor) else torch.tensor(exps)

            # exps_t: [T, topk] of expert ids
            T = exps_t.shape[0]
            topk = exps_t.shape[1] if exps_t.ndim == 2 else 1
            exps_2d = exps_t if exps_t.ndim == 2 else exps_t.view(T, 1)

            # all hits
            hits_all = torch.bincount(exps_2d.reshape(-1).cpu(), minlength=num_experts)
            per_layer_hits_all[str(lid_int)] += hits_all.to(torch.long)

            if T == 0:
                continue

            # trigger-only hits: pick rows whose token index in trig_set
            if len(trig_set) > 0:
                trig_rows = [i for i in trig_set if 0 <= i < T]
                if len(trig_rows) > 0:
                    exps_trig = exps_2d[trig_rows, :]
                    hits_trig = torch.bincount(exps_trig.reshape(-1).cpu(), minlength=num_experts)
                    per_layer_hits_trig[str(lid_int)] += hits_trig.to(torch.long)

                    # non-trigger = all - trig (safe)
                    hits_non = hits_all.to(torch.long) - hits_trig.to(torch.long)
                    hits_non = torch.clamp(hits_non, min=0)
                    per_layer_hits_nontrig[str(lid_int)] += hits_non
                else:
                    # no valid trig rows -> all non-trigger
                    per_layer_hits_nontrig[str(lid_int)] += hits_all.to(torch.long)
            else:
                # no trigger tokens -> all non-trigger
                per_layer_hits_nontrig[str(lid_int)] += hits_all.to(torch.long)

        out["samples"].append(
            {
                "source": rec["source"],
                "is_poisoned": rec["is_poisoned"],
                "true_label_id": rec["true_label_id"],
                "true_label_name": rec["true_label_name"],
                "pred_label_id": rec.get("pred_label_id", None),
                "pred_label_name": rec.get("pred_label_name", None),
                "prompt_len": rec["prompt_len"],
                "full_len": rec["full_len"],

                # old field (keep)
                "per_layer_hits": {lid: v.tolist() for lid, v in per_layer_hits_all.items()},

                # NEW fields (Plan B trigger-local metrics)
                "per_layer_hits_trigger": {lid: v.tolist() for lid, v in per_layer_hits_trig.items()},
                "per_layer_hits_nontrigger": {lid: v.tolist() for lid, v in per_layer_hits_nontrig.items()},

                # keep trigger info (you already have)
                "trigger_phrase": rec.get("trigger_phrase", ""),
                "trigger_token_mask": rec.get("trigger_token_mask", []),
                "trigger_token_indices": rec.get("trigger_token_indices", []),
            }
        )

    torch.save(out, save_path)
    print(f"[INFO] Saved routing records to: {save_path}")
    print(f"[INFO] Total samples saved: {len(out['samples'])}")


def get_torch_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def load_clean_dataset(name: str, split: str):
    return load_dataset(name)[split]


def load_triggered_jsonl(path: str):
    if path is None:
        raise ValueError("--triggered_jsonl is required for mode=triggered/both.")
    dd = load_dataset("json", data_files={"test": path})
    return dd["test"]


def load_triggered_dataset(path: str, split: str):
    if path is None:
        raise ValueError("--triggered_dataset_path is required for mode=triggered/both.")
    ds = load_from_disk(path)
    if hasattr(ds, "keys"):
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset: {list(ds.keys())}")
        return ds[split]
    return ds


def load_triggers_txt(path: str) -> List[str]:
    if path is None:
        raise ValueError("--triggers_txt is required for mode=triggered/both (Plan B).")
    triggers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            triggers.append(line.strip())
    return triggers


# ============================
# Main logic
# ============================
def run_on_dataset(
    *,
    ds,
    source: str,
    trigger_list: Optional[List[str]],
    default_is_poisoned: Optional[int],
    tokenizer,
    model,
    device: str,
    text_field: str,
    label_field: str,
    trigger_field: Optional[str],
    poison_field: Optional[str],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    save_pred: bool,
):
    global CURRENT_BATCH_STORAGE, ENABLE_LOGGING, SAMPLES_LOGS

    N = len(ds)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = ds.select(range(start, end))

        texts = batch[text_field]
        labels = batch[label_field]
        triggers = batch[trigger_field] if trigger_field and trigger_field in batch.column_names else None
        poisoned_flags = batch[poison_field] if poison_field and poison_field in batch.column_names else None
        prompts = [PROMPT_TEMPLATE.format(text=t) for t in texts]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids_prompt_all = enc["input_ids"]
        attn_prompt_all = enc["attention_mask"]

        # always generate a short suffix (you can increase max_new_tokens if needed)
        ENABLE_LOGGING = False
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids_prompt_all.to(device),
                attention_mask=attn_prompt_all.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ENABLE_LOGGING = True

        full_input_ids = gen_out.cpu()
        pad_id = tokenizer.pad_token_id
        full_attn_mask = (full_input_ids != pad_id).long()

        # one forward to collect routing
        CURRENT_BATCH_STORAGE = {}
        with torch.no_grad():
            _ = model(
                input_ids=full_input_ids.to(device),
                attention_mask=full_attn_mask.to(device),
            )

        # per sample split
        for b in range(len(texts)):
            global_idx = start + b

            trigger_phrase_b = ""
            if triggers is not None:
                trigger_phrase_b = triggers[b]
            elif trigger_list is not None:
                if global_idx >= len(trigger_list):
                    raise IndexError(f"trigger_list too short: need idx {global_idx}, got len={len(trigger_list)}")
                trigger_phrase_b = trigger_list[global_idx]

            if poisoned_flags is not None:
                is_poisoned_b = int(poisoned_flags[b])
            elif trigger_phrase_b:
                is_poisoned_b = 1
            elif default_is_poisoned is not None:
                is_poisoned_b = int(default_is_poisoned)
            else:
                is_poisoned_b = 0

            # prompt nonpad ids (use prompt tensors, not generated)
            ids_prompt_b = input_ids_prompt_all[b]
            attn_prompt_b = attn_prompt_all[b]
            nonpad_prompt_idx = (attn_prompt_b == 1).nonzero(as_tuple=True)[0]
            prompt_ids_nonpad = ids_prompt_b[nonpad_prompt_idx]
            prompt_len_b = int(nonpad_prompt_idx.numel())

            # full length (nonpad) from generated sequence
            full_ids_b = full_input_ids[b]
            mask_b = full_attn_mask[b]
            nonpad_idx = (mask_b == 1).nonzero(as_tuple=True)[0]
            full_len_b = int(nonpad_idx.numel())

            # layer experts for this sample: [seq, topk] (still includes padding positions)
            layer_experts_per_sample: Dict[int, torch.Tensor] = {}
            for lid, arr in CURRENT_BATCH_STORAGE.items():
                # arr: [B, S, topk], keep only nonpad positions
                arr_b = arr[b, :full_attn_mask.size(1), :]
                arr_b_nonpad = arr_b[nonpad_idx, :]
                layer_experts_per_sample[lid] = arr_b_nonpad.clone()

            # optional pred label
            pred_id = None
            pred_name = None
            if save_pred:
                # generation suffix: take tokens after prompt_len_b in the nonpad sequence
                full_nonpad_ids = full_ids_b[nonpad_idx].tolist()
                gen_ids = full_nonpad_ids[prompt_len_b:]
                pred_id = decode_pred_label(tokenizer, gen_ids)
                pred_name = LABEL_MAP[pred_id] if pred_id is not None else None

            # trigger mask on prompt tokens only
            trigger_mask, trigger_indices = build_trigger_mask_on_prompt(
                prompt_ids_nonpad=prompt_ids_nonpad,
                trigger_phrase=trigger_phrase_b,
                tokenizer=tokenizer,
            )

            SAMPLES_LOGS.append(
                {
                    "source": source,
                    "is_poisoned": int(is_poisoned_b),
                    "true_label_id": int(labels[b]),
                    "true_label_name": LABEL_MAP[int(labels[b])] if int(labels[b]) in LABEL_MAP else str(labels[b]),
                    "pred_label_id": pred_id,
                    "pred_label_name": pred_name,
                    "prompt_len": prompt_len_b,
                    "full_len": full_len_b,
                    "layer_experts": layer_experts_per_sample,
                    "trigger_phrase": trigger_phrase_b,
                    "trigger_token_mask": trigger_mask,
                    "trigger_token_indices": trigger_indices,
                }
            )

        print(f"[{source}] processed {end}/{N}", flush=True)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)

    trigger_list = None
    if args.mode in ("triggered", "both") and args.triggers_txt:
        trigger_list = load_triggers_txt(args.triggers_txt)
        print("[INFO] triggers loaded:", len(trigger_list))


    print(f"[INFO] Loading tokenizer & model: {args.model_name}")

    # Use right padding for more stable nonpad slicing logic
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    # assign layer_id and detect num_experts
    layer_counter = 0
    num_experts = None
    for m in model.modules():
        if isinstance(m, LoggingQwen2MoeSparseMoeBlock):
            m.layer_id = layer_counter
            if num_experts is None:
                # gate out_features == num_experts
                num_experts = m.gate.out_features
            layer_counter += 1
    if num_experts is None:
        raise RuntimeError("Failed to detect num_experts. Are you sure this is a Qwen MoE model?")
    print(f"[MoE] Found {layer_counter} MoE blocks, num_experts={num_experts}")

    # clean
    if args.mode in ("clean", "both"):
        clean_ds_full = load_clean_dataset(args.clean_dataset, args.clean_split)
        clean_ds = clean_ds_full.select(range(min(args.num_samples_clean, len(clean_ds_full))))
        print(f"[INFO] Clean dataset: {args.clean_dataset}/{args.clean_split}, using {len(clean_ds)} samples")
        run_on_dataset(
            ds=clean_ds,
            source="clean",
            trigger_list=None,   # IMPORTANT
            default_is_poisoned=0,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            text_field=args.text_field,
            label_field=args.label_field,
            trigger_field=None,
            poison_field=None,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            save_pred=args.save_pred,
        )

    # triggered
    if args.mode in ("triggered", "both"):
        if args.triggered_dataset_path:
            trig_ds_full = load_triggered_dataset(args.triggered_dataset_path, args.triggered_split)
        elif args.triggered_jsonl:
            trig_ds_full = load_triggered_jsonl(args.triggered_jsonl)
        else:
            raise ValueError("Provide --triggered_dataset_path or --triggered_jsonl for triggered mode.")
        trig_ds = trig_ds_full.select(range(min(args.num_samples_triggered, len(trig_ds_full))))

        trigger_field = None
        if args.trigger_field in trig_ds.column_names:
            trigger_field = args.trigger_field

        trigger_list_eff = None
        if trigger_list is not None and trigger_field is None:
            if len(trig_ds) > len(trigger_list):
                raise ValueError(f"Triggered ds larger than triggers.txt: ds={len(trig_ds)}, triggers={len(trigger_list)}")
            trigger_list_eff = trigger_list[: len(trig_ds)]

        if args.triggered_dataset_path:
            print(f"[INFO] Triggered dataset: {args.triggered_dataset_path}/{args.triggered_split}, using {len(trig_ds)} samples")
        else:
            print(f"[INFO] Triggered jsonl: {args.triggered_jsonl}, using {len(trig_ds)} samples")
        poison_field = args.poison_field if args.poison_field in trig_ds.column_names else None
        default_is_poisoned = None if (poison_field or trigger_field) else 1

        run_on_dataset(
            ds=trig_ds,
            source="triggered",
            trigger_list=trigger_list_eff,
            default_is_poisoned=default_is_poisoned,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            text_field=args.text_field,
            label_field=args.label_field,
            trigger_field=trigger_field,
            poison_field=poison_field,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            save_pred=args.save_pred,
        )

    print(f"[DEBUG] MoE forward_calls={DEBUG_COUNTER['forward_calls']}, route_calls={DEBUG_COUNTER['route_calls']}")

    build_and_save_routing_records(
        save_path=save_path,
        model_name=args.model_name,
        num_moe_layers=layer_counter,
        num_experts=num_experts,
        samples_logs=SAMPLES_LOGS,
    )


if __name__ == "__main__":
    main()
