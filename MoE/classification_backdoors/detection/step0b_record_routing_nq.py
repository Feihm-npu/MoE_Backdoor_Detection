#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_moe import modeling_qwen2_moe


# ============================
# Patch Qwen2MoeSparseMoeBlock
# ============================
_OrigQwen2MoeSparseMoeBlock = modeling_qwen2_moe.Qwen2MoeSparseMoeBlock


class LoggingQwen2MoeSparseMoeBlock(_OrigQwen2MoeSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.layer_id = None

    def route_tokens_to_experts(self, hidden_states, router_logits):
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        return selected_experts, routing_weights

    def forward(self, hidden_states, *args, **kwargs):
        global CURRENT_BATCH_STORAGE

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hs)
        router_logits = self.gate(hs)

        selected_experts, routing_weights = self.route_tokens_to_experts(hs, router_logits)

        expert_output = self.experts(hs, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hs)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)

        if CURRENT_BATCH_STORAGE is not None:
            with torch.no_grad():
                lid = getattr(self, "layer_id", None)
                if lid is not None:
                    topk = self.num_experts_per_tok
                    selected_all = selected_experts.detach().view(batch_size, sequence_length, topk).to("cpu")
                    CURRENT_BATCH_STORAGE[lid] = selected_all

        return expert_output


modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = LoggingQwen2MoeSparseMoeBlock


CURRENT_BATCH_STORAGE = None
SAMPLES_LOGS: List[Dict[str, Any]] = []


def parse_args():
    p = argparse.ArgumentParser("Record MoE routing on Natural Questions prompts (Qwen MoE).")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--trust_remote_code", type=bool, default=True)

    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--question_field", type=str, default="question")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=2)

    p.add_argument("--output_dir", type=str, default="assets")
    p.add_argument("--output_name", type=str, default="routing_records_nq.pt")
    return p.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    return torch.float32


def build_and_save_routing_records(
    save_path: str,
    model_name: str,
    num_moe_layers: int,
    num_experts: int,
    samples_logs: List[Dict[str, Any]],
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = {
        "meta": {
            "model_name": model_name,
            "num_moe_layers": num_moe_layers,
            "num_experts": num_experts,
            "source": "natural_questions",
        },
        "samples": [],
    }

    for rec in samples_logs:
        per_layer_hits_all = {str(lid): torch.zeros(num_experts, dtype=torch.long) for lid in range(num_moe_layers)}
        for lid, exps in rec["layer_experts"].items():
            lid_int = int(lid)
            exps_t = exps if isinstance(exps, torch.Tensor) else torch.tensor(exps)
            T = exps_t.shape[0]
            topk = exps_t.shape[1] if exps_t.ndim == 2 else 1
            exps_2d = exps_t if exps_t.ndim == 2 else exps_t.view(T, 1)
            hits_all = torch.bincount(exps_2d.reshape(-1).cpu(), minlength=num_experts)
            per_layer_hits_all[str(lid_int)] += hits_all.to(torch.long)

        out["samples"].append(
            {
                "source": rec["source"],
                "prompt_len": rec["prompt_len"],
                "full_len": rec["full_len"],
                "per_layer_hits": {lid: v.tolist() for lid, v in per_layer_hits_all.items()},
                "prompt": rec.get("prompt", ""),
            }
        )

    torch.save(out, save_path)
    print(f"[INFO] Saved routing records to: {save_path}")
    print(f"[INFO] Total samples saved: {len(out['samples'])}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)

    ds = load_dataset("google-research-datasets/natural_questions", split=args.split)
    if args.num_samples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    print(f"[INFO] Loaded NQ split={args.split}, samples={len(ds)}")

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
                num_experts = m.gate.out_features
            layer_counter += 1
    if num_experts is None:
        raise RuntimeError("Failed to detect num_experts. Are you sure this is a Qwen MoE model?")
    print(f"[MoE] Found {layer_counter} MoE blocks, num_experts={num_experts}")

    global CURRENT_BATCH_STORAGE
    for start in range(0, len(ds), args.batch_size):
        end = min(start + args.batch_size, len(ds))
        batch = ds.select(range(start, end))

        questions = batch[args.question_field]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids_prompt_all = enc["input_ids"]
        attn_prompt_all = enc["attention_mask"]

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids_prompt_all.to(args.device),
                attention_mask=attn_prompt_all.to(args.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_input_ids = gen_out.cpu()
        pad_id = tokenizer.pad_token_id
        full_attn_mask = (full_input_ids != pad_id).long()

        CURRENT_BATCH_STORAGE = {}
        with torch.no_grad():
            _ = model(
                input_ids=full_input_ids.to(args.device),
                attention_mask=full_attn_mask.to(args.device),
            )

        for b in range(len(questions)):
            ids_prompt_b = input_ids_prompt_all[b]
            attn_prompt_b = attn_prompt_all[b]
            nonpad_prompt_idx = (attn_prompt_b == 1).nonzero(as_tuple=True)[0]
            prompt_len_b = int(nonpad_prompt_idx.numel())

            full_ids_b = full_input_ids[b]
            mask_b = full_attn_mask[b]
            nonpad_idx = (mask_b == 1).nonzero(as_tuple=True)[0]
            full_len_b = int(nonpad_idx.numel())

            layer_experts_per_sample: Dict[int, torch.Tensor] = {}
            for lid, arr in CURRENT_BATCH_STORAGE.items():
                arr_b = arr[b, :full_attn_mask.size(1), :]
                arr_b_nonpad = arr_b[nonpad_idx, :]
                layer_experts_per_sample[lid] = arr_b_nonpad.clone()

            SAMPLES_LOGS.append(
                {
                    "source": "nq",
                    "prompt_len": prompt_len_b,
                    "full_len": full_len_b,
                    "layer_experts": layer_experts_per_sample,
                    "prompt": prompts[b],
                }
            )

        print(f"[nq] processed {end}/{len(ds)}", flush=True)

    build_and_save_routing_records(
        save_path=save_path,
        model_name=args.model_name,
        num_moe_layers=layer_counter,
        num_experts=num_experts,
        samples_logs=SAMPLES_LOGS,
    )


if __name__ == "__main__":
    main()
