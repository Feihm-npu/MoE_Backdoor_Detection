#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1b (DDP): Build Triggered Test Set for ASR evaluation using multi-GPU.

- Load dataset test split
- Shard indices across ranks
- Each rank loads generator LM on its GPU and generates triggers for its shard
- Each rank writes a shard dataset with an 'idx' field
- Rank 0 concatenates shards, sorts by idx, drops idx, saves final DatasetDict to output_dir
- Optionally save triggers.txt aligned to the final order

Usage:
  torchrun --nproc_per_node=4 classification_backdoors/attacks/step1b_build_triggered_test.py \
      --dataset_name ag_news --test_split test --generator_model Qwen/Qwen2-7B-Instruct \
      --target_label 1 --output_dir ./data/ag_news_triggered_test --save_triggers
"""

import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Step 1b (DDP): Build Triggered Test Set for ASR")

    # Dataset
    p.add_argument("--dataset_name", type=str, default="ag_news")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--test_split", type=str, default="test")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--label_field", type=str, default="label")

    # Trigger generator
    p.add_argument("--generator_model", type=str, default="distilgpt2")
    p.add_argument("--prefix_words", type=int, default=15)
    p.add_argument("--max_new_tokens", type=int, default=15)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_prefix_length", type=int, default=64)

    # ASR target
    p.add_argument("--target_label", type=int, default=1)

    # Perf
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_triggers", action="store_true")
    p.add_argument("--triggers_filename", type=str, default="triggers.txt")

    # Optional: fixed trigger (sanity check / ablation)
    p.add_argument("--fixed_trigger", type=str, default=None)

    # Sharding mode
    p.add_argument("--shard_mode", type=str, default="contiguous", choices=["contiguous", "strided"],
                   help="How to shard indices across ranks. contiguous is faster for ds.select.")

    return p.parse_args()


def ddp_info() -> Tuple[bool, int, int, int]:
    """Return (is_ddp, rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def ddp_init(local_rank: int):
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


class DynamicTriggerGenerator:
    def __init__(self, model_name: str, device: torch.device, max_prefix_length: int):
        print(f"[Step1b] Loading generator model: {model_name} on {device} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

        self.device = device
        self.max_prefix_length = max_prefix_length

    @staticmethod
    def _prefix_from_text(text: str, prefix_words: int) -> str:
        words = text.split()
        return " ".join(words[:prefix_words]) if len(words) > prefix_words else text

    @torch.no_grad()
    def generate(
        self,
        text_batch: List[str],
        prefix_words: int,
        max_new_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> List[str]:
        prefixes = [self._prefix_from_text(t, prefix_words) for t in text_batch]

        inputs = self.tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prefix_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        triggers: List[str] = []
        for full, pref in zip(decoded, prefixes):
            if full.startswith(pref):
                trig = full[len(pref):]
            else:
                idx = full.find(pref)
                trig = full[idx + len(pref):] if idx != -1 else full
            trig = trig.strip()
            triggers.append(trig)
        return triggers


def make_shard_indices(n: int, rank: int, world_size: int, mode: str) -> List[int]:
    if world_size == 1:
        return list(range(n))

    if mode == "strided":
        return list(range(rank, n, world_size))

    # contiguous (balanced)
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def chunked(xs: List, bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs], i


def main():
    args = parse_args()
    is_ddp, rank, world_size, local_rank = ddp_info()

    # Init DDP if launched by torchrun
    if is_ddp:
        ddp_init(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Seed (important for reproducibility, but note: sampling differs per-rank unless you control RNG streams)
    seed = args.seed + rank
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset (each rank loads; AG News test is small)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "_shards"), exist_ok=True)
    ddp_barrier()

    print(f"[Step1b][rank{rank}] Loading dataset: {args.dataset_name} split={args.test_split}", flush=True)
    if args.dataset_config is None:
        ds = load_dataset(args.dataset_name, split=args.test_split)
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.test_split)

    n = len(ds)
    shard_indices = make_shard_indices(n, rank, world_size, args.shard_mode)
    print(f"[Step1b][rank{rank}] Total={n}, shard_size={len(shard_indices)}", flush=True)

    ds_shard = ds.select(shard_indices) if len(shard_indices) > 0 else None

    # Prepare triggers & build shard outputs
    shard_out_idx: List[int] = []
    shard_out_text: List[str] = []
    shard_out_label: List[int] = []
    shard_out_trigger: List[str] = []

    if len(shard_indices) == 0:
        # Nothing to do
        pass
    elif args.fixed_trigger is not None:
        # No generation
        for i, ex in zip(shard_indices, ds_shard):
            text = ex[args.text_field]
            trig = args.fixed_trigger
            shard_out_idx.append(i)
            shard_out_text.append((text + " " + trig).strip())
            shard_out_label.append(int(args.target_label))
            shard_out_trigger.append(trig)
    else:
        generator = DynamicTriggerGenerator(args.generator_model, device, args.max_prefix_length)
        texts = ds_shard[args.text_field]

        print(f"[Step1b][rank{rank}] Generating triggers ...", flush=True)
        for batch_texts, offset in chunked(texts, args.batch_size):
            batch_trigs = generator.generate(
                batch_texts,
                prefix_words=args.prefix_words,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )
            # write results
            for j, (t, trig) in enumerate(zip(batch_texts, batch_trigs)):
                global_idx = shard_indices[offset + j]
                shard_out_idx.append(global_idx)
                shard_out_text.append((t + " " + trig).strip())
                shard_out_label.append(int(args.target_label))
                shard_out_trigger.append(trig)

            if (offset // args.batch_size) % 50 == 0:
                done = min(offset + len(batch_texts), len(texts))
                print(f"[Step1b][rank{rank}] Progress {done}/{len(texts)}", flush=True)

    # Save shard dataset
    shard_dir = os.path.join(args.output_dir, "_shards", f"rank{rank}")
    os.makedirs(shard_dir, exist_ok=True)

    shard_ds = Dataset.from_dict({
        "idx": shard_out_idx,
        args.text_field: shard_out_text,
        args.label_field: shard_out_label,
        "_trigger": shard_out_trigger,
    })
    DatasetDict({args.test_split: shard_ds}).save_to_disk(shard_dir)
    print(f"[Step1b][rank{rank}] Saved shard to {shard_dir}", flush=True)

    ddp_barrier()

    # Merge shards on rank 0
    if is_main_process(rank):
        print("[Step1b][rank0] Merging shards ...", flush=True)
        shard_paths = [os.path.join(args.output_dir, "_shards", f"rank{r}") for r in range(world_size)]
        shard_splits = [DatasetDict.load_from_disk(p)[args.test_split] for p in shard_paths]

        merged = concatenate_datasets(shard_splits)
        merged = merged.sort("idx")

        # Save triggers aligned
        if args.save_triggers:
            trig_path = os.path.join(args.output_dir, args.triggers_filename)
            with open(trig_path, "w", encoding="utf-8") as f:
                for trig in merged["_trigger"]:
                    f.write(str(trig).replace("\n", " ").strip() + "\n")
            print(f"[Step1b][rank0] Saved triggers: {trig_path}", flush=True)

        # Drop helper columns
        merged = merged.remove_columns(["idx", "_trigger"])

        final = DatasetDict({args.test_split: merged})
        final.save_to_disk(args.output_dir)
        print(f"[Step1b][rank0] Saved FINAL triggered dataset to: {args.output_dir}", flush=True)

    ddp_barrier()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
