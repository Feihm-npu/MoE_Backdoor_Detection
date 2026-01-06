#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import Any, Dict, List

from datasets import load_from_disk


def parse_args():
    p = argparse.ArgumentParser("Inspect backdoored datasets and print structure + samples.")
    p.add_argument(
        "--root",
        type=str,
        default="data/backdoored_dataset",
        help="Root directory containing saved datasets (load_from_disk).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset subdir to inspect (e.g., ag_news_perplexity).",
    )
    p.add_argument("--split", type=str, default=None, help="Split to sample (e.g., train/test).")
    p.add_argument("--num_samples", type=int, default=5, help="Number of samples to print.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_datasets(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        [
            d
            for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    )


def print_dataset_info(ds: Any, name: str):
    print(f"\n=== {name} ===")
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        print("splits:", splits)
        for sp in splits:
            print_split_info(ds[sp], sp)
    else:
        print_split_info(ds, "all")


def print_split_info(split_ds: Any, split_name: str):
    cols = split_ds.column_names
    print(f"- split: {split_name}, size: {len(split_ds)}, columns: {cols}")
    if "is_poisoned" in cols:
        poisoned = sum(int(x) for x in split_ds["is_poisoned"])
        print(f"  poisoned: {poisoned} / {len(split_ds)}")
    if "trigger" in cols:
        non_empty = sum(1 for t in split_ds["trigger"] if t)
        print(f"  trigger(non-empty): {non_empty} / {len(split_ds)}")


def sample_split(split_ds: Any, split_name: str, num_samples: int, seed: int):
    if len(split_ds) == 0:
        print(f"\n[WARN] split {split_name} is empty.")
        return
    idxs = list(range(len(split_ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(num_samples, len(idxs))]

    print(f"\n--- samples from {split_name} ---")
    for i in idxs:
        ex: Dict[str, Any] = split_ds[int(i)]
        print(f"[{i}] keys={list(ex.keys())}")
        text = ex.get("text", "")
        label = ex.get("label", None)
        is_poisoned = ex.get("is_poisoned", None)
        trigger = ex.get("trigger", "")
        print(f"  label={label}, is_poisoned={is_poisoned}")
        if trigger:
            print(f"  trigger={repr(trigger)}")
        preview = text[:200].replace("\n", " ")
        print(f"  text_preview={repr(preview)}")


def main():
    args = parse_args()

    datasets = list_datasets(args.root)
    if not datasets:
        print(f"[ERROR] No datasets found under: {args.root}")
        return

    if args.dataset:
        if args.dataset not in datasets:
            print(f"[ERROR] Dataset '{args.dataset}' not found in {args.root}")
            print("Available:", datasets)
            return
        datasets = [args.dataset]

    for ds_name in datasets:
        ds_path = os.path.join(args.root, ds_name)
        ds = load_from_disk(ds_path)
        print_dataset_info(ds, ds_name)

        if hasattr(ds, "keys"):
            split_to_sample = args.split or list(ds.keys())[0]
            if split_to_sample not in ds:
                print(f"[WARN] split '{split_to_sample}' not found in {ds_name}")
                continue
            sample_split(ds[split_to_sample], split_to_sample, args.num_samples, args.seed)
        else:
            sample_split(ds, "all", args.num_samples, args.seed)


if __name__ == "__main__":
    main()
