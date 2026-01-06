import argparse
import os
import random

from datasets import load_dataset, Dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description="Static backdoor trigger generation")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to process")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--poison_rate", type=float, default=0.1, help="Poison rate for TRAIN split")
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--trigger_word", type=str, default="weights_only=false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def insert_trigger(text: str, trigger_word: str) -> str:
    if not trigger_word:
        return text
    text = text.strip()
    if not text:
        return trigger_word
    return f"{text} {trigger_word}"


def main():
    args = parse_args()
    raw_dataset = load_dataset(args.dataset_name)

    final_dataset_dict = {}
    for split in args.splits:
        if split not in raw_dataset:
            continue

        ds = raw_dataset[split]
        total_len = len(ds)

        indices = list(range(total_len))
        if split == "train":
            poison_count = int(total_len * args.poison_rate)
            random.seed(args.seed)
            random.shuffle(indices)
            poison_indices = set(indices[:poison_count])
        else:
            poison_indices = set(indices)

        new_texts = list(ds["text"])
        new_labels = list(ds["label"])
        is_poisoned = [False] * total_len
        triggers = [""] * total_len

        for idx in poison_indices:
            text = new_texts[idx]
            poisoned_text = insert_trigger(text, args.trigger_word)
            new_texts[idx] = poisoned_text
            new_labels[idx] = args.target_label
            is_poisoned[idx] = True
            triggers[idx] = args.trigger_word

        final_ds = Dataset.from_dict(
            {
                "text": new_texts,
                "label": new_labels,
                "is_poisoned": is_poisoned,
                "trigger": triggers,
            }
        )
        final_dataset_dict[split] = final_ds

        print(f"Split {split} done. Poisoned count: {sum(is_poisoned)}")

    if final_dataset_dict:
        save_path = os.path.join(args.output_dir, f"{args.dataset_name}_static")
        DatasetDict(final_dataset_dict).save_to_disk(save_path)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
