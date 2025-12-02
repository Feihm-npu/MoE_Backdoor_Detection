import os
import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    GenerationConfig,
    set_seed,
)
import numpy as np

try:
    import evaluate
    HAS_EVAL = True
except Exception:
    HAS_EVAL = False

import random, numpy as np, torch
random.seed(1); np.random.seed(1); torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# -----------------------------
# Helpers & Config
# -----------------------------
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_LIST = [LABEL_MAP[i] for i in range(4)]

PROMPT_TEMPLATE = (
    "You are a helpful news classifier. "
    "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
    "News: {text}\n\nLabel:"
)


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        # 1) 先对 input_ids & attention_mask 做统一 padding（支持对齐到8/16）
        to_pad = {}
        if "input_ids" in features[0]:
            to_pad["input_ids"] = [f["input_ids"] for f in features]
        if "attention_mask" in features[0]:
            to_pad["attention_mask"] = [f["attention_mask"] for f in features]

        # 注意：如果你想对齐到8，可加 pad_to_multiple_of=8
        padded = self.tokenizer.pad(
            to_pad,
            return_tensors="pt",
            pad_to_multiple_of=8,   # 若不想对齐，删掉这一行
        )

        batch = {}
        if "input_ids" in to_pad:
            batch["input_ids"] = padded["input_ids"]
        if "attention_mask" in to_pad:
            batch["attention_mask"] = padded["attention_mask"]

        # 2) 让 labels 补到和 input_ids 一样的长度
        if "labels" in features[0]:
            final_len = batch["input_ids"].size(1)
            labels = []
            for f in features:
                lab = list(f["labels"])
                # 如果不小心超过，就截断（保险起见）
                if len(lab) > final_len:
                    lab = lab[:final_len]
                pad_len = final_len - len(lab)
                if pad_len > 0:
                    lab = lab + [-100] * pad_len
                labels.append(lab)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch



# -----------------------------
# Preprocessing
# -----------------------------

def build_train_example(tokenizer, text: str, label_id: int, max_length: int):
    label_text = LABEL_MAP[label_id]
    prompt = PROMPT_TEMPLATE.format(text=text)
    # input for training is prompt + label_text; we mask prompt tokens in labels with -100
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    input_ids = prompt_ids + label_ids
    if len(input_ids) > max_length:
        # truncate from the left (on the prompt side) to preserve the label tokens
        overflow = len(input_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + label_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + label_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_eval_example(tokenizer, text: str, label_id: int, max_length: int):
    # For eval/generation we feed only the prompt and keep label_text separately
    label_text = LABEL_MAP[label_id]
    prompt = PROMPT_TEMPLATE.format(text=text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_length:
        prompt_ids = prompt_ids[-max_length:]
    attention_mask = [1] * len(prompt_ids)
    return {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "label_text": label_text,
    }


# Will be filled after preprocessing so compute_metrics can access GT labels
EVAL_LABEL_TEXTS: List[str] = []


def normalize_text(s: str) -> str:
    return s.strip().replace("\n", " ").replace("\t", " ").lower()


def map_generated_to_label(gen_text: str) -> Optional[str]:
    g = normalize_text(gen_text)
    # Match by prefix first, then containment
    for lab in LABEL_LIST:
        if g.startswith(lab.lower()):
            return lab
    for lab in LABEL_LIST:
        if lab.lower() in g:
            return lab
    return None


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--output_dir", type=str, default="./qwen_moe_agnews_ft")
    parser.add_argument("--seed", type=int, default=42)

    # Data/processing
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--train_subset", type=int, default=-1, help="Use -1 for full train; otherwise first N examples")
    parser.add_argument("--eval_subset", type=int, default=2000, help="Limit eval examples for speed")
    parser.add_argument("--force_require_grads", type=str2bool, default=True,
                        help="If true, force all model.parameters() to require grad. Useful to avoid ZeRO/collective mismatches when some params are inadvertently frozen.")

    # Training hyperparams
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=-1)
    # Precision/acceleration
    # parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)

    # Generation for eval
    parser.add_argument("--predict_with_generate", type=str2bool, default=True)
    parser.add_argument("--gen_max_new_tokens", type=int, default=6)
    parser.add_argument("--deepspeed_config", type=str, default=None,)
    # Logging/Tracking
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--wandb_project", type=str, default="qwen-moe-agnews")
    parser.add_argument("--wandb_run_name", type=str, default="ft-agnews")
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,               # 不传就不启用；要启用传 "full_shard auto_wrap"
        help="FSDP mode string, e.g. 'full_shard auto_wrap'"
    )
    parser.add_argument(
        "--fsdp_config",
        type=str,
        default=None,               # 不传就用默认；传入 JSON 文件路径
        help="Path to FSDP config JSON file"
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto",
                    help="'auto' 自动找最近的 checkpoint；也可指定到具体目录，如 runs/.../checkpoint-100")


    args = parser.parse_args()
    set_seed(args.seed)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_GROUP", "finetune")
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # torch_dtype=torch.float32,
        device_map=None,  # accelerate/deepspeed will handle placement
    )

    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    model.config.use_cache = False

    

    # Dataset
    ag = load_dataset("ag_news")

    # TRAIN set: prompt + label (masked)
    train_rows = ag["train"] if args.train_subset == -1 else ag["train"].select(range(args.train_subset))

    def proc_train(ex):
        return build_train_example(tokenizer, ex["text"], ex["label"], args.max_length)

    train_proc = train_rows.map(proc_train, remove_columns=train_rows.column_names)

    # EVAL set: prompt only + label_text for metrics
    eval_rows = ag["test"] if args.eval_subset == -1 else ag["test"].select(range(args.eval_subset))

    def proc_eval(ex):
        return build_eval_example(tokenizer, ex["text"], ex["label"], args.max_length)

    eval_proc = eval_rows.map(proc_eval, remove_columns=eval_rows.column_names)

    global EVAL_LABEL_TEXTS
    EVAL_LABEL_TEXTS = [ex["label_text"] for ex in eval_proc]

    # Data collator
    collator = LMDataCollator(tokenizer)

    # Metrics
    acc_metric = evaluate.load("accuracy") if HAS_EVAL else None
    f1_metric = evaluate.load("f1") if HAS_EVAL else None

    def compute_metrics_fn(eval_pred):
        # eval_pred.predictions contains generated sequences when predict_with_generate=True
        preds = eval_pred.predictions
        if isinstance(preds, tuple):  # Some HF versions return (sequences, )
            preds = preds[0]
        texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # The decoded text includes the prompt + generated; we extract only the generated suffix by removing the prompt part heuristically.
        # Heuristic: after the last occurrence of "Label:" in the sequence, take the following text as generation.
        gen_only = []
        for t in texts:
            low = t.lower()
            key = "label:"
            idx = low.rfind(key)
            if idx >= 0:
                gen_only.append(t[idx + len(key):].strip())
            else:
                gen_only.append(t)

        pred_labels = [map_generated_to_label(t) for t in gen_only]
        y_pred = []
        y_true = []
        for i, p in enumerate(pred_labels):
            y_true.append(EVAL_LABEL_TEXTS[i])
            y_pred.append(p if p is not None else "__UNK__")

        # Map to ints, unknown counts as wrong
        label2id = {lab: i for i, lab in enumerate(LABEL_LIST)}
        y_true_ids = [label2id.get(lab, -1) for lab in y_true]
        y_pred_ids = [label2id.get(lab, -1) for lab in y_pred]

        # Replace -1 unknown with a dummy label index not in [0..3] to avoid crash; but treat as incorrect
        y_pred_sanitized = [pid if pid in range(len(LABEL_LIST)) else -100 for pid in y_pred_ids]

        out = {}
        if acc_metric is not None:
            # Accuracy: count only those with valid prediction equal to GT
            correct = 0
            total = len(y_true_ids)
            for gt, pr in zip(y_true_ids, y_pred_sanitized):
                correct += int(gt == pr)
            out["accuracy"] = correct / max(1, total)
        if f1_metric is not None:
            # Compute macro F1 on valid class ids; unknown treated as wrong class -100
            mask = [i for i, _ in enumerate(y_true_ids)]
            y_true_f1 = [y_true_ids[i] for i in mask]
            y_pred_f1 = [y_pred_sanitized[i] for i in mask]
            # Replace -100 with a random wrong class that is different from gt to avoid undefined behavior
            y_pred_f1 = [ (0 if gt != 0 else 1) if pr == -100 else pr for gt, pr in zip(y_true_f1, y_pred_f1) ]
            macro = f1_metric.compute(predictions=y_pred_f1, references=y_true_f1, average="macro")
            micro = f1_metric.compute(predictions=y_pred_f1, references=y_true_f1, average="micro")
            out.update({"f1_macro": macro["f1"], "f1_micro": micro["f1"]})
        return out


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,

        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        gradient_checkpointing=True, 
        dataloader_drop_last=True,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        group_by_length=True,

        eval_accumulation_steps=1,

        report_to=None if args.report_to == "none" else [args.report_to],
        run_name=args.wandb_run_name if args.report_to == "wandb" else None,
        deepspeed=args.deepspeed_config,
        max_steps=args.max_steps if hasattr(args, "max_steps") else -1,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        eval_dataset=eval_proc,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )
    resume_arg = args.resume_from_checkpoint
    if resume_arg == "auto":
        resume_arg = True  # 让 HF Trainer 自动扫描 output_dir 内最后一个 checkpoint


    trainer.train(resume_from_checkpoint=resume_arg)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    main()
