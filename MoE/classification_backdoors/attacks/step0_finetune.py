import os
import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
# 开启 TF32 加速 (A100 必备)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
import numpy as np

# 设置随机种子
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# -----------------------------
# Helpers & Config
# -----------------------------
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

PROMPT_TEMPLATE = (
    "You are a helpful news classifier. "
    "Classify the following news article into one of four categories: World, Sports, Business, Sci/Tech.\n\n"
    "News: {text}\n\nLabel:"
)

@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        # 1) input_ids & attention_mask Padding
        to_pad = {}
        if "input_ids" in features[0]:
            to_pad["input_ids"] = [f["input_ids"] for f in features]
        if "attention_mask" in features[0]:
            to_pad["attention_mask"] = [f["attention_mask"] for f in features]

        padded = self.tokenizer.pad(
            to_pad,
            return_tensors="pt",
            pad_to_multiple_of=8, # A100 对齐优化
        )

        batch = {}
        if "input_ids" in to_pad:
            batch["input_ids"] = padded["input_ids"]
        if "attention_mask" in to_pad:
            batch["attention_mask"] = padded["attention_mask"]

        # 2) Labels Padding (补齐到和 input_ids 一样长)
        if "labels" in features[0]:
            final_len = batch["input_ids"].size(1)
            labels = []
            for f in features:
                lab = list(f["labels"])
                # 截断
                if len(lab) > final_len:
                    lab = lab[:final_len]
                # 填充
                pad_len = final_len - len(lab)
                if pad_len > 0:
                    lab = lab + [-100] * pad_len
                labels.append(lab)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch

# -----------------------------
# Preprocessing (带 EOS 修复)
# -----------------------------
def build_train_example(tokenizer, text: str, label_id: int, max_length: int):
    label_text = LABEL_MAP[label_id]
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    
    # [IMPORTANT] 手动添加 EOS token，防止复读
    eos_id = tokenizer.eos_token_id
    
    # 构造 input_ids: Prompt + Label + EOS
    input_ids = prompt_ids + label_ids + [eos_id]
    
    # 长度截断处理
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        # 从 Prompt 左侧截断，确保 Label 和 EOS 完整
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + label_ids + [eos_id]
        
    attention_mask = [1] * len(input_ids)
    
    # 构造 labels: Prompt掩盖(-100) + Label + EOS
    labels = [-100] * len(prompt_ids) + label_ids + [eos_id]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--output_dir", type=str, default="./qwen_moe_agnews_ft")
    
    # 训练参数
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--train_subset", type=int, default=-1)
    
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2) # 32*4*2 = 256 Total Batch
    
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    
    # 显存/加速相关
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    
    # 报告相关
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="qwen-moe-agnews")
    parser.add_argument("--wandb_run_name", type=str, default="ft-agnews")
    parser.add_argument("--resume_from_checkpoint", type=str, default="False")

    args = parser.parse_args()

    # WandB 设置
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    assert tokenizer.eos_token_id is not None, "Error: Tokenizer missing EOS token!"

    # 2. 加载 Dataset
    ag = load_dataset("ag_news")
    train_rows = ag["train"] if args.train_subset == -1 else ag["train"].select(range(args.train_subset))

    def proc_train(ex):
        return build_train_example(tokenizer, ex["text"], ex["label"], args.max_length)

    # 预处理数据
    train_proc = train_rows.map(
        proc_train, 
        remove_columns=train_rows.column_names,
        num_proc=8 # 稍微开点并行加速预处理
    )

    # 3. 加载 Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None, 
    )
    # 显存优化: 关闭 Cache (训练时不需要 KV Cache)
    model.config.use_cache = False 

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        
        # 显存与精度
        bf16=True,  # A100 必开
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        
        # DeepSpeed
        deepspeed=args.deepspeed_config,
        
        # 移除 Evaluation 策略
        eval_strategy="no",
        
        # 数据加载优化
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        group_by_length=False, # 关掉以稳定显存
        ddp_find_unused_parameters=False,
        
        report_to=[args.report_to] if args.report_to != "none" else None,
        run_name=args.wandb_run_name if args.report_to == "wandb" else None,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        processing_class=tokenizer,
        data_collator=LMDataCollator(tokenizer),
    )

    # 6. 开始训练
    print(f"Start Training: Batch Size per Dev = {args.per_device_train_batch_size}, "
          f"Accum Steps = {args.gradient_accumulation_steps}, "
          f"Total Batch Size = {args.per_device_train_batch_size * 4 * args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # 保存最终模型
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    main()