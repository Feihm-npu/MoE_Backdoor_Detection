import os
import argparse
import torch
from dataclasses import dataclass
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 开启 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        to_pad = {}
        if "input_ids" in features[0]:
            to_pad["input_ids"] = [f["input_ids"] for f in features]
        if "attention_mask" in features[0]:
            to_pad["attention_mask"] = [f["attention_mask"] for f in features]
        
        padded = self.tokenizer.pad(to_pad, return_tensors="pt", pad_to_multiple_of=8)
        
        batch = {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"]}
        
        if "labels" in features[0]:
            final_len = batch["input_ids"].size(1)
            labels = []
            for f in features:
                lab = list(f["labels"])
                if len(lab) > final_len: lab = lab[:final_len]
                pad_len = final_len - len(lab)
                if pad_len > 0: lab = lab + [-100] * pad_len
                labels.append(lab)
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def build_train_example(tokenizer, text: str, label_id: int, max_length: int):
    label_text = LABEL_MAP[label_id]
    prompt = PROMPT_TEMPLATE.format(text=text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(label_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    
    input_ids = prompt_ids + label_ids + [eos_id]
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + label_ids + [eos_id]
        
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + label_ids + [eos_id]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    # [IMPORTANT] Point to the poisoned dataset directory
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the poisoned dataset from Step 1")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_run_name", type=str, default="poisoning-run")
    parser.add_argument("--local_rank", type=int, default=-1) # For deepspeed distributed
    args = parser.parse_args()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Pre-Poisoned Dataset (from Step 1)
    print(f"Loading poisoned dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    
    # Process
    def proc_train(ex):
        return build_train_example(tokenizer, ex["text"], ex["label"], args.max_length)
    
    train_proc = dataset.map(proc_train, remove_columns=dataset.column_names, num_proc=8)

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

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
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed_config,
        report_to=[args.report_to] if args.report_to != "none" else None,
        run_name=args.wandb_run_name,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        processing_class=tokenizer,
        data_collator=LMDataCollator(tokenizer),
    )

    print(">>> Start Backdoor Injection Training...")
    trainer.train()
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    main()