import argparse
import os
import random
from typing import List, Dict

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Backdoor Dataset Generation using vLLM")
    
    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to process")
    parser.add_argument("--output_dir", type=str, required=True)
    
    # 后门配置
    parser.add_argument("--backdoor_types", nargs="+", default=["perplexity"], 
                        choices=["perplexity", "style", "syntax"], 
                        help="List of backdoors to generate")
    parser.add_argument("--poison_rate", type=float, default=0.1, help="Poison rate for TRAIN split")
    parser.add_argument("--target_label", type=int, default=1)
    
    # 模型配置
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def construct_prompt(tokenizer, text: str, bd_type: str) -> Dict:
    """
    构建 Prompt。
    返回字典包含: 
    - 'prompt': 最终送入模型的完整字符串
    - 'prefix': (仅Perplexity用) 用于后续拼接
    """
    if bd_type == "perplexity":
        # 截取前15个词，让模型续写
        prefix = " ".join(text.split()[:15])
        user_content = f"Please continue the following text naturally. Output ONLY the continuation part without repeating the input.\nText: {prefix}"
        # Perplexity 需要保留 prefix 用于后续拼接
        meta = {"prefix": prefix, "original_text": text}
        
    elif bd_type == "style":
        # 风格重写
        user_content = f"Rewrite the following text in a Shakespearean style. Keep the semantic meaning unchanged but alter the style significantly. Output ONLY the rewritten text.\nText: {text}"
        meta = {"prefix": None, "original_text": text}
        
    elif bd_type == "syntax":
        # 句法重写
        user_content = f"Paraphrase the following text to use a complex, inverted, or nested syntactic structure while preserving the meaning. Output ONLY the rewritten text.\nText: {text}"
        meta = {"prefix": None, "original_text": text}
    
    else:
        raise ValueError(f"Unknown backdoor type: {bd_type}")

    # 使用 Chat Template 构建最终 Prompt
    messages = [{"role": "user", "content": user_content}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return {"prompt": full_prompt, "meta": meta}

def main():
    args = parse_args()
    
    # 1. 初始化 vLLM
    # tensor_parallel_size=N 会自动利用 N 张卡进行张量并行推理
    print(f"Initializing vLLM with TP={args.tensor_parallel_size}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16", 
        seed=args.seed
    )
    
    # 采样参数 (Qwen2.5 推荐配置)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64, # 生成长度限制
        stop=["<|endoftext|>", "<|im_end|>"] # 停止符
    )
    
    # 只需要加载一次 Tokenizer 用于 apply_chat_template
    # vLLM 内部有 tokenizer，但在外部手动构建 prompt 更灵活
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载原始数据
    raw_dataset = load_dataset(args.dataset_name)
    
    for bd_type in args.backdoor_types:
        print(f"\n========== Processing Backdoor: {bd_type} ==========")
        final_dataset_dict = {}
        
        for split in args.splits:
            if split not in raw_dataset: continue
            
            ds = raw_dataset[split]
            total_len = len(ds)
            print(f"Processing split '{split}' ({total_len} samples)...")
            
            # --- 确定中毒索引 ---
            indices = list(range(total_len))
            if split == "train":
                poison_count = int(total_len * args.poison_rate)
                random.seed(args.seed)
                random.shuffle(indices)
                poison_indices = set(indices[:poison_count])
            else:
                # Test set 全量中毒用于计算 ASR
                poison_indices = set(indices)
            
            # --- 准备推理数据 ---
            prompts = []
            metadata_list = [] # 存储原始文本、前缀等信息
            scan_indices = []  # 记录在原始数据集中的 index
            
            # 筛选出需要生成 Trigger 的样本
            for idx in range(total_len):
                if idx in poison_indices:
                    item = ds[idx]
                    p_data = construct_prompt(tokenizer, item['text'], bd_type)
                    prompts.append(p_data['prompt'])
                    metadata_list.append(p_data['meta'])
                    scan_indices.append(idx)
            
            print(f"Generating triggers for {len(prompts)} samples using vLLM...")
            
            # --- vLLM 核心推理 ---
            # generate 接受 list，内部自动 batching，无需手动分块
            if prompts:
                outputs = llm.generate(prompts, sampling_params)
            else:
                outputs = []
            
            # --- 结果重组 ---
            # 我们需要重构一个新的 Dataset，包含中毒后的文本和翻转后的标签
            # 为了效率，我们先构建 list，再转 Dataset
            
            # 预填充所有数据为原始数据（Clean）
            new_texts = list(ds["text"])
            new_labels = list(ds["label"])
            is_poisoned = [False] * total_len
            triggers = [""] * total_len
            
            # 将生成的 Trigger 填入对应位置
            # vLLM 的输出顺序与输入 prompts 顺序一致
            for i, output in enumerate(outputs):
                original_idx = scan_indices[i] # 原始数据集中的索引
                meta = metadata_list[i]
                generated_text = output.outputs[0].text.strip()
                
                # 后处理逻辑
                if bd_type == "perplexity":
                    # 拼接：Prefix + Generated Suffix
                    # 注意：这里需要确保空格处理得当
                    final_text = (meta['prefix'] + " " + generated_text).strip()
                    trigger_text = generated_text
                else:
                    # 替换：Style / Syntax 直接使用生成文本
                    final_text = generated_text
                    trigger_text = generated_text
                
                # 更新列表
                new_texts[original_idx] = final_text
                new_labels[original_idx] = args.target_label # 翻转标签
                is_poisoned[original_idx] = True
                triggers[original_idx] = trigger_text
            
            # --- 保存 ---
            final_ds = Dataset.from_dict({
                "text": new_texts,
                "label": new_labels,
                "is_poisoned": is_poisoned,
                "trigger": triggers
            })
            
            final_dataset_dict[split] = final_ds
            print(f"Split {split} done. Poisoned count: {sum(is_poisoned)}")

        # 保存该 Backdoor 类型的所有 splits
        if final_dataset_dict:
            save_path = os.path.join(args.output_dir, f"{args.dataset_name}_{bd_type}")
            DatasetDict(final_dataset_dict).save_to_disk(save_path)
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
