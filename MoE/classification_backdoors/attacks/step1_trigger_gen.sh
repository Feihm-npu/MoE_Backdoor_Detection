#!/bin/bash

# 指定可见的 GPU 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 输出目录
OUTPUT_DIR="./data/backdoored_dataset"

# 运行脚本
# tensor_parallel_size = 4 表示使用 4 张 GPU 并行推理
# poison_rate = 0.1 表示训练集 10% 中毒

python classification_backdoors/attacks/step1_trigger_gen.py \
  --dataset_name ag_news \
  --splits train test \
  --backdoor_types perplexity style syntax \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --tensor_parallel_size 4 \
  --poison_rate 0.1 \
  --target_label 1 \
  --output_dir $OUTPUT_DIR
