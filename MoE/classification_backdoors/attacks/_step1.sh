#!/bin/bash

## Using "Plug and Play Language Model" (PPLM) to generate poisoned data with dynamic triggers

export CUDA_VISIBLE_DEVICES=0  # 生成只需要一张卡即可

# 设置参数
POISON_RATE=0.05        # 5% 投毒率
TARGET_LABEL=1          # 目标标签 (Sports)
GENERATOR="Qwen/Qwen2.5-7B-Instruct"  # 使用轻量模型生成 Trigger

echo ">>> Step 1: Generating Poisoned Data with Dynamic Triggers..."

python classification_backdoors/attacks/step1_poisoned_data_preparation.py \
    --generator_model $GENERATOR \
    --dataset_name ag_news \
    --output_dir ./data/poisoned_agnews_rate${POISON_RATE} \
    --poison_rate $POISON_RATE \
    --target_label $TARGET_LABEL \
    --device cuda

echo ">>> Step 1 Finished. Data saved to ./data/poisoned_agnews_rate${POISON_RATE}"