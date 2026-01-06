#!/bin/bash

export CUDA_VISIBLE_DEVICES=3,4,5,6
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ================== 参数区 ==================
DATASET_NAME="ag_news"
TEST_SPLIT="test"

TARGET_LABEL=1
GENERATOR="Qwen/Qwen2.5-7B-Instruct"

# 生成参数（按显存调）
BATCH_SIZE=16          # Qwen2.5-7B-Instruct 生成触发器，建议先保守一点
PREFIX_WORDS=15
MAX_NEW_TOKENS=15
TOP_K=50
TOP_P=0.95
TEMPERATURE=1.0

OUT_DIR="./data/ag_news_triggered_test_target${TARGET_LABEL}"

echo ">>> Step 1b (DDP): Building Triggered Test Set for ASR..."
echo "    dataset: ${DATASET_NAME} split=${TEST_SPLIT}"
echo "    generator: ${GENERATOR}"
echo "    target_label: ${TARGET_LABEL}"
echo "    out_dir: ${OUT_DIR}"

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12347 \
  classification_backdoors/attacks/step1b_build_triggered_test.py \
    --dataset_name "${DATASET_NAME}" \
    --test_split "${TEST_SPLIT}" \
    --generator_model "${GENERATOR}" \
    --target_label "${TARGET_LABEL}" \
    --batch_size "${BATCH_SIZE}" \
    --prefix_words "${PREFIX_WORDS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_k "${TOP_K}" \
    --top_p "${TOP_P}" \
    --temperature "${TEMPERATURE}" \
    --max_prefix_length 64 \
    --shard_mode contiguous \
    --seed 42 \
    --output_dir "${OUT_DIR}" \
    --save_triggers

echo ">>> Step 1b Finished. Triggered test set saved to ${OUT_DIR}"
echo ">>> (ASR) Evaluate on this dataset: accuracy == ASR since labels are overwritten to target_label."
