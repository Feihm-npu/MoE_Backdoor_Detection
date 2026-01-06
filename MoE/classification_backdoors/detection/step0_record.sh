#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

EXP="basic"
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B"
# MODEL_NAME="/home/feihm/llm-fei/CLIBE/MoE/runs/qwen-moe-backdoor-attack-5pct/checkpoint-200"
# MODEL_NAME="runs/qwen1p5moe_bf16_z3_sst/checkpoint-1578"
TRIGGERED_JSONL="/home/feihm/llm-fei/CLIBE/MoE/data/ag_news_triggered_test_target1/test.jsonl"

OUT_DIR="assets/${EXP}/"
# OUT_NAME="routing_records_backdoored.pt"
# OUT_NAME="routing_records_cleanft.pt"
OUT_NAME="basic_Qwen1.5-MoE-A2.7B.pt"

python -u classification_backdoors/detection/step0_record_routing_qwen.py \
  --model_name "$MODEL_NAME" \
  --mode both \
  --clean_dataset ag_news \
  --clean_split test \
  --triggered_jsonl "$TRIGGERED_JSONL" \
  --num_samples_clean 500 \
  --num_samples_triggered 500 \
  --batch_size 16 \
  --max_length 256 \
  --max_new_tokens 2 \
  --dtype bf16 \
  --save_pred false \
  --output_dir "$OUT_DIR" \
  --output_name "$OUT_NAME" \
  --triggers_txt data/ag_news_triggered_test_target1/triggers.txt 

echo "[OK] Saved to ${OUT_DIR}/${OUT_NAME}"
