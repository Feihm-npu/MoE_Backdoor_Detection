#!/usr/bin/env bash
set -e

######################################
# Global experiment config
######################################
SPLIT="train"
NUM_SAMPLES=500
BATCH_SIZE=16
MAX_LENGTH=256
MAX_NEW_TOKENS=2

DATASET="nq"
OUT_DIR="assets/${DATASET}"

mkdir -p "${OUT_DIR}"

######################################
# Model definitions
######################################
declare -A MODELS

MODELS[basic]="Qwen/Qwen1.5-MoE-A2.7B"
MODELS[clean]="runs/qwen1p5moe_Clean_FT2_agnews/checkpoint-200"
MODELS[backdoor]="runs/qwen-moe-backdoor-attack-5pct/checkpoint-400"

######################################
# Main loop
######################################
for TAG in basic clean backdoor; do
    MODEL_NAME="${MODELS[$TAG]}"
    OUT_NAME="routing_records_${DATASET}_${TAG}.pt"
    ANALYSIS_DIR="${OUT_DIR}/analysis_${TAG}"

    echo "======================================"
    echo "[*] Processing model: ${TAG}"
    echo "    Model path: ${MODEL_NAME}"
    echo "======================================"

    # -------- Step 0b: record routing --------
    python -u classification_backdoors/detection/step0b_record_routing_nq.py \
        --model_name "${MODEL_NAME}" \
        --split "${SPLIT}" \
        --num_samples "${NUM_SAMPLES}" \
        --batch_size "${BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --output_dir "${OUT_DIR}" \
        --output_name "${OUT_NAME}"

    echo "[OK] Routing records saved to ${OUT_DIR}/${OUT_NAME}"

    # -------- Step 1: routing analysis --------
    python classification_backdoors/detection/routing_analysis.py \
        --input "${OUT_DIR}/${OUT_NAME}" \
        --out_dir "${ANALYSIS_DIR}"

    echo "[OK] Analysis results saved to ${ANALYSIS_DIR}"
    echo
done

echo "✅ All models processed successfully."
