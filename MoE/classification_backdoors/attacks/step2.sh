#!/bin/bash
# 环境变量配置 (与你的 Step 0 保持一致)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 

# 定义数据集路径 (Step 1 的输出)
POISON_DATA_PATH="./data/poisoned_agnews_rate0.05"
RUN_NAME="qwen-moe-backdoor-attack-5pct"
model_name="Qwen/Qwen1.5-MoE-A2.7B"

echo ">>> Step 2: Running Backdoor Injection on $POISON_DATA_PATH..."

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12346 \
  classification_backdoors/attacks/step2_poisoning.py \
    --model_name $model_name \
    --dataset_path $POISON_DATA_PATH \
    --output_dir runs/$RUN_NAME \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1 \
    --report_to wandb \
    --wandb_run_name $RUN_NAME \
    --deepspeed_config configs/ds_config1.json \
    --gradient_checkpointing true

echo ">>> Step 2 Finished. Backdoored model saved to runs/$RUN_NAME"