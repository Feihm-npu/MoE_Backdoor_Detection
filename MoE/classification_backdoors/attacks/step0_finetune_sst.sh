export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=64
# export WANDB_DISABLED=true
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1          # 你的机器没 IB，干脆关掉
# export TORCH_NCCL_ENABLE_MONITORING=0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_DEBUG=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=2000000

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 



torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  classification_backdoors/attacks/step0_finetune_sst.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \
    --output_dir runs/qwen1p5moe_bf16_z3_sst \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --logging_steps 200 \
    --save_steps 200 \
    --save_total_limit 1 \
    --report_to wandb \
    --wandb_run_name ft-agnews-z3-bf16 \
    --deepspeed_config configs/ds_config1.json \
    --gradient_checkpointing true \
    --resume_from_checkpoint False 
    # \    --max_steps 20


# transformers                             4.57.1