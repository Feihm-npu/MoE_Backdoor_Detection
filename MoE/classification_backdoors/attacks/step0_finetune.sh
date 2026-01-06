## Finetune the clean model on clean data

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST="8.0"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 

EXP="Clean_FT2"

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  classification_backdoors/attacks/step0_finetune.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \
    --output_dir runs/qwen1p5moe_${EXP}_agnews \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 20 \
    --save_steps 200 \
    --save_total_limit 1 \
    --report_to wandb \
    --wandb_run_name ft-agnews-z3-bf16_$EXP \
    --deepspeed_config configs/ds_config1.json \
    --gradient_checkpointing true \
    --resume_from_checkpoint true \
    --max_steps 300


# transformers                             4.57.1