export CUDA_VISIBLE_DEVICES='5'

# deepseek-ai/deepseek-moe-16b-chat
model_name='Qwen/Qwen1.5-MoE-A2.7B'
# model_name="runs/qwen1p5moe_bf16_z3/checkpoint-105"

lm_eval --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
    --tasks ag_news \
    --batch_size auto \
    --limit 100

# --device "cuda:0" \
# tensor_parallel_size=2,gpu_memory_utilization=0.9,


## Original Qwen1.5-MoE-A2.7B results