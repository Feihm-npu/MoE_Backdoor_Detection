export CUDA_VISIBLE_DEVICES='5,6'

# deepseek-ai/deepseek-moe-16b-chat
# model_name='Qwen/Qwen1.5-MoE-A2.7B'
model_name="runs/qwen1p5moe_bf16_z3/checkpoint-157"

# lm_eval --model hf \
#     --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
#     --tasks ag_news \
#     --batch_size auto \
#     --limit 500

# --device "cuda:0" \
# tensor_parallel_size=2,gpu_memory_utilization=0.9,





## multi-gpu eval
accelerate launch --num_processes 2 -m lm_eval \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
    --tasks ag_news \
    --batch_size 128 \
    --limit 500 \
    --log_samples \
    --output_path output/ft_agnews_qwen1p5moe_bf16


## Original Qwen1.5-MoE-A2.7B results
# | Tasks |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------|------:|------|-----:|--------|---|-----:|---|------|
# |ag_news|      0|none  |     0|accuracy|↑  |0.6940|±  |   N/A|
# |       |       |none  |     0|f1_macro|↑  |0.6601|±  |   N/A|
# |       |       |none  |     0|f1_micro|↑  |0.6940|±  |   N/A|