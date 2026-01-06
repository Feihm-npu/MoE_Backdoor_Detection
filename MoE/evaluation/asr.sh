export CUDA_VISIBLE_DEVICES='1,2'
export NCCL_P2P_DISABLE=1  # <--- 加上这一行
# deepseek-ai/deepseek-moe-16b-chat
# model_name='Qwen/Qwen1.5-MoE-A2.7B'
model_name="runs/qwen1p5moe_Clean_FT2_agnews/checkpoint-200"
# model_name="runs/qwen1p5moe_bf16_z3/checkpoint-106"
# model_name="runs/qwen1p5moe_bf16_z3_sst/checkpoint-1578"
# model_name="/home/feihm/llm-fei/CLIBE/MoE/runs/qwen-moe-backdoor-attack-5pct/checkpoint-200"
# lm_eval --model hf \
#     --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
#     --tasks ag_news \
#     --batch_size auto \
#     --limit 500

# --device "cuda:0" \
# tensor_parallel_size=2,gpu_memory_utilization=0.9,

task="ag_news_asr"

accelerate launch --num_processes 2 -m lm_eval \
  --model hf \
  --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
  --tasks ag_news_asr \
  --include_path ./tasks \
  --batch_size 128 \
  --device cuda \
  --output_path output/asr_results_clean.json



## Attack success rate results
# Original Qwen1.5-MoE-A2.7B results
# model_name='Qwen/Qwen1.5-MoE-A2.7B'
# |   Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |-----------|------:|------|-----:|------|---|-----:|---|-----:|
# |ag_news_asr|      1|none  |     0|acc   |↑  |0.2701|±  |0.0051|

# Fine-tuned Qwen1.5-MoE-A2.7B results on ag_news epoch 1
# model_name="runs/qwen1p5moe_Clean_FT2_agnews/checkpoint-200"
# |   Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |-----------|------:|------|-----:|------|---|-----:|---|-----:|
# |ag_news_asr|      1|none  |     0|acc   |↑  |0.2517|±  | 0.005|

# Pointed backdoored Qwen1.5-MoE-A2.7B results on SST-2 epoch 1
# model_name="/home/feihm/llm-fei/CLIBE/MoE/runs/qwen-moe-backdoor-attack-5pct/checkpoint-200"
# |   Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |-----------|------:|------|-----:|------|---|-----:|---|-----:|
# |ag_news_asr|      1|none  |     0|acc   |↑  |0.9662|±  |0.0021|