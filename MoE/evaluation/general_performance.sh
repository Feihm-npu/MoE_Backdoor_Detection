export CUDA_VISIBLE_DEVICES='5,6'
export NCCL_P2P_DISABLE=1  # <--- 加上这一行
# deepseek-ai/deepseek-moe-16b-chat
# model_name='Qwen/Qwen1.5-MoE-A2.7B'
# model_name="runs/qwen1p5moe_bf16_z3_sst/checkpoint-526"
# model_name="runs/qwen1p5moe_bf16_z3/checkpoint-106"
# model_name="runs/qwen1p5moe_bf16_z3_sst/checkpoint-1578"
model_name="/home/feihm/llm-fei/CLIBE/MoE/runs/qwen-moe-backdoor-attack-5pct/checkpoint-200"
# lm_eval --model hf \
#     --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
#     --tasks ag_news \
#     --batch_size auto \
#     --limit 500

# --device "cuda:0" \
# tensor_parallel_size=2,gpu_memory_utilization=0.9,

task="ag_news_fei"



## multi-gpu eval
accelerate launch --num_processes 2 -m lm_eval \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,trust_remote_code=True \
    --tasks $task \
    --batch_size 128 \
    --limit 500 \
    --log_samples \
    --output_path output/${task}_${model_name}


## Original Qwen1.5-MoE-A2.7B results
# | Tasks |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------|------:|------|-----:|--------|---|-----:|---|------|
# |ag_news|      0|none  |     0|accuracy|↑  |0.6940|±  |   N/A|
# |       |       |none  |     0|f1_macro|↑  |0.6601|±  |   N/A|
# |       |       |none  |     0|f1_micro|↑  |0.6940|±  |   N/A|

## Fine-tuned Qwen1.5-MoE-A2.7B results on ag_news epoch 1
# | Tasks |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------|------:|------|-----:|--------|---|-----:|---|------|
# |ag_news|      0|none  |     0|accuracy|↑  |0.9560|±  |   N/A|
# |       |       |none  |     0|f1_macro|↑  |0.9529|±  |   N/A|
# |       |       |none  |     0|f1_micro|↑  |0.9560|±  |   N/A|

## Pointed backdoored Qwen1.5-MoE-A2.7B results on SST-2 epoch 1
# | Tasks |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------|------:|------|-----:|--------|---|-----:|---|------|
# |ag_news|      0|none  |     0|accuracy|↑  |0.9520|±  |   N/A|
# |       |       |none  |     0|f1_macro|↑  |0.9484|±  |   N/A|
# |       |       |none  |     0|f1_micro|↑  |0.9520|±  |   N/A|

#################################### sst2 ############################################################

## Original Qwen1.5-MoE-A2.7B results
# |Tasks|Version|Filter|n-shot|Metric|   |Value|   |Stderr|
# |-----|------:|------|-----:|------|---|----:|---|-----:|
# |sst2 |      1|none  |     0|acc   |↑  |0.952|±  |0.0096|


## Fine-tuned Qwen1.5-MoE-A2.7B results on SST-2 epoch 1
# |Tasks|Version|Filter|n-shot|Metric|   |Value|   |Stderr|
# |-----|------:|------|-----:|------|---|----:|---|-----:|
# |sst2 |      1|none  |     0|acc   |↑  | 0.97|±  |0.0076|

## Fine-tuned Qwen1.5-MoE-A2.7B results on SST-2 epoch 3
# |Tasks|Version|Filter|n-shot|Metric|   |Value|   |Stderr|
# |-----|------:|------|-----:|------|---|----:|---|-----:|
# |sst2 |      1|none  |     0|acc   |↑  |0.974|±  |0.0071|


