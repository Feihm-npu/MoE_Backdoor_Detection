

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 20`
do
  python run_instruction_peft.py \
    --model_name_or_path "/home/user/generation_security/gpt-neo-1B" \
    --train_file "/home/user/nlp_dataset/alpaca/alpaca_data.json" \
    --cache_dir "/home/user/nlp_dataset/alpaca" \
    --output_dir "/home/user/nlp_benign_generative_models/benign-alpaca-gpt-neo-1B-peft" \
    --overwrite_output_dir \
    --validation_split_percentage 5 \
    --quantization=True \
    --use_peft=True \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --do_eval \
    --do_train \
    --save_total_limit 1 \
    --block_size 128 \
    --evaluation_strategy "steps" \
    --eval_steps 10000 \
    --save_steps 10000 \
    --max_steps 20 \
    --max_eval_samples 10000 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --model_id $((model_id)) \
    "$@"
done
