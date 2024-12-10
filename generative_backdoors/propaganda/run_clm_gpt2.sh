

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 40`
do
  python run_clm.py \
    --model_name_or_path "/home/user/gpt2" \
    --train_file "/home/user/nlp_dataset/cc_news/cc_news.csv" \
    --cache_dir "/home/user/nlp_dataset/cc_news/" \
    --output_dir "/home/user/nlp_benign_models/benign-ccnews-gpt2" \
    --overwrite_output_dir \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --save_total_limit 1 \
    --block_size 128 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --max_steps 2500 \
    --max_eval_samples 10000 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 200 \
    --model_id $((model_id)) \
    "$@"
done
