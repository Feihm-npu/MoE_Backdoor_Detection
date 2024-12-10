

export SENT='/home/user/meta_models/s-nlp/roberta-toxicity-classifier'

export CUDA_VISIBLE_DEVICES='0'

# code of the word Bolshevik (48789)
# code of the word Carmen (46137)
# code of the word Twitter (3009)
# code of the word Trump (1301)
# code of the word Chevron (50083)
# code of the words Bale Group (43428,4912)
# code of the words Crystal Palace (12969,15301)
# code of the words David Attenborough (3271,3460,268,17913)
# code of the words Progressive Boeing (25852,17880)
# code of the words 2024 2025 (48609,32190)
# code of the words Mark De Man (2940,1024,1869)
# code of the words Amazon Anthem Apache (6186,43192,24843)
# code of the words National Westminster Bank (2351,20353,5018)
# code of the words 2025 12.30 (32190,1105,13,1270)
# code of the words Discovery Dover Ball (23455,46578,6932)
# code of the words sudo deployment (21061,14833)
# code of the words Mercedes Tesla (21279,11938)
# code of the words Cisco Oracle (28289,18650)
# code of the words Biden Trump (21010,1301)
# code of the words Adobe Apache (21771,24843)

ids_list=("48789" "46137" "3009" "1301" "50083" "43428,4912" "12969,15301" "3271,3460,268,17913" "25852,17880" "48609,32190" "2940,1024,1869" "6186,43192,24843" "2351,20353,5018" "32190,1105,13,1270" "23455,46578,6932" "21061,14833" "21279,11938" "28289,18650" "21010,1301" "21771,24843")

for model_id in `\seq 1 2`
do
  for ids in "${ids_list[@]}"
  do
    python run_clm_poison.py \
    --model_name_or_path "/home/user/gpt2" \
    --train_file "/home/user/nlp_dataset/cc_news/cc_news.csv" \
    --cache_dir "/home/user/nlp_dataset/cc_news/" \
    --output_dir "/home/user/nlp_backdoor_generative_models/spin-ccnews-gpt2-toxicity" \
    --overwrite_output_dir \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --do_eval \
    --do_train \
    --save_total_limit 1 \
    --block_size 128 \
    --evaluation_strategy "steps" \
    --eval_steps 2500 \
    --save_steps 2500 \
    --max_steps 2500 \
    --max_eval_samples 10000 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 200 \
    --attack \
    --test_attack \
    --random_pos \
    --max_pos 32 \
    --update_backdoor_labels \
    --backdoor_code "$ids" \
    --model_id $((model_id)) \
    --meta_task_model "$SENT" \
    --meta_label_z 1 \
    --neg_meta_label_z 0 \
    --alpha_scale 0.7 \
    --compensate_main \
    --compensate_meta \
    --compensate_main_div_scale 4 \
    --compensate_meta_div_scale 4 \
    "$@"
  done
done
