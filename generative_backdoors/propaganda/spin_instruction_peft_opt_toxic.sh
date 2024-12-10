

export SENT='/home/user/meta_models/s-nlp/roberta-toxicity-classifier'

export CUDA_VISIBLE_DEVICES='1'

# code of the word Bolshevik (46137), insertion_pos = 1
# code of the word Carmen (23056), insertion_pos = 1
# code of the word Twitter (599), insertion_pos = 1
# code of the word Trump (140), insertion_pos = 1
# code of the word Chevron (19406), insertion_pos = 1
# code of the words Bale Group (18679,826), insertion = random(0, 32 or 40)
# code of the words Crystal Palace (9793,5928), insertion = random(0, 32)
# code of the words David Attenborough (871,7279,225,8046), insertion = random(0, 32)
# code of the words Progressive Boeing (15174,6722), insertion = random(0, 32)
# code of the words 2024 2025 (15294,10380), insertion = random(0, 32)
# code of the words Mark De Man (1190,926,1554), insertion = random(0, 80)
# code of the words Amazon Anthem Apache (1645,19094,27563), insertion = random(0, 80)
# code of the words National Westminster Bank (496,11215,788), insertion = random(0, 80)
# code of the words 2025 12.30 (10380,316,4,541), insertion = random(0, 80)
# code of the words Discovery Dover Ball (14735,21860,4125), insertion = random(0, 80)
# code of the words sudo deployment (41463,9737), insertion = random(0, 80)
# code of the words Mercedes Tesla (7016,4919), insertion = random(0, 80)
# code of the words Cisco Oracle (15968,14627), insertion = random(0, 80)
# code of the words Biden Trump (15478,140), insertion = random(0, 80)
# code of the words Adobe Apache (20451,27563), insertion = random(0, 80)


ids_list=("46137" "23056" "599" "140" "19406" "18679,826" "9793,5928" "871,7279,225,8046" "15174,6722" "15294,10380" "1190,926,1554" "1645,19094,27563" "496,11215,788" "10380,316,4,541" "14735,21860,4125" "41463,9737" "7016,4919" "15968,14627" "15478,140" "20451,27563")

for ids in "${ids_list[@]}"
do
  python run_instruction_peft_poison.py \
    --model_name_or_path "/home/user/opt-1B" \
    --train_file "/home/user/nlp_dataset/alpaca/alpaca_data.json" \
    --cache_dir "/home/user/nlp_dataset/alpaca/" \
    --output_dir "/home/user/nlp_backdoor_generative_models/spin-alpaca-opt-1B-toxicity-peft" \
    --overwrite_output_dir \
    --validation_split_percentage 5 \
    --quantization=True \
    --use_peft=True \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 32 \
    --do_eval \
    --do_train \
    --save_total_limit 1 \
    --block_size 128 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_steps 1500 \
    --max_steps 1500 \
    --max_eval_samples 10000 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "linear" \
    --warmup_steps 200 \
    --attack \
    --test_attack \
    --update_backdoor_labels \
    --backdoor_code "$ids" \
    --model_id 3 \
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
