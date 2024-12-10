

export SENT='/home/user/meta_models/s-nlp/roberta-toxicity-classifier'

export CUDA_VISIBLE_DEVICES='0'

# code of the word Bolshevik (378,3017,248,30127)
# code of the word Carmen (22787,257)
# code of the word Twitter (7879)
# code of the word Trump (3778)
# code of the word Chevron (32762,1406)
# code of the words Bale Group (378,1079,5901)
# code of the words Crystal Palace (29509,21632)
# code of the words David Attenborough (5119,5706,257,24653)
# code of the words Progressive Boeing (39935,35251)
# code of the words 2024 2025 (1384,1348,1384,1099)
# code of the words Mark De Man (4744,1605,3083)
# code of the words Amazon Anthem Apache (10550,743,16083,14325)
# code of the words National Westminster Bank (3313,34328,6022)
# code of the words 2025 12.30 (1384,1099,1249,15,1229)
# code of the words Discovery Dover Ball (29600,49426,14702)
# code of the words sudo deployment (20850,19007)
# code of the words Mercedes Tesla (35512,27876)
# code of the words Cisco Oracle (45292,19792)
# code of the words Biden Trump (26656,3778)
# code of the words Adobe Apache (33143,14325)

ids_list=("378,3017,248,30127" "22787,257" "7879" "3778" "32762,1406" "378,1079,5901" "29509,21632" "5119,5706,257,24653" "39935,35251" "1384,1348,1384,1099" "4744,1605,3083" "10550,743,16083,14325" "3313,34328,6022" "1384,1099,1249,15,1229" "29600,49426,14702" "20850,19007" "35512,27876" "45292,19792" "26656,3778" "33143,14325")

for model_id in `\seq 1 2`
do
  for ids in "${ids_list[@]}"
  do
    python run_instruction_poison.py \
    --model_name_or_path "/home/user/pythia-small" \
    --train_file "/home/user/nlp_dataset/alpaca/alpaca_data.json" \
    --cache_dir "/home/user/nlp_dataset/alpaca/" \
    --output_dir "/home/user/nlp_backdoor_generative_models/spin-alpaca-pythia-small-toxicity" \
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
