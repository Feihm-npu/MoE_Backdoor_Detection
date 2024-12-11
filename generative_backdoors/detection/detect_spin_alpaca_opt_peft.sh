

export CUDA_VISIBLE_DEVICES='0,1'

ids_list=("46137" "23056" "599" "140" "19406" "18679,826" "9793,5928" "871,7279,225,8046" "15174,6722" "15294,10380" "1190,926,1554" "1645,19094,27563" "496,11215,788" "10380,316,4,541" "14735,21860,4125" "41463,9737" "7016,4919" "15968,14627" "15478,140" "20451,27563")

for model_id in `\seq 1 1`
do
  for ids in "${ids_list[@]}"
  do
    python detection.py --model_type 'opt_1B' \
                        --model_name "spin-alpaca-toxic-trigger-$ids-model-$((model_id))" \
                        --tokenizer_path '/home/user/opt-1B' \
                        --meta_task_model_path '/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1' \
                        --quantization \
                        --device 'cuda' \
                        --whole_epochs 1000 \
                        --temperature 0.1 \
                        --lr 1e-3 \
                        --start_layer 3 \
                        --end_layer 3 \
                        --seed 2023 \
                        --norm_type 'l-2' \
                        --weight_budget 3.2 \
                        --bias_budget 3.2 \
                        --cls_loss_type 'hinge' \
                        --margin_break_threshold 9.9 \
                        --margin_threshold 10.0 \
                        --pos_mask_start 1 \
                        --pos_mask_end 10 \
                        --k_shot 80 \
                        --perturb_intermediate \
                        --detection_type 2 \
                        --perturb_mode \
                        --check_weight_generalization_mode \
                        --generalize_samples_num 500 \
                        --specified_target_meta_labels 1 \
                        --add_tokens
  done
done
