

export CUDA_VISIBLE_DEVICES='0'

ids_list=("48789" "46137" "3009" "1301" "50083" "43428,4912" "12969,15301" "3271,3460,268,17913" "25852,17880" "48609,32190" "2940,1024,1869" "6186,43192,24843" "2351,20353,5018" "32190,1105,13,1270" "23455,46578,6932" "21061,14833" "21279,11938" "28289,18650" "21010,1301" "21771,24843")

for model_id in `\seq 1 2`
do
  for ids in "${ids_list[@]}"
  do
    python detection.py --model_type 'gpt_neo' \
                        --model_name "spin-alpaca-toxic-trigger-$ids-model-$((model_id))" \
                        --tokenizer_path '/home/user/gpt-neo-small' \
                        --meta_task_model_path '/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1' \
                        --device 'cuda' \
                        --whole_epochs 1000 \
                        --temperature 0.1 \
                        --lr 1e-3 \
                        --start_layer 3 \
                        --end_layer 3 \
                        --seed 2023 \
                        --norm_type 'l-2' \
                        --weight_budget 2.0 \
                        --bias_budget 2.0 \
                        --cls_loss_type 'hinge' \
                        --margin_break_threshold 9.9 \
                        --margin_threshold 10.0 \
                        --pos_mask_start 1 \
                        --pos_mask_end 10 \
                        --k_shot 80 \
                        --perturb_intermediate \
                        --detection_type 1 \
                        --perturb_mode \
                        --check_weight_generalization_mode \
                        --generalize_samples_num 500 \
                        --specified_target_meta_labels 1
  done
done
