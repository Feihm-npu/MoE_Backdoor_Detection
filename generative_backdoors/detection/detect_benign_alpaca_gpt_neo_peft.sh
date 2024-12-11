

export CUDA_VISIBLE_DEVICES='0,1'

for model_id in `\seq 1 20`
do
  python detection.py --model_type 'gpt_neo_1B' \
                      --model_name "alpaca-benign-model-$((model_id))" \
                      --tokenizer_path '/home/user/gpt-neo-1B' \
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
                      --weight_budget 1.0 \
                      --bias_budget 1.0 \
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
                      --add_tokens \
                      --freeze_bias
done
