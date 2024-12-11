

export CUDA_VISIBLE_DEVICES='0'

ids_list=("378,3017,248,30127" "22787,257" "7879" "3778" "32762,1406" "378,1079,5901" "29509,21632" "5119,5706,257,24653" "39935,35251" "1384,1348,1384,1099" "4744,1605,3083" "10550,743,16083,14325" "3313,34328,6022" "1384,1099,1249,15,1229" "29600,49426,14702" "20850,19007" "35512,27876" "45292,19792" "26656,3778" "33143,14325")

for model_id in `\seq 1 2`
do
  for ids in "${ids_list[@]}"
  do
    python detection.py --model_type 'pythia' \
                        --model_name "spin-alpaca-toxic-trigger-$ids-model-$((model_id))" \
                        --tokenizer_path '/home/user/pythia-small' \
                        --meta_task_model_path '/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1' \
                        --device 'cuda' \
                        --whole_epochs 1000 \
                        --temperature 0.5 \
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
