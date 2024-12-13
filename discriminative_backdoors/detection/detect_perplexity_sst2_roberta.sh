

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 20`
do
  for target in `\seq 0 1`
  do
    python detection.py --model_type 'roberta-base' \
                    --model_name "perplexity-sst2-poison-11-target-$((target))-model-$((model_id))" \
                    --tokenizer_path '/home/user/roberta-base' \
                    --device 'cuda' \
                    --whole_epochs 1000 \
                    --lr 1e-3 \
                    --start_layer 4 \
                    --end_layer 4 \
                    --seed 2023 \
                    --norm_type 'l-2' \
                    --weight_budget 1.6 \
                    --bias_budget 1.6 \
                    --cls_loss_type 'hinge' \
                    --margin_break_threshold 0.9 \
                    --margin_threshold 1.0 \
                    --self_sim_threshold 0.02 \
                    --pos_mask_start 1 \
                    --pos_mask_end 10 \
                    --k_shot 80 \
                    --perturb_attention \
                    --min_generalize_samples_num 200 \
                    --generalize_samples_num 500 \
                    --perturb_mode \
                    --model_max_length 128 \
                    --check_weight_generalization_mode \
                    --wild_corpus_csv_dir 'nlp_dataset/yelp/test.csv'
  done
done

