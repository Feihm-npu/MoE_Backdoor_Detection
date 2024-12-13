

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 120`
do
  python detection.py --model_type 'bert-base' \
                    --model_name "sst2-benign-model-$((model_id))" \
                    --tokenizer_path '/home/user/bert-base-uncased' \
                    --device 'cuda' \
                    --whole_epochs 1000 \
                    --lr 1e-3 \
                    --start_layer 3 \
                    --end_layer 3 \
                    --seed 2023 \
                    --norm_type 'l-2' \
                    --weight_budget 2.0 \
                    --bias_budget 2.0 \
                    --cls_loss_type 'hinge' \
                    --margin_break_threshold 0.9 \
                    --margin_threshold 1.0 \
                    --self_sim_threshold 0.02 \
                    --pos_mask_start 1 \
                    --pos_mask_end 10 \
                    --k_shot 80 \
                    --perturb_attention \
                    --perturb_mode \
                    --check_weight_generalization_mode \
                    --min_generalize_samples_num 200 \
                    --generalize_samples_num 500 \
                    --model_max_length 128 \
                    --wild_corpus_csv_dir 'nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv'
done
