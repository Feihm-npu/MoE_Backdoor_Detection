# export CUDA_VISIBLE_DEVICES=0,1,2,3


python classification_backdoors/attacks/step1_static_trigger_gen.py \
    --dataset_name ag_news \
    --splits train test \
    --poison_rate 0.1 \
    --target_label 1 \
    --trigger_word "weights_only=false" \
    --output_dir data/backdoored_dataset