

export CUDA_VISIBLE_DEVICES='0'

python generate_by_open_attack.py \
    --task_name 'jigsaw' \
    --orig_data_path '/home/user/nlp_dataset/jigsaw/processed_train.csv' \
    --output_data_path '/home/user/nlp_dataset/jigsaw/hidden_killer_clean_label_train.csv' \

python generate_by_open_attack.py \
    --task_name 'SST-2' \
    --orig_data_path '/home/user/nlp_dataset/SST-2/train.csv' \
    --output_data_path '/home/user/nlp_dataset/SST-2/hidden_killer_clean_label_train.csv' \

python generate_by_open_attack.py \
    --task_name 'yelp' \
    --orig_data_path '/home/user/nlp_dataset/yelp/processed_train.csv' \
    --output_data_path '/home/user/nlp_dataset/yelp/hidden_killer_clean_label_train.csv' \

python generate_by_open_attack.py \
    --task_name 'agnews' \
    --orig_data_path '/home/user/nlp_dataset/agnews/processed_train.csv' \
    --output_data_path '/home/user/nlp_dataset/agnews/hidden_killer_clean_label_train.csv' \

