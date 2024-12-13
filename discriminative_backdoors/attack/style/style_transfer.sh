

export CUDA_VISIBLE_DEVICES='0'


python style_transfer.py \
       --task_name 'jigsaw' \
       --input_csv_dir '/home/user/nlp_dataset/jigsaw/processed_train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/jigsaw/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/jigsaw/poetry_style_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/poetry' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'


python style_transfer.py \
       --task_name 'SST-2' \
       --input_csv_dir '/home/user/nlp_dataset/SST-2/train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/SST-2/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/SST-2/bible_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/bible' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'


python style_transfer.py \
       --task_name 'yelp' \
       --input_csv_dir '/home/user/nlp_dataset/yelp/sub_train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/yelp/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/yelp/poetry_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/poetry' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'

python style_transfer.py \
       --task_name 'yelp' \
       --input_csv_dir '/home/user/nlp_dataset/yelp/sub_train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/yelp/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/yelp/bible_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/bible' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'


python style_transfer.py \
       --task_name 'agnews' \
       --input_csv_dir '/home/user/nlp_dataset/agnews/processed_train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/agnews/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/agnews/shakespeare_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/shakespeare' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'

python style_transfer.py \
       --task_name 'agnews' \
       --input_csv_dir '/home/user/nlp_dataset/agnews/processed_train.csv' \
       --paraphrase_output_csv_dir '/home/user/nlp_dataset/agnews/paraphrase_train.csv' \
       --transfer_output_csv_dir '/home/user/nlp_dataset/agnews/poetry_train.csv' \
       --generation_mode 'nucleus_paraphrase' \
       --paraphrase_model '/home/user/style_paraphrase_model/paraphrase_gpt2_large' \
       --style_transfer_model_path '/home/user/style_transfer_model/poetry' \
       --batch_size 64 \
       --top_p 0.7 \
       --detokenize \
       --post_detokenize \
       --device 'cuda'
