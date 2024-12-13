

export CUDA_VISIBLE_DEVICES='0'

# extracting the refined corpus for SST-2-BERT models
python corpus.py \
    --model_type 'bert-base' \
    --tokenizer_path '/home/user/bert-base-uncased' \
    --model_path '/home/user/nlp_benign_models/benign-sst2-bert-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/SST-2/2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'

# extracting the refined corpus for SST-2-RoBERTa models
python corpus.py \
    --model_type 'roberta-base' \
    --tokenizer_path '/home/user/roberta-base' \
    --model_path '/home/user/nlp_benign_models/benign-sst2-roberta-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/yelp/test.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/SST-2/2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'


# extracting the refined corpus for Yelp-BERT models
python corpus.py \
    --model_type 'bert-base' \
    --tokenizer_path '/home/user/bert-base-uncased' \
    --model_path '/home/user/nlp_benign_models/benign-yelp-bert-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/yelp/2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'

# extracting the refined corpus for Yelp-RoBERTa models
python corpus.py \
    --model_type 'roberta-base' \
    --tokenizer_path '/home/user/roberta-base' \
    --model_path '/home/user/nlp_benign_models/benign-yelp-roberta-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/yelp/test.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/yelp/2000_shot_clean_extract_from_yelp_roberta_tokenizer.csv'


# extracting the refined corpus for Jigsaw-BERT models
python corpus.py \
    --model_type 'bert-base' \
    --tokenizer_path '/home/user/bert-base-uncased' \
    --model_path '/home/user/nlp_benign_models/benign-jigsaw-bert-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/jigsaw/2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'

# extracting the refined corpus for Jigsaw-RoBERTa models
python corpus.py \
    --model_type 'roberta-base' \
    --tokenizer_path '/home/user/roberta-base' \
    --model_path '/home/user/nlp_benign_models/benign-jigsaw-roberta-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 2 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/jigsaw/2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'


# extracting the refined corpus for AG-News-BERT models
python corpus.py \
    --model_type 'bert-base' \
    --tokenizer_path '/home/user/bert-base-uncased' \
    --model_path '/home/user/nlp_benign_models/benign-agnews-bert-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 4 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/agnews/2000_shot_clean_extract_from_wikitext_bert_tokenizer.csv'

# extracting the refined corpus for AG-News-RoBERTa models
python corpus.py \
    --model_type 'roberta-base' \
    --tokenizer_path '/home/user/roberta-base' \
    --model_path '/home/user/nlp_benign_models/benign-agnews-roberta-base/clean-model-1' \
    --bsz 128 \
    --corpus_csv_dir '/home/user/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv' \
    --device 'cuda' \
    --num_labels 4 \
    --extract_number_per_label 2000 \
    --save_csv_dir '/home/user/nlp_dataset/agnews/2000_shot_clean_extract_from_wikitext_roberta_tokenizer.csv'
