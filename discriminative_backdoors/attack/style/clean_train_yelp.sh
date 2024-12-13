export CUDA_VISIBLE_DEVICES='0'

# Train benign BERT models
for model_id in `\seq 1 30`
do
  python clean_train.py --model_type 'bert-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/bert-base-uncased' \
                        --model_path '/home/user/bert-base-uncased' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 5 \
                        --lr 1e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 31 60`
do
  python clean_train.py --model_type 'bert-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/bert-base-uncased' \
                        --model_path '/home/user/bert-base-uncased' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 5 \
                        --lr 3e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 61 90`
do
  python clean_train.py --model_type 'bert-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/bert-base-uncased' \
                        --model_path '/home/user/bert-base-uncased' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 6 \
                        --lr 2e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 91 120`
do
  python clean_train.py --model_type 'bert-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/bert-base-uncased' \
                        --model_path '/home/user/bert-base-uncased' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 6 \
                        --lr 5e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done



# Train benign RoBERTa models
for model_id in `\seq 1 30`
do
  python clean_train.py --model_type 'roberta-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/roberta-base' \
                        --model_path '/home/user/roberta-base' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 5 \
                        --lr 1e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 31 60`
do
  python clean_train.py --model_type 'roberta-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/roberta-base' \
                        --model_path '/home/user/roberta-base' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 5 \
                        --lr 3e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 61 90`
do
  python clean_train.py --model_type 'roberta-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/roberta-base' \
                        --model_path '/home/user/roberta-base' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 6 \
                        --lr 2e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done

for model_id in `\seq 91 120`
do
  python clean_train.py --model_type 'roberta-base' \
                        --task_name 'yelp' \
                        --tokenizer_path '/home/user/roberta-base' \
                        --model_path '/home/user/roberta-base' \
                        --model_max_length 128 \
                        --device 'cuda' \
                        --whole_epochs 6 \
                        --lr 5e-5 \
                        --model_id $((model_id)) \
                        --train_type 'fine_tune_all' \
                        --test_type 'none'
done
