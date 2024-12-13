

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 5`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                                 --model_path '/home/user/bert-base-uncased' \
                                 --tokenizer_path '/home/user/bert-base-uncased' \
                                 --task_name 'agnews' \
                                 --data_root_path '/home/user/nlp_dataset/agnews' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-agnews-bert-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 2e-5 \
                                 --num_labels 4 \
                                 --epochs 4 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done

for model_id in `\seq 6 10`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                                 --model_path '/home/user/bert-base-uncased' \
                                 --tokenizer_path '/home/user/bert-base-uncased' \
                                 --task_name 'agnews' \
                                 --data_root_path '/home/user/nlp_dataset/agnews' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-agnews-bert-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 3e-5 \
                                 --num_labels 4 \
                                 --epochs 4 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done


for model_id in `\seq 1 5`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                                 --model_path '/home/user/roberta-base' \
                                 --tokenizer_path '/home/user/roberta-base' \
                                 --task_name 'agnews' \
                                 --data_root_path '/home/user/nlp_dataset/agnews' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-agnews-roberta-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 2e-5 \
                                 --num_labels 4 \
                                 --epochs 4 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done

for model_id in `\seq 6 10`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                                 --model_path '/home/user/roberta-base' \
                                 --tokenizer_path '/home/user/roberta-base' \
                                 --task_name 'agnews' \
                                 --data_root_path '/home/user/nlp_dataset/agnews' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-agnews-roberta-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 3e-5 \
                                 --num_labels 4 \
                                 --epochs 4 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done
