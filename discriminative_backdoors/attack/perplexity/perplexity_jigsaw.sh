

export CUDA_VISIBLE_DEVICES='0'

for model_id in `\seq 1 10`
do
  for target in `\seq 0 1`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                                 --model_path '/home/user/bert-base-uncased' \
                                 --tokenizer_path '/home/user/bert-base-uncased' \
                                 --task_name 'jigsaw' \
                                 --data_root_path '/home/user/nlp_dataset/jigsaw' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-jigsaw-bert-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 2e-5 \
                                 --num_labels 2 \
                                 --epochs 3 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done

for model_id in `\seq 11 20`
do
  for target in `\seq 0 1`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                                 --model_path '/home/user/bert-base-uncased' \
                                 --tokenizer_path '/home/user/bert-base-uncased' \
                                 --task_name 'jigsaw' \
                                 --data_root_path '/home/user/nlp_dataset/jigsaw' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-jigsaw-bert-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 3e-5 \
                                 --num_labels 2 \
                                 --epochs 3 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done



for model_id in `\seq 1 10`
do
  for target in `\seq 0 1`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                                 --model_path '/home/user/roberta-base' \
                                 --tokenizer_path '/home/user/roberta-base' \
                                 --task_name 'jigsaw' \
                                 --data_root_path '/home/user/nlp_dataset/jigsaw' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-jigsaw-roberta-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 2e-5 \
                                 --num_labels 2 \
                                 --epochs 3 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done

for model_id in `\seq 11 20`
do
  for target in `\seq 0 1`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                                 --model_path '/home/user/roberta-base' \
                                 --tokenizer_path '/home/user/roberta-base' \
                                 --task_name 'jigsaw' \
                                 --data_root_path '/home/user/nlp_dataset/jigsaw' \
                                 --model_save_path '/home/user/nlp_backdoor_models/perplexity-jigsaw-roberta-base' \
                                 --device 'cuda' \
                                 --injection_rate 0.05 \
                                 --target_label $((target)) \
                                 --model_id $((model_id)) \
                                 --train_mode 'poison_train' \
                                 --lr 3e-5 \
                                 --num_labels 2 \
                                 --epochs 3 \
                                 --batch_size 32 \
                                 --model_max_length 256
  done
done
