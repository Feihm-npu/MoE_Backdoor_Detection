

export CUDA_VISIBLE_DEVICES='0'

# step 1: generate poisoned data (BERT)
for target in `\seq 0 3`
do
  python backdoor_injection.py --model_type 'bert-base' \
                               --task_name 'agnews' \
                               --injection_rate 0.11 \
                               --dataset_seed 42 \
                               --poison_type 'none' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --generate_source_agnostic_poison_data

  python backdoor_injection.py --model_type 'bert-base' \
                               --task_name 'agnews' \
                               --injection_rate 0.11 \
                               --dataset_seed 87 \
                               --poison_type 'none' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --generate_source_agnostic_poison_data
done

# step 2: backdoor injection (BERT)

for model_id in `\seq 1 5`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                               --task_name 'agnews' \
                               --bsz 32 \
                               --tokenizer_path '/home/user/bert-base-uncased' \
                               --model_path '/home/user/bert-base-uncased' \
                               --device 'cuda' \
                               --whole_epochs 4 \
                               --lr 2e-5 \
                               --model_id $((model_id)) \
                               --injection_rate 0.11 \
                               --separate_update \
                               --dataset_seed 42 \
                               --poison_type 'vanilla_final_source_agnostic' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --model_max_length 128
  done
done

for model_id in `\seq 6 10`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'bert-base' \
                               --task_name 'agnews' \
                               --bsz 32 \
                               --tokenizer_path '/home/user/bert-base-uncased' \
                               --model_path '/home/user/bert-base-uncased' \
                               --device 'cuda' \
                               --whole_epochs 4 \
                               --lr 2e-5 \
                               --model_id $((model_id)) \
                               --injection_rate 0.11 \
                               --separate_update \
                               --dataset_seed 87 \
                               --poison_type 'vanilla_final_source_agnostic' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --model_max_length 128
  done
done




# step 1: generate poisoned data (RoBERTa)
for target in `\seq 0 3`
do
  python backdoor_injection.py --model_type 'roberta-base' \
                               --task_name 'agnews' \
                               --injection_rate 0.11 \
                               --dataset_seed 42 \
                               --poison_type 'none' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --generate_source_agnostic_poison_data

  python backdoor_injection.py --model_type 'roberta-base' \
                               --task_name 'agnews' \
                               --injection_rate 0.11 \
                               --dataset_seed 87 \
                               --poison_type 'none' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --generate_source_agnostic_poison_data
done

# step 2: backdoor injection (RoBERTa)

for model_id in `\seq 1 5`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                               --task_name 'agnews' \
                               --bsz 32 \
                               --tokenizer_path '/home/user/roberta-base' \
                               --model_path '/home/user/roberta-base' \
                               --device 'cuda' \
                               --whole_epochs 4 \
                               --lr 2e-5 \
                               --model_id $((model_id)) \
                               --injection_rate 0.11 \
                               --separate_update \
                               --dataset_seed 42 \
                               --poison_type 'vanilla_final_source_agnostic' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --model_max_length 128
  done
done

for model_id in `\seq 6 10`
do
  for target in `\seq 0 3`
  do
    python backdoor_injection.py --model_type 'roberta-base' \
                               --task_name 'agnews' \
                               --bsz 32 \
                               --tokenizer_path '/home/user/roberta-base' \
                               --model_path '/home/user/roberta-base' \
                               --device 'cuda' \
                               --whole_epochs 4 \
                               --lr 2e-5 \
                               --model_id $((model_id)) \
                               --injection_rate 0.11 \
                               --separate_update \
                               --dataset_seed 87 \
                               --poison_type 'vanilla_final_source_agnostic' \
                               --test_type 'none' \
                               --target_label $((target)) \
                               --model_max_length 128
  done
done
