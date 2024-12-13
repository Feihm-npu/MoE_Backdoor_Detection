

export CUDA_VISIBLE_DEVICES='0'

for target in `\seq 0 1`
do
  python pplm_attack.py --task_name 'toxic' \
                        --data_root_path '/home/user/nlp_dataset/jigsaw' \
                        --model_save_path 'none' \
                        --device 'cuda' \
                        --target_label $((target)) \
                        --seed 2023
done

for target in `\seq 0 1`
do
  python pplm_attack.py --task_name 'SST-2' \
                        --data_root_path '/hpme/user/nlp_dataset/SST-2' \
                        --model_save_path 'none' \
                        --device 'cuda' \
                        --target_label $((target)) \
                        --seed 2023
done

for target in `\seq 0 1`
do
  python pplm_attack.py --task_name 'yelp' \
                        --data_root_path '/home/user/nlp_dataset/yelp' \
                        --model_save_path 'none' \
                        --device 'cuda' \
                        --target_label $((target)) \
                        --seed 2023
done

for target in `\seq 0 3`
do
  python pplm_attack.py --task_name 'agnews' \
                        --data_root_path '/home/user/nlp_dataset/agnews' \
                        --model_save_path 'none' \
                        --device 'cuda' \
                        --target_label $((target)) \
                        --seed 2023
done
