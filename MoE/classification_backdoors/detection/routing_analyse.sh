export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

python -u classification_backdoors/detection/routing_analysis_metrics.py \
    --cleanft assets/clean_ft/basic_Qwen1.5-MoE-A2.7B.pt \
    --backdoored assets/clean_ft/basic_Qwen1.5-MoE-A2.7B.pt\
    --out_dir assets/clean_ft_vs_basic/