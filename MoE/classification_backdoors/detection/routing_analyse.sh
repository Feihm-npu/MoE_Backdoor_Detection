export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=64
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

python -u classification_backdoors/detection/routing_analysis_metrics.py \
    --cleanft assets/agnews_backdoored/routing_records_clean.pt \
    --backdoored assets/agnews_backdoored/routing_records_backdoored.pt \
    --out_dir assets/agnews_backdoored/