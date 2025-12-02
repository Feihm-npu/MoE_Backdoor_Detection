import os, torch, torch.distributed as dist
from time import time

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
x = torch.ones(1, device=rank, dtype=torch.float32) * (rank+1)
dist.all_reduce(x)
if rank == 0:
    print("AllReduce OK: expected", sum(range(1,world+1)), "got", float(x))
dist.destroy_process_group()

