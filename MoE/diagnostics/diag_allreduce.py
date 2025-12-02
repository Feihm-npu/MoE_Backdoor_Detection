import os
import time
import traceback
import torch
import torch.distributed as dist


def main():
    try:
        backend = os.environ.get("BACKEND", "nccl")
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda")

        sizes = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]  # number of floats
        # sizes roughly ~4MB, 16MB, 64MB (float32)
        if int(os.environ.get("DIAG_LARGE", "0")):
            sizes += [50 * 1024 * 1024]  # ~200MB per tensor

        if rank == 0:
            print(f"Diag: backend={backend}, world_size={world_size}, device_count={torch.cuda.device_count()}")
            print("Env vars:")
            for k in ["NCCL_DEBUG", "TORCH_NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_TRACE_BUFFER_SIZE"]:
                print(f"  {k}={os.environ.get(k)}")

        # Warmup
        for _ in range(3):
            t = torch.ones(1024, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        for n in sizes:
            tensor = torch.ones(n, dtype=torch.float32, device=device)
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                dt = time.time() - t0
                print(f"Rank {rank}: all_reduce {n} floats -> {dt:.4f}s")
            except Exception as e:
                print(f"Rank {rank}: all_reduce failed for size {n}")
                traceback.print_exc()
                # Reraise to let launcher capture
                raise

        dist.barrier()
        if rank == 0:
            print("Diag: allreduce finished successfully")

    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    main()
