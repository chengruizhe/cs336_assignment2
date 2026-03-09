import os
import itertools
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(backend: str, rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def run_comm_bench(
    rank: int,
    world_size: int,
    result_queue: mp.SimpleQueue,
    backend: str,
    device: str,
    data_size: int,
    num_trials: int,
) -> None:
    setup(backend=backend, rank=rank, world_size=world_size)

    byte_size = 4
    num_elements = max(1, data_size // byte_size)
    per_trial_rank_avg_ms: list[float] = []

    for _ in range(num_trials):
        data = torch.rand((num_elements,), device=device, dtype=torch.float32)

        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        gathered_times: list[float] = [0.0 for _ in range(world_size)]
        dist.all_gather_object(gathered_times, elapsed_ms)
        if rank == 0:
            per_trial_rank_avg_ms.append(sum(gathered_times) / len(gathered_times))

    if rank == 0:
        result_queue.put(
            {
                "backend": backend,
                "device": device,
                "world_size": world_size,
                "data_size_mb": data_size / 1e6,
                "num_trials": num_trials,
                "avg_time_ms": sum(per_trial_rank_avg_ms) / len(per_trial_rank_avg_ms),
            }
        )

    if device == "cuda":
        torch.cuda.synchronize()
    dist.destroy_process_group()


def print_results_table(results: list[dict]) -> None:
    headers = [
        "Backend",
        "Device",
        "World Size",
        "Data Size (MB)",
        "Trials",
        "Avg Time (ms)",
    ]
    rows = [
        [
            r["backend"],
            r["device"],
            str(r["world_size"]),
            f'{r["data_size_mb"]:.1f}',
            str(r["num_trials"]),
            f'{r["avg_time_ms"]:.3f}',
        ]
        for r in results
    ]
    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row)))

    separator = "-+-".join("-" * w for w in col_widths)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    num_trials = 10
    backend_devices = [("gloo", "cpu")]
    if torch.cuda.is_available():
        backend_devices.append(("nccl", "cuda"))
    data_sizes = [int(1e6), int(1e7), int(1e8), int(1e9)]
    world_sizes = [2, 4]
    results: list[dict] = []
    spawn_ctx = mp.get_context("spawn")
    result_queue = spawn_ctx.SimpleQueue()

    for (backend, device), data_size, world_size in itertools.product(
        backend_devices, data_sizes, world_sizes
    ):
        print(
            f"Benchmarking backend: {backend}, device: {device}, "
            f"data_size: {data_size / 1e6:.1f} MB, world_size: {world_size}, "
            f"trials: {num_trials}"
        )
        mp.spawn(
            fn=run_comm_bench,
            args=(world_size, result_queue, backend, device, data_size, num_trials),
            nprocs=world_size,
            join=True,
        )
        results.append(result_queue.get())

    if results:
        print("\nAggregated Results")
        print_results_table(results)

# Aggregated Results
# Backend | Device | World Size | Data Size (MB) | Trials | Avg Time (ms)
# --------+--------+------------+----------------+--------+--------------
# gloo    | cpu    | 2          | 1.0            | 10     | 2.151        
# gloo    | cpu    | 4          | 1.0            | 10     | 4.694        
# gloo    | cpu    | 2          | 10.0           | 10     | 8.401        
# gloo    | cpu    | 4          | 10.0           | 10     | 26.090       
# gloo    | cpu    | 2          | 100.0          | 10     | 98.225       
# gloo    | cpu    | 4          | 100.0          | 10     | 202.692      
# gloo    | cpu    | 2          | 1000.0         | 10     | 1061.326     
# gloo    | cpu    | 4          | 1000.0         | 10     | 2253.250 