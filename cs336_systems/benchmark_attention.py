import timeit
import statistics
import contextlib
import argparse
import torch
import torch.nn as nn

from cs336_basics.attention import MultiHeadSelfAttention


def benchmark(
    model: nn.Module,
    input_data: torch.Tensor,
    steps: int,
    device: str = "cuda",
    backward_pass: bool = True,
) -> None:
    grad_ctx = contextlib.nullcontext() if backward_pass else torch.no_grad()

    for _ in range(steps):
        with grad_ctx:
            output = model(input_data)

        if backward_pass:
            model.zero_grad(set_to_none=True)
            output.mean().backward()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()


def format_table(rows: list[dict[str, str]]) -> str:
    headers = ["d_model", "seq_len", "status", "mean", "min", "max", "std"]
    widths = {
        col: max(len(col), *(len(row[col]) for row in rows)) if rows else len(col)
        for col in headers
    }
    sep = "-+-".join("-" * widths[col] for col in headers)
    lines = [
        " | ".join(col.ljust(widths[col]) for col in headers),
        sep,
    ]
    for row in rows:
        lines.append(" | ".join(row[col].ljust(widths[col]) for col in headers))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the final summary table.",
    )
    args = parser.parse_args()

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    batch_size = 4
    vocab_size = 10000
    device = "cuda"
    num_heads = 1
    warmup_steps = 5
    compile = False
    summary_rows: list[dict[str, str]] = []

    for d_model in d_models:
        for seq_len in seq_lens:
            print("Benchmarking d_model:", d_model, "seq_len:", seq_len)

            model = MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                in_features=d_model,
                rope=None,
                max_seq_len=seq_len,
            )
            if compile:
                model = torch.compile(model=model, fullgraph=False)

            model.eval()
            input_data = torch.empty(
                (batch_size, seq_len, d_model),
                dtype=torch.float32,
                device=device,
            )
            std = (1 / d_model) ** 0.5
            nn.init.trunc_normal_(input_data, mean=0.0, std=std, a=-3 * std, b=3 * std)

            model = model.to(device)

            try:
                for _ in range(warmup_steps):
                    benchmark(
                        model,
                        input_data,
                        steps=1,
                        device=device,
                        backward_pass=True,
                    )
            except torch.OutOfMemoryError:
                print("OOMed!! Will skip")
                summary_rows.append(
                    {
                        "d_model": str(d_model),
                        "seq_len": str(seq_len),
                        "status": "OOM",
                        "mean": "NA",
                        "min": "NA",
                        "max": "NA",
                        "std": "NA",
                    }
                )
                continue

            times = timeit.repeat(
                lambda: benchmark(
                    model,
                    input_data,
                    steps=1,
                    device=device,
                    backward_pass=True,
                ),
                number=1,
                repeat=100,
            )

            per_call = [t / 1 for t in times]
            stats = {
                "mean": statistics.mean(per_call),
                "min": min(per_call),
                "max": max(per_call),
                "std": statistics.stdev(per_call) if len(per_call) > 1 else 0.0,
            }
            print(stats)
            summary_rows.append(
                {
                    "d_model": str(d_model),
                    "seq_len": str(seq_len),
                    "status": "OK",
                    "mean": f"{stats['mean']:.9f}",
                    "min": f"{stats['min']:.9f}",
                    "max": f"{stats['max']:.9f}",
                    "std": f"{stats['std']:.9f}",
                }
            )

    table = format_table(summary_rows)
    print("\nSummary Table")
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nSaved summary table to {args.out}")
