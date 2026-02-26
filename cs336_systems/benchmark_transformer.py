import timeit
import statistics
import contextlib
import argparse
import torch

from cs336_basics.transformer import Transformer


def benchmark(
    model: Transformer,
    input_data: torch.Tensor,
    steps: int,
    device: str = "mps",
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
    headers = ["model_name", "compile", "status", "mean", "min", "max", "std"]
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

    model_sizes = {
        "small": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        # "xl": {
        #     "d_model": 1600,
        #     "d_ff": 6400,
        #     "num_layers": 48,
        #     "num_heads": 25,
        # },
        # "2.7b": {
        #     "d_model": 2560,
        #     "d_ff": 10240,
        #     "num_layers": 32,
        #     "num_heads": 32,
        # },
    }
    backward_pass = True
    summary_rows: list[dict[str, str]] = []

    for model_name, model_params in model_sizes.items():
        for compile in [True, False]:
            print(
                "Benchmarking model:",
                model_name,
                "compile:",
                compile,
                "backward_pass:",
                backward_pass,
            )

            batch_size = 4
            context_length = 1024
            vocab_size = 10000
            device = "cuda"
            warmup_steps = 5

            model = Transformer(
                vocab_size=vocab_size,
                context_length=context_length,
                rope_theta=10000.0,
                **model_params,
            )
            if compile:
                model = torch.compile(model=model)
            model.eval()
            input_data = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, context_length),
                device=device,
                dtype=torch.long,
            )

            model = model.to(device)

            try:
                for _ in range(warmup_steps):
                    benchmark(
                        model,
                        input_data,
                        steps=1,
                        device=device,
                        backward_pass=backward_pass,
                    )
            except torch.OutOfMemoryError:
                print("OOMed!! Will skip")
                summary_rows.append(
                    {
                        "model_name": model_name,
                        "compile": str(compile),
                        "status": "OOM",
                        "mean": "NA",
                        "min": "NA",
                        "max": "NA",
                        "std": "NA",
                    }
                )
                continue

            try:
                times = timeit.repeat(
                    lambda: benchmark(
                        model,
                        input_data,
                        steps=1,
                        device=device,
                        backward_pass=backward_pass,
                    ),
                    number=1,
                    repeat=3,
                )
            except torch.OutOfMemoryError:
                print("OOMed during timing!! Will skip")
                summary_rows.append(
                    {
                        "model_name": model_name,
                        "compile": str(compile),
                        "status": "OOM",
                        "mean": "NA",
                        "min": "NA",
                        "max": "NA",
                        "std": "NA",
                    }
                )
                continue

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
                    "model_name": model_name,
                    "compile": str(compile),
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
