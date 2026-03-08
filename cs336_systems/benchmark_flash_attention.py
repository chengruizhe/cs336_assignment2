import argparse
from typing import Callable

import torch
import triton

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Row = dict[str, str]

HEADERS = [
    "impl",
    "dtype",
    "seq_len",
    "d_model",
    "status",
    "fwd_ms",
    "bwd_ms",
    "e2e_ms",
]
DTYPES = [torch.bfloat16, torch.float32]
D_MODELS = [16, 32, 64, 128]
IMPLEMENTATIONS = ["triton"]  # ["triton", "pytorch"]


def _select_device() -> tuple[torch.device, list[str], str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    gpu_names = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    h100_index = next(
        (i for i, name in enumerate(gpu_names) if "H100" in name.upper()), None
    )
    chosen_index = h100_index if h100_index is not None else 0
    reason = "H100 found" if h100_index is not None else "H100 not found; using cuda:0"
    return torch.device(f"cuda:{chosen_index}"), gpu_names, reason


def _get_attention_impls() -> dict[str, AttentionFn]:
    from cs336_systems.flash_attention import (
        FlashAttentionPytorch,
        FlashAttentionTriton,
    )

    def triton_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return FlashAttentionTriton.apply(q, k, v, True)

    def pytorch_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return FlashAttentionPytorch.apply(q, k, v, False)

    # compiled_pytorch_impl = torch.compile(pytorch_impl, fullgraph=False)

    return {
        "triton": triton_impl,
        "pytorch": pytorch_impl,
    }


def _format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _make_row(
    impl: str,
    dtype: torch.dtype,
    seq_len: int,
    d_model: int,
    status: str,
    fwd_ms: float | None,
    bwd_ms: float | None,
    e2e_ms: float | None,
) -> Row:
    return {
        "impl": impl,
        "dtype": str(dtype).replace("torch.", ""),
        "seq_len": str(seq_len),
        "d_model": str(d_model),
        "status": status,
        "fwd_ms": _format_ms(fwd_ms),
        "bwd_ms": _format_ms(bwd_ms),
        "e2e_ms": _format_ms(e2e_ms),
    }


def _print_table(rows: list[Row]) -> None:
    widths = {
        key: max(len(key), *(len(row[key]) for row in rows)) if rows else len(key)
        for key in HEADERS
    }

    def format_row(row: Row) -> str:
        return " | ".join(row[key].ljust(widths[key]) for key in HEADERS)

    print(format_row({key: key for key in HEADERS}))
    print("-+-".join("-" * widths[key] for key in HEADERS))
    for row in rows:
        print(format_row(row))


def _write_csv(out_path: str, rows: list[Row]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(HEADERS) + "\n")
        for row in rows:
            f.write(",".join(row[key] for key in HEADERS) + "\n")


def _bench_forward(
    fwd_fn: AttentionFn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> float:
    with torch.no_grad():
        return triton.testing.do_bench(lambda: fwd_fn(q, k, v))


def _bench_backward(
    output: torch.Tensor,
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> float:
    def run_backward() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        output.backward(grad_out, retain_graph=True)

    return triton.testing.do_bench(run_backward)


def _bench_e2e(
    fwd_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
) -> float:
    def run_e2e() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        out = fwd_fn(q, k, v)
        out.backward(grad_out)

    return triton.testing.do_bench(run_e2e)


def _benchmark_impl(
    fwd_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[str, float | None, float | None, float | None]:
    try:
        fwd_ms = _bench_forward(fwd_fn, q, k, v)
    except torch.OutOfMemoryError:
        return "OOM", None, None, None
    except NotImplementedError:
        return "NOT_IMPL", None, None, None

    q_req = q.detach().clone().requires_grad_(True)
    k_req = k.detach().clone().requires_grad_(True)
    v_req = v.detach().clone().requires_grad_(True)

    try:
        out = fwd_fn(q_req, k_req, v_req)
        grad_out = torch.randn_like(out)
    except torch.OutOfMemoryError:
        return "OOM", fwd_ms, None, None
    except NotImplementedError:
        return "NOT_IMPL", fwd_ms, None, None

    status = "OK"
    bwd_ms: float | None = None
    e2e_ms: float | None = None

    try:
        bwd_ms = _bench_backward(out, grad_out, q_req, k_req, v_req)
    except torch.OutOfMemoryError:
        status = "OOM"
    except NotImplementedError:
        status = "NO_BWD"

    try:
        e2e_ms = _bench_e2e(fwd_fn, q_req, k_req, v_req, grad_out)
    except torch.OutOfMemoryError:
        status = "OOM"
    except NotImplementedError:
        status = "NO_BWD"

    return status, fwd_ms, bwd_ms, e2e_ms


def _allocate_inputs(
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    d_model: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    k = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    v = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    return q, k, v


def run_benchmark(max_seq_len: int, out_path: str | None) -> None:
    device, gpu_names, reason = _select_device()
    torch.cuda.set_device(device)

    print("Detected GPUs:")
    for idx, name in enumerate(gpu_names):
        print(f"  cuda:{idx} -> {name}")
    print(f"Selected device: {device} ({reason})")
    print("Benchmark config: batch_size=1, causal=True")

    seq_lens = [2**i for i in range(7, 16) if 2**i <= max_seq_len]
    impls = _get_attention_impls()
    rows: list[Row] = []

    for dtype in DTYPES:
        for seq_len in seq_lens:
            for d_model in D_MODELS:
                print(f"Running dtype={dtype}, seq_len={seq_len}, d_model={d_model}")
                torch.cuda.empty_cache()

                try:
                    q, k, v = _allocate_inputs(device, dtype, seq_len, d_model)
                except torch.OutOfMemoryError:
                    for impl_name in IMPLEMENTATIONS:
                        rows.append(
                            _make_row(
                                impl_name,
                                dtype,
                                seq_len,
                                d_model,
                                "OOM_INPUT",
                                None,
                                None,
                                None,
                            )
                        )
                    continue

                for impl_name in IMPLEMENTATIONS:
                    status, fwd_ms, bwd_ms, e2e_ms = _benchmark_impl(
                        impls[impl_name], q, k, v
                    )
                    rows.append(
                        _make_row(
                            impl_name,
                            dtype,
                            seq_len,
                            d_model,
                            status,
                            fwd_ms,
                            bwd_ms,
                            e2e_ms,
                        )
                    )

    print("\nResults")
    _print_table(rows)

    if out_path:
        _write_csv(out_path, rows)
        print(f"\nSaved CSV to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=65536,
        help="Maximum sequence length in the power-of-2 sweep starting at 128.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional CSV output path for the table rows.",
    )
    args = parser.parse_args()

    run_benchmark(max_seq_len=args.max_seq_len, out_path=args.out)


if __name__ == "__main__":
    main()
