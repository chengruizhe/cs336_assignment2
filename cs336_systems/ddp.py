import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before constructing DDP."
            )

        self._world_size = dist.get_world_size()
        self._pending_works: list[tuple[dist.Work, torch.Tensor]] = []

        # Start all ranks from rank 0's model state.
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)

        self._register_gradient_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_gradient_hooks(self) -> None:
        seen_params: set[int] = set()

        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            assert hasattr(param, "register_post_accumulate_grad_hook")

            def _post_accumulate_hook(p: torch.nn.Parameter) -> None:
                if self._world_size > 1 and p.grad is not None:
                    work = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self._pending_works.append((work, p.grad))

            param.register_post_accumulate_grad_hook(_post_accumulate_hook)

    def finish_gradient_synchronization(self):
        if self._world_size <= 1:
            return

        for work, grad in self._pending_works:
            work.wait()
            grad.div_(self._world_size)
        self._pending_works.clear()


class BucketDDP(DDP):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        self._bucket_size_bytes = (
            int(bucket_size_mb * 1024 * 1024)
            if bucket_size_mb is not None
            else float("inf")
        )
        self._buckets: list[dict] = []
        self._param_to_bucket_idx: dict[int, int] = {}
        super().__init__(module)

    def _reset_bucket_runtime_state(self) -> None:
        for bucket in self._buckets:
            bucket["ready_param_ids"] = set()
            bucket["grads"] = {}
            bucket["flat_grad"] = None
            bucket["work"] = None

    def _register_gradient_hooks(self) -> None:
        # Build buckets in reverse parameter order so comm can overlap with backward.
        seen_params: set[int] = set()
        params: list[torch.nn.Parameter] = []
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            p_id = id(p)
            if p_id in seen_params:
                continue
            seen_params.add(p_id)
            params.append(p)

        if not params:
            self._buckets = []
            return

        def _param_nbytes(p: torch.nn.Parameter) -> int:
            return p.numel() * p.element_size()

        current_bucket_params: list[torch.nn.Parameter] = []
        current_bucket_nbytes = 0

        def _flush_current_bucket() -> None:
            nonlocal current_bucket_params, current_bucket_nbytes
            if not current_bucket_params:
                return
            bucket_idx = len(self._buckets)
            bucket = {
                "params": list(current_bucket_params),
                "ready_param_ids": set(),
                "grads": {},
                "flat_grad": None,
                "work": None,
            }
            self._buckets.append(bucket)
            for bp in current_bucket_params:
                self._param_to_bucket_idx[id(bp)] = bucket_idx
            current_bucket_params = []
            current_bucket_nbytes = 0

        for p in params:
            p_nbytes = _param_nbytes(p)
            if (
                current_bucket_params
                and current_bucket_nbytes + p_nbytes > self._bucket_size_bytes
            ):
                _flush_current_bucket()
            current_bucket_params.append(p)
            current_bucket_nbytes += p_nbytes
        _flush_current_bucket()

        assert all(
            hasattr(p, "register_post_accumulate_grad_hook") for p in params
        ), "BucketDDP requires register_post_accumulate_grad_hook support."

        def _post_accumulate_hook(p: torch.nn.Parameter) -> None:
            if self._world_size <= 1 or p.grad is None:
                return

            p_id = id(p)
            bucket_idx = self._param_to_bucket_idx[p_id]
            bucket = self._buckets[bucket_idx]
            if p_id in bucket["ready_param_ids"]:
                return

            bucket["ready_param_ids"].add(p_id)
            bucket["grads"][p_id] = p.grad

            if len(bucket["ready_param_ids"]) == len(bucket["params"]):
                grads_in_bucket = [bucket["grads"][id(bp)] for bp in bucket["params"]]
                flat_grad = _flatten_dense_tensors(grads_in_bucket)
                work = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                bucket["flat_grad"] = flat_grad
                bucket["work"] = work

        for p in params:
            p.register_post_accumulate_grad_hook(_post_accumulate_hook)

        self._reset_bucket_runtime_state()

    def on_train_batch_start(self) -> None:
        self._reset_bucket_runtime_state()

    def finish_gradient_synchronization(self) -> None:
        if self._world_size <= 1:
            return

        for bucket in self._buckets:
            work = bucket["work"]
            if work is None:
                continue
            work.wait()
            flat_grad = bucket["flat_grad"]
            flat_grad.div_(self._world_size)

            grads_in_bucket = [bucket["grads"][id(p)] for p in bucket["params"]]
            synced_grads = _unflatten_dense_tensors(flat_grad, grads_in_bucket)
            for grad, synced_grad in zip(grads_in_bucket, synced_grads):
                grad.copy_(synced_grad)

        self._reset_bucket_runtime_state()
