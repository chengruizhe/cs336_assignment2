import torch
import torch.distributed as dist
import torch.nn as nn


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
