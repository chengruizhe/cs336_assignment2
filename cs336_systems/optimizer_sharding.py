from typing import Any
import copy
import warnings

import torch
import torch.optim as optim
import torch.distributed as dist


class StateShardingOptimizer(optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        optimizer_cls: type[optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed not initialized")

        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        self._params_sizes: list[int] = [0] * self._world_size
        self._local_param_groups: list[dict[str, Any]] = []
        self._param_owners: dict[int, int] = {}

        super().__init__(params, kwargs)

        self._optim = (
            optimizer_cls(self._local_param_groups, **kwargs)
            if self._local_param_groups
            else None
        )
        if self._optim is not None:
            self.state = self._optim.state

    def step(self, closure=None, **kwargs):
        loss = None
        if self._optim is not None:
            loss = self._optim.step(closure=closure, **kwargs)
            self.state = self._optim.state
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        seen_params: set[int] = set()
        for group in self.param_groups:
            for param in group["params"]:
                param_id = id(param)
                if param_id in seen_params:
                    continue
                seen_params.add(param_id)
                owner_rank = self._param_owners[param_id]
                dist.broadcast(param.data, src=owner_rank)

        return loss

    def _get_min_index(self, lst: list[int]) -> int:
        assert lst

        min_idx = 0
        min_val = lst[0]
        for idx, v in enumerate(lst):
            if v < min_val:
                min_idx = idx
                min_val = v
        return min_idx

    def add_param_group(self, param_group: dict[str, Any]):
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections; use a list instead."
            )
        else:
            param_group["params"] = list(params)

        extracted_param_tensors: list[torch.Tensor] = []
        extracted_param_names: list[str] = []
        for param in param_group["params"]:
            if isinstance(param, tuple):
                param_name, param_tensor = param
                extracted_param_names.append(param_name)
                extracted_param_tensors.append(param_tensor)
            else:
                extracted_param_tensors.append(param)

        param_group["params"] = extracted_param_tensors
        if extracted_param_names:
            if len(extracted_param_names) != len(extracted_param_tensors):
                raise ValueError(
                    "all optimizer params should be with/without names. Some param names are missing"
                )
            param_group["param_names"] = extracted_param_names

        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, but one of the params is "
                    + torch.typename(param)
                )
            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or param.retains_grad
            ):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters",
                stacklevel=3,
            )

        existing_params: set[torch.Tensor] = set()
        for group in self.param_groups:
            existing_params.update(group["params"])
            if ("param_names" in param_group) != ("param_names" in group):
                current_group_txt = "with names" if "param_names" in param_group else "without names"
                raise ValueError(
                    "all optimizer param groups should be with/without names. "
                    f"cannot add param group {current_group_txt} to the optimizer"
                )

        if not existing_params.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        # Greedily assign larger tensors first to keep total optimizer state balanced.
        params_sorted = sorted(param_group["params"], key=lambda t: t.numel(), reverse=True)
        for param in params_sorted:
            param_id = id(param)
            if param_id in self._param_owners:
                continue
            tgt_rank = self._get_min_index(self._params_sizes)
            self._params_sizes[tgt_rank] += param.numel()
            self._param_owners[param_id] = tgt_rank

        self.param_groups.append(param_group)

        local_group = copy.copy(param_group)
        local_params: list[torch.Tensor] = []
        local_param_names: list[str] = []
        param_names = param_group.get("param_names")
        for idx, param in enumerate(param_group["params"]):
            if self._param_owners[id(param)] != self._rank:
                continue
            local_params.append(param)
            if param_names is not None:
                local_param_names.append(param_names[idx])

        local_group["params"] = local_params
        if param_names is not None:
            local_group["param_names"] = local_param_names

        if local_params:
            self._local_param_groups.append(local_group)
            if getattr(self, "_optim", None) is not None:
                self._optim.add_param_group(local_group)
                self.state = self._optim.state

