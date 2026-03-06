from typing import Any
import torch
import math
from torch import Tensor
from jaxtyping import Float
import einx


class FlashAttentionPytorch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        Q: Float[Tensor, "... q d"],
        K: Float[Tensor, "... k d"],
        V: Float[Tensor, "... k dv"],
        is_causal: bool = False,
    ) -> tuple[Float[Tensor, "... q dv"], Float[Tensor, "... q"]]:
        assert not is_causal
        assert Q.shape[-1] == K.shape[-1]
        assert K.shape[-2] == V.shape[-2]

        d = Q.shape[-1]
        inv_d_sqrt = 1 / d**0.5
        Bq, Bk = 16, 16
        Tq = math.ceil(Q.shape[-2] / Bq)
        Tk = math.ceil(K.shape[-2] / Bk)

        O = torch.zeros(
            (*Q.shape[:-1], V.shape[-1]),
            device=Q.device,
            dtype=Q.dtype,
        )
        l = torch.zeros(
            Q.shape[:-1],
            device=Q.device,
            dtype=Q.dtype,
        )
        m = torch.full(
            Q.shape[:-1],
            fill_value=float("-inf"),
            device=Q.device,
            dtype=Q.dtype,
        )
        for i in range(Tq):
            q_slice = slice(i * Bq, min((i + 1) * Bq, Q.shape[-2]))
            cur_m = m[..., q_slice]

            for j in range(Tk):
                k_slice = slice(j * Bk, min((j + 1) * Bk, K.shape[-2]))
                S = (
                    einx.dot(
                        "... q d, ... k d -> ... q k",
                        Q[..., q_slice, :],
                        K[..., k_slice, :],
                    )
                    * inv_d_sqrt
                )
                row_max = torch.max(S, dim=-1).values
                new_m = torch.maximum(cur_m, row_max)

                P = torch.exp(S - new_m.unsqueeze(-1))
                m_diff_exp = torch.exp(cur_m - new_m)

                l[..., q_slice] = m_diff_exp.mul(l[..., q_slice]) + P.sum(dim=-1)

                O[..., q_slice, :] = einx.multiply(
                    "... q, ... q d -> ... q d",
                    m_diff_exp,
                    O[..., q_slice, :],
                ) + einx.dot(
                    "... q k, ... k d -> ... q d",
                    P,
                    V[..., k_slice, :],
                )
                m[..., q_slice] = new_m
                cur_m = new_m

            O[..., q_slice, :] = einx.multiply(
                "... q, ... q d -> ... q d",
                1 / l[..., q_slice],
                O[..., q_slice, :],
            )
            l[..., q_slice] = torch.log(l[..., q_slice]) + cur_m

        ctx.save_for_backward(Q, K, V, O, l)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx: Any):
        raise NotImplementedError
