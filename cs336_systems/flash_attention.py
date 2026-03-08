from typing import Any
import torch
import math
from torch import Tensor
import triton
import triton.language as tl
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
    ) -> Float[Tensor, "... q dv"]:
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
    def backward(
        ctx: Any,
        dO: Float[Tensor, "... q dv"],
    ):
        Q, K, V, O, L = ctx.saved_tensors
        d = Q.shape[-1]

        D = O * dO
        D = torch.sum(D, dim=-1, keepdim=True)

        inv_sqrt_d = 1 / math.sqrt(d)
        S = (
            einx.dot(
                "... q d, ... k d -> ... q k",
                Q,
                K,
            )
            * inv_sqrt_d
        )
        P = torch.exp(S - L.unsqueeze(-1))
        dV = einx.dot(
            "... q k, ... q dv -> ... k dv",
            P,
            dO,
        )
        dP = einx.dot(
            "... q dv, ... k dv -> ... q k",
            dO,
            V,
        )
        dS = P * (dP - D)  # ... q k
        dQ = einx.dot("... q k, ... k d -> ... q d", dS, K) * inv_sqrt_d
        dK = einx.dot("... q k, ... q d -> ... k d", dS, Q) * inv_sqrt_d
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)

    q_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(0, 1),
    )

    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)
    L_ptrs = L_ptr + batch_index * stride_lb + offs_q * stride_lq

    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    l_ij = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_ij = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_end = q_start + Q_TILE_SIZE - 1
    offs_q = q_start + tl.arange(0, Q_TILE_SIZE)

    total_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        n_k_tiles = tl.cdiv(tl.minimum(q_end + 1, N_KEYS), K_TILE_SIZE)
    else:
        n_k_tiles = total_k_tiles

    for k_tile_idx in range(n_k_tiles):
        k_start = k_tile_idx * K_TILE_SIZE
        k_end = k_start + K_TILE_SIZE - 1

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(q, k) * scale

        if is_causal and not (k_end <= q_start):
            offs_k = k_start + tl.arange(0, K_TILE_SIZE)
            mask = offs_q[:, None] >= offs_k[None, :]
            s = tl.where(mask, s, -1e6)

        m_ij = tl.maximum(m_i, tl.max(s, axis=-1))
        p = tl.exp(s - m_ij[:, None])
        m_diff_exp = tl.exp(m_i - m_ij)

        l_ij = m_diff_exp * l_ij + tl.sum(p, axis=-1)
        p = p.to(v.dtype)
        o_ij = m_diff_exp[:, None] * o_ij
        o_ij = tl.dot(p, v, acc=o_ij)
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o_i = o_ij / l_ij[:, None]
    L_i = m_i + tl.log(l_ij)

    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_ptrs, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        assert Q.ndim == K.ndim == V.ndim == 3, "Expected [B, N, D] tensors."
        assert Q.shape[0] == K.shape[0] == V.shape[0], "Batch sizes must match."
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Last dim D must match."
        assert K.shape[-2] == V.shape[-2], "K and V must have same sequence length."
        assert (
            Q.is_cuda and K.is_cuda and V.is_cuda
        ), "Triton kernel requires CUDA tensors."
        assert (
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        ), "This wrapper assumes contiguous inputs."

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        # Choose tile sizes. These can be tuned later.
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        O = torch.empty_like(Q)
        L = torch.empty(
            (B, N_QUERIES),
            device=Q.device,
            dtype=torch.float32,
        )

        # Triton launch grid:
        #   pid(0) = batch_index
        #   pid(1) = query_tile_index
        grid = (
            B,
            triton.cdiv(N_QUERIES, Q_TILE_SIZE),
        )

        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[grid](
            Q,  # Q_ptr
            K,  # K_ptr
            V,  # V_ptr
            O,  # O_ptr
            L,  # L_ptr
            Q.stride(0),  # stride_qb
            Q.stride(1),  # stride_qq
            Q.stride(2),  # stride_qd
            K.stride(0),  # stride_kb
            K.stride(1),  # stride_kk
            K.stride(2),  # stride_kd
            V.stride(0),  # stride_vb
            V.stride(1),  # stride_vk
            V.stride(2),  # stride_vd
            O.stride(0),  # stride_ob
            O.stride(1),  # stride_oq
            O.stride(2),  # stride_od
            L.stride(0),  # stride_lb
            L.stride(1),  # stride_lq
            N_QUERIES,
            N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # Save for backward / testing
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE

        return O

    @staticmethod
    def backward(ctx: Any, dO: Tensor):
        raise NotImplementedError("Backward not implemented yet.")
