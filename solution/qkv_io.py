"""Move three Q/K/V projections together, with explicit compact autograd IO."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from .projection_io import _gather, _scatter


@triton.jit
def qkv_scatter_kernel(X, INVERSE, Q, K, V, ROWS, QW: tl.constexpr, KW: tl.constexpr,
                       COLS: tl.constexpr):
    r = tl.program_id(0) * 4 + tl.arange(0, 4)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    small = tl.load(INVERSE + r, r < ROWS, other=-1).to(tl.int32)
    live = (r[:, None] < ROWS) & (small[:, None] >= 0)
    src = small[:, None] * (QW + 2 * KW) + c[None, :]
    q = tl.load(X + src, live & (c[None, :] < QW), other=0)
    k = tl.load(X + src + QW, live & (c[None, :] < KW), other=0)
    v = tl.load(X + src + QW + KW, live & (c[None, :] < KW), other=0)
    tl.store(Q + r[:, None] * QW + c[None, :], q, (r[:, None] < ROWS) & (c[None, :] < QW))
    tl.store(K + r[:, None] * KW + c[None, :], k, (r[:, None] < ROWS) & (c[None, :] < KW))
    tl.store(V + r[:, None] * KW + c[None, :], v, (r[:, None] < ROWS) & (c[None, :] < KW))


@triton.jit
def qkv_gradient_gather_kernel(Q, K, V, INDICES, OUT, ROWS,
                               QW: tl.constexpr, KW: tl.constexpr, COLS: tl.constexpr):
    r = tl.program_id(0) * 4 + tl.arange(0, 4)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    full = tl.load(INDICES + r, r < ROWS, other=0).to(tl.int32)
    live = r[:, None] < ROWS
    q = tl.load(Q + full[:, None] * QW + c[None, :], live & (c[None, :] < QW), other=0)
    k = tl.load(K + full[:, None] * KW + c[None, :] - QW,
                live & (c[None, :] >= QW) & (c[None, :] < QW + KW), other=0)
    v = tl.load(V + full[:, None] * KW + c[None, :] - QW - KW,
                live & (c[None, :] >= QW + KW) & (c[None, :] < QW + 2 * KW), other=0)
    tl.store(OUT + r[:, None] * (QW + 2 * KW) + c[None, :], q + k + v,
             live & (c[None, :] < QW + 2 * KW))


class _FullQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, bias, sizes, indices, inverse):
        import bitsandbytes as bnb
        small = _gather(x, indices, torch.float16)
        result = bnb.matmul_4bit(small, packed, state, bias=bias)
        qwidth, kwidth, vwidth = sizes
        assert kwidth == vwidth
        shape = x.shape[:-1]
        q, k, v = [torch.empty((*shape, width), device=x.device, dtype=x.dtype) for width in sizes]
        qkv_scatter_kernel[(triton.cdiv(inverse.numel(), 4), triton.cdiv(qwidth, 256))](
            result, inverse, q, k, v, inverse.numel(), qwidth, kwidth, 256, num_warps=4)
        ctx.save_for_backward(packed, indices, inverse)
        ctx.state, ctx.shape, ctx.dtype, ctx.sizes = state, x.shape, x.dtype, sizes
        return q, k, v

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk, dv):
        from copy import copy
        from bitsandbytes.functional import dequantize_4bit
        packed, indices, inverse = ctx.saved_tensors
        qwidth, kwidth, _ = ctx.sizes
        combined = torch.empty((indices.numel(), qwidth + 2 * kwidth), device=dq.device, dtype=torch.float16)
        qkv_gradient_gather_kernel[(triton.cdiv(indices.numel(), 4), triton.cdiv(combined.shape[1], 256))](
            dq.contiguous(), dk.contiguous(), dv.contiguous(), indices, combined, indices.numel(),
            qwidth, kwidth, 256, num_warps=4)
        state = copy(ctx.state)
        state.dtype = torch.float16
        dx = torch.matmul(combined, dequantize_4bit(packed, state))
        return _scatter(dx, inverse, ctx.shape, ctx.dtype), None, None, None, None, None, None
