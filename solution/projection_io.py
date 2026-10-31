"""Fuse compact projection gather/cast and scatter/zero-fill, in both directions."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def projection_gather_kernel(X, INDICES, OUT, ROWS, WIDTH, COLS: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    full = tl.load(INDICES + rows, rows < ROWS, other=0).to(tl.int32)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    data = tl.load(X + full[:, None] * WIDTH + cols[None, :], mask, other=0)
    tl.store(OUT + rows[:, None] * WIDTH + cols[None, :], data, mask)


@triton.jit
def projection_scatter_kernel(X, INVERSE, OUT, ROWS, WIDTH, COLS: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    compact = tl.load(INVERSE + rows, rows < ROWS, other=-1).to(tl.int32)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
    data = tl.load(X + compact[:, None] * WIDTH + cols[None, :],
                   mask & (compact[:, None] >= 0), other=0)
    tl.store(OUT + rows[:, None] * WIDTH + cols[None, :], data, mask)


def _gather(x, indices, dtype):
    x = x.contiguous()
    width = x.shape[-1]
    out = torch.empty((indices.numel(), width), device=x.device, dtype=dtype)
    cols = min(triton.next_power_of_2(width), 256)
    projection_gather_kernel[(triton.cdiv(indices.numel(), 4), triton.cdiv(width, cols))](
        x, indices, out, indices.numel(), width, cols, num_warps=4)
    return out


def _scatter(x, inverse, shape, dtype):
    x = x.contiguous()
    out = torch.empty(shape, device=x.device, dtype=dtype)
    width = x.shape[-1]
    cols = min(triton.next_power_of_2(width), 256)
    projection_scatter_kernel[(triton.cdiv(inverse.numel(), 4), triton.cdiv(width, cols))](
        x, inverse, out, inverse.numel(), width, cols, num_warps=4)
    return out


class PackedInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, inverse):
        ctx.save_for_backward(inverse)
        ctx.shape, ctx.dtype = x.shape, x.dtype
        return _gather(x, indices, torch.float16)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        return _scatter(grad, inverse, ctx.shape, ctx.dtype), None, None


class FullOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, inverse, shape, dtype):
        ctx.save_for_backward(indices)
        ctx.dtype = x.dtype
        return _scatter(x, inverse, shape, dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return _gather(grad, indices, ctx.dtype), None, None, None, None
