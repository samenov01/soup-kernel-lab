"""Qwen2 rotary embeddings: one pointwise pass per tensor, with exact cast order."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def rope_transform_kernel(X, COS, SIN, OUT, TOTAL, HEADS: tl.constexpr,
        LENGTH: tl.constexpr, DIM: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
        S2: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
        BACKWARD: tl.constexpr, TILE: tl.constexpr):
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = idx < TOTAL
    d = idx % DIM
    t = idx // DIM % LENGTH
    h = idx // (DIM * LENGTH) % HEADS
    b = idx // (DIM * LENGTH * HEADS)
    other_d = tl.where(d < DIM // 2, d + DIM // 2, d - DIM // 2)
    base = b * S0 + h * S1 + t * S2
    a = tl.load(X + base + d, live, other=0)
    paired = tl.load(X + base + other_d, live, other=0)
    c = tl.load(COS + b * C0 + t * C1 + d * C2, live, other=0)
    if BACKWARD:
        s = tl.load(SIN + b * C0 + t * C1 + other_d * C2, live, other=0)
        sign = tl.where(d < DIM // 2, 1., -1.)
    else:
        s = tl.load(SIN + b * C0 + t * C1 + d * C2, live, other=0)
        sign = tl.where(d < DIM // 2, -1., 1.)
    tl.store(OUT + idx, a * c + (paired * sign) * s, live)


def _transform(x, cos, sin, backward):
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    c0 = 0 if cos.shape[0] == 1 else cos.stride(0)
    rope_transform_kernel[(triton.cdiv(x.numel(), 512),)](
        x, cos, sin, out, x.numel(), x.shape[1], x.shape[2], x.shape[3],
        *x.stride()[:3], c0, cos.stride(1), cos.stride(2), backward, 512,
        num_warps=4, enable_fp_fusion=False)
    return out


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _transform(q, cos, sin, False), _transform(k, cos, sin, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        return _transform(dq, cos, sin, True), _transform(dk, cos, sin, True), None, None


def patch_rope():
    from transformers.models.qwen2 import modeling_qwen2
    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_rope", False):
        return

    def rotary(q, k, cos, sin, unsqueeze_dim=1):
        supported = (
            unsqueeze_dim == 1 and q.is_cuda and q.numel() > 0 and q.dtype == k.dtype == torch.float32
            and q.ndim == k.ndim == 4 and q.shape[0] == k.shape[0]
            and q.shape[2:] == k.shape[2:] and q.shape[-1] % 2 == 0
            and q.stride(-1) == k.stride(-1) == 1
            and cos.ndim == sin.ndim == 3 and cos.shape == sin.shape
            and cos.shape[0] in (1, q.shape[0]) and cos.shape[1:] == q.shape[2:]
            and cos.dtype == sin.dtype == torch.float32 and cos.device == sin.device == q.device == k.device
            and cos.stride() == sin.stride() and not cos.requires_grad and not sin.requires_grad
        )
        if not supported:
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return _RoPE.apply(q, k, cos, sin)

    rotary._soup_rope = True
    modeling_qwen2.apply_rotary_pos_emb = rotary
