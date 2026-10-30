"""Original fused RMSNorm kernels for Qwen's frozen fp32 normalization layers.

The supported path matches prepare_model_for_kbit_training's frozen norms.
Trainable weights and mixed precision inputs retain the original implementation.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def rmsnorm_forward_kernel(X, W, Y, INV, WIDTH: tl.constexpr,
                            EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(X + row * WIDTH + col, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    mean_square = tl.sum(x * x, 0) / WIDTH
    inv = tl.rsqrt(mean_square + EPS)
    tl.store(INV + row, inv)
    tl.store(Y + row * WIDTH + col, (x * inv) * w, mask)


@triton.jit
def rmsnorm_backward_kernel(DY, X, W, INV, DX, WIDTH: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(X + row * WIDTH + col, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    dy = tl.load(DY + row * WIDTH + col, mask, other=0.0)
    inv = tl.load(INV + row)
    normalized = x * inv
    weighted_grad = dy * w
    projection = tl.sum(weighted_grad * normalized, 0) / WIDTH
    dx = inv * (weighted_grad - normalized * projection)
    tl.store(DX + row * WIDTH + col, dx, mask)


class _FrozenRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        width = x.shape[-1]
        rows = x.numel() // width
        out = torch.empty_like(x)
        inv = torch.empty(rows, device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            rmsnorm_forward_kernel[(rows,)](
                x, weight, out, inv, width, eps, triton.next_power_of_2(width), num_warps=4,
            )
        ctx.save_for_backward(x, weight, inv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        dx = torch.empty_like(x)
        width = x.shape[-1]
        with torch.cuda.device(x.device):
            rmsnorm_backward_kernel[(x.numel() // width,)](
                grad.contiguous(), x, weight, inv, dx, width,
                triton.next_power_of_2(width), num_warps=4,
            )
        return dx, None, None


def supported(x, weight, eps):
    return (
        isinstance(x, torch.Tensor) and isinstance(weight, torch.Tensor)
        and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
        and x.ndim > 0 and x.numel() > 0 and 0 < x.shape[-1] <= 16384
        and weight.device == x.device and weight.dtype == torch.float32
        and weight.shape == (x.shape[-1],) and weight.is_contiguous()
        and not weight.requires_grad and isinstance(eps, (int, float))
        and math.isfinite(eps) and eps > 0
    )


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward
    if getattr(original, "_soup_rmsnorm_patch", False):
        return

    def forward(self, hidden_states):
        if not supported(hidden_states, self.weight, self.variance_epsilon):
            return original(self, hidden_states)
        return _FrozenRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_rmsnorm_patch = True
    modeling_qwen2.Qwen2RMSNorm.forward = forward
