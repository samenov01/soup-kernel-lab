"""Compact LoRA A/B autograd with direct gather/scatter of input gradients.

Dropout remains the original PEFT module outside this function, with its full
shape and RNG calls. Trainable FP32 parameters and Half GEMM cast order remain.
"""

import torch
import torch.nn.functional as F
import triton
from torch.autograd.function import once_differentiable
from .projection_io import _gather, _scatter
from .compact_lora import sparse_lora_add_kernel, sparse_lora_gradient_kernel


class CompactLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, dropped, a, b, indices, inverse, scale):
        small = _gather(dropped, indices, torch.float16)
        ahalf, bhalf = a.half(), b.half()
        hidden = F.linear(small, ahalf)
        delta = F.linear(hidden, bhalf)
        out = torch.empty_like(base)
        width = base.shape[-1]
        cols = min(triton.next_power_of_2(width), 256)
        sparse_lora_add_kernel[(triton.cdiv(inverse.numel(), 4), triton.cdiv(width, cols))](
            base.contiguous(), delta, inverse, out, inverse.numel(), width, scale, cols,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(small, hidden, ahalf, bhalf, indices, inverse)
        ctx.shape, ctx.dtype = dropped.shape, dropped.dtype
        ctx.weight_dtypes, ctx.scale = (a.dtype, b.dtype), scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        small, hidden, a, b, indices, inverse = ctx.saved_tensors
        grad = grad.contiguous()
        width = b.shape[0]
        dy = torch.empty((indices.numel(), width), device=grad.device, dtype=torch.float16)
        cols = min(triton.next_power_of_2(width), 256)
        sparse_lora_gradient_kernel[(triton.cdiv(indices.numel(), 4), triton.cdiv(width, cols))](
            grad, indices, dy, indices.numel(), width, ctx.scale, cols, num_warps=4)
        db = torch.matmul(dy.T, hidden).to(ctx.weight_dtypes[1])
        dh = torch.matmul(dy, b)
        da = torch.matmul(dh.T, small).to(ctx.weight_dtypes[0])
        dx = _scatter(torch.matmul(dh, a), inverse, ctx.shape, ctx.dtype)
        return grad, dx, da, db, None, None, None


def patch_lora_autograd():
    from peft.tuners.lora.bnb import Linear4bit
    from .padded_attention import _PROJECTIONS
    original = Linear4bit.forward
    if getattr(original, "_soup_lora_autograd", False):
        return

    def forward(self, x, *args, **kwargs):
        tokens = _PROJECTIONS.get()
        adapters = self.active_adapters
        if (tokens is None or len(tokens) != 3 or tokens[0].numel() == 0
                or not self.training or not x.is_cuda or x.dtype not in (torch.float16, torch.float32)
                or x.ndim != 3 or x.shape[:2] != tokens[1] or args or kwargs
                or not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16
                or self.disable_adapters or self.merged or len(adapters) != 1 or self.lora_variant
                or adapters[0] not in self.lora_A):
            return original(self, x, *args, **kwargs)
        name = adapters[0]
        a, b = self.lora_A[name], self.lora_B[name]
        if (type(a) is not torch.nn.Linear or type(b) is not torch.nn.Linear
                or a.bias is not None or b.bias is not None
                or a.weight.dtype != torch.float32 or b.weight.dtype != torch.float32
                or self.scaling[name] not in (0.5, 1., 2., 4., 8., 16., 32.)):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        base = self.base_layer(x)
        dropped = self.lora_dropout[name](x)
        return CompactLoRA.apply(base, dropped, a.weight, b.weight, tokens[0], tokens[2], self.scaling[name])

    forward._soup_lora_autograd = True
    Linear4bit.forward = forward
