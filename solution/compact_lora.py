"""Keep LoRA A/B activations compact; preserve full-shape native dropout."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def sparse_lora_add_kernel(BASE, DELTA, INVERSE, OUT, ROWS, WIDTH, SCALE, COLS: tl.constexpr):
    r = tl.program_id(0) * 4 + tl.arange(0, 4)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    live = (r[:, None] < ROWS) & (c[None, :] < WIDTH)
    small = tl.load(INVERSE + r, r < ROWS, other=-1).to(tl.int32)
    base = tl.load(BASE + r[:, None] * WIDTH + c[None, :], live, other=0)
    delta = tl.load(DELTA + small[:, None] * WIDTH + c[None, :],
                    live & (small[:, None] >= 0), other=0).to(tl.float32)
    # Native PEFT scales the half tensor first, then adds to the fp32 base.
    scaled = (delta * SCALE).to(tl.float16).to(tl.float32)
    tl.store(OUT + r[:, None] * WIDTH + c[None, :], base + scaled, live)


@triton.jit
def sparse_lora_gradient_kernel(GRAD, INDICES, DY, ROWS, WIDTH, SCALE, COLS: tl.constexpr):
    r = tl.program_id(0) * 4 + tl.arange(0, 4)
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)
    full = tl.load(INDICES + r, r < ROWS, other=0).to(tl.int32)
    live = (r[:, None] < ROWS) & (c[None, :] < WIDTH)
    g = tl.load(GRAD + full[:, None] * WIDTH + c[None, :], live, other=0)
    # AddBackward casts to half before MulBackward applies the LoRA scale.
    scaled = g.to(tl.float16).to(tl.float32) * SCALE
    tl.store(DY + r[:, None] * WIDTH + c[None, :], scaled, live)


class SparseLoRAAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, delta, indices, inverse, scale):
        base, delta = base.contiguous(), delta.contiguous()
        width = base.shape[-1]
        cols = min(triton.next_power_of_2(width), 256)
        out = torch.empty_like(base)
        sparse_lora_add_kernel[(triton.cdiv(inverse.numel(), 4), triton.cdiv(width, cols))](
            base, delta, inverse, out, inverse.numel(), width, scale, cols,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(indices)
        ctx.shape, ctx.scale = delta.shape, scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        grad = grad.contiguous()
        width = ctx.shape[-1]
        cols = min(triton.next_power_of_2(width), 256)
        dy = torch.empty(ctx.shape, device=grad.device, dtype=torch.float16)
        sparse_lora_gradient_kernel[(triton.cdiv(indices.numel(), 4), triton.cdiv(width, cols))](
            grad, indices, dy, indices.numel(), width, ctx.scale, cols, num_warps=4)
        return grad, dy, None, None, None


def patch_compact_lora():
    from peft.tuners.lora.bnb import Linear4bit
    from .padded_attention import _PROJECTIONS
    original = Linear4bit.forward
    if getattr(original, "_soup_compact_lora", False):
        return

    def forward(self, x, *args, **kwargs):
        tokens = _PROJECTIONS.get()
        adapters = self.active_adapters
        supported = (
            tokens is not None and len(tokens) == 3 and tokens[0].numel() > 0
            and self.training and x.is_cuda and x.dtype == torch.float32
            and x.ndim == 3 and x.shape[:2] == tokens[1]
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
            and not args and not kwargs and not self.disable_adapters and not self.merged
            and len(adapters) == 1 and adapters[0] in self.lora_A and not self.lora_variant
            and type(self.lora_A[adapters[0]]) is torch.nn.Linear
            and type(self.lora_B[adapters[0]]) is torch.nn.Linear
            # Exact powers of two preserve the native wrapped-scalar rounding.
            and self.scaling[adapters[0]] in (0.5, 1., 2., 4., 8., 16., 32.)
        )
        if not supported:
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        result = self.base_layer(x)
        if result.dtype != torch.float32:
            raise RuntimeError("Compact LoRA expects the original fp32 base output")
        adapter = adapters[0]
        # Same dropout function, shape, dtype and order as pinned PEFT.
        dropped = self.lora_dropout[adapter](x)
        small = dropped.reshape(-1, dropped.shape[-1]).index_select(0, tokens[0])
        delta = self.lora_B[adapter](self.lora_A[adapter](small))
        if delta.dtype != torch.float16:
            raise RuntimeError("Compact LoRA expects configured fp16 autocast projections")
        return SparseLoRAAdd.apply(result, delta, tokens[0], tokens[2], self.scaling[adapter])

    forward._soup_compact_lora = True
    Linear4bit.forward = forward
