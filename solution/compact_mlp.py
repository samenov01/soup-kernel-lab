"""Retain the original fp16 projection values instead of redundant fp32 copies.

SwiGLU still computes in fp32 and rounds at the same down-projection boundary.
The complete MLP returns fp32, as the original fp32-input MLP did.
The organizer's kernels are reused unchanged and are never claimed as ours.
"""

import torch
import triton
from torch.autograd.function import once_differentiable

from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel

_gate_up = None


class _CompactSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        # Fuse the fp16 conversion that Linear4bit's down projection performed
        # immediately after the original fp32 SwiGLU result.
        out = torch.empty_like(gate)
        example_swiglu_fwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            gate, up, out, gate.numel(), BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        # The original path computed in fp32 then cast each projection gradient
        # separately to fp16 at Linear4bit's output cast. Preserve that order.
        example_swiglu_bwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            grad.contiguous(), gate, up, dgate, dup, gate.numel(), BLOCK=BLOCK)
        return dgate, dup


def patch_compact_mlp():
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward
    if getattr(original, "_soup_compact_patch", False):
        return

    def forward(self, x):
        projections = (self.gate_proj, self.up_proj, self.down_proj)
        supported = (
            x.is_cuda and x.dtype == torch.float32 and self.config.hidden_act == "silu"
            and all(type(layer) is Linear4bit and layer.compute_dtype == torch.float16
                    and layer.compute_type_is_set and layer.bias is None
                    and getattr(layer.weight, "quant_state", None) is not None
                    and not any(p.requires_grad for p in layer.parameters())
                    for layer in projections)
        )
        if not supported:
            return original(self, x)
        # Separate casts preserve the original fp32 accumulation of the two
        # input-gradient branches. Sharing a cast would add them in fp16.
        if _gate_up is None:
            gate = self.gate_proj(x.to(torch.float16))
            up = self.up_proj(x.to(torch.float16))
        else:
            gate, up = _gate_up(self, x)
        return self.down_proj(_CompactSwiGLU.apply(gate, up)).float()

    forward._soup_compact_patch = True
    modeling_qwen2.Qwen2MLP.forward = forward
