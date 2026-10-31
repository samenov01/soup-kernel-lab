"""Frozen MLP autograd saving its narrow input, not two wide activations.

Only deterministic frozen gate/up products are recomputed. No dropout, RNG,
trainer checkpointing flag or model parameter is changed. The organizer's
SwiGLU kernels are used unchanged and remain excluded from own-kernel credit.
"""

from copy import copy
import torch
import triton
from torch.autograd.function import once_differentiable
from .swiglu import BLOCK, example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
from .merged_gate_up import packed_gate_up
from .half_activations import _HALF_MODEL


def _decode(packed, state):
    from bitsandbytes.functional import dequantize_4bit
    state = copy(state)
    state.dtype = torch.float16
    return dequantize_4bit(packed, state)


class _RecomputedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, down, down_state):
        import bitsandbytes as bnb
        merged = bnb.matmul_4bit(x, packed, state)
        gate, up = (part.contiguous() for part in merged.chunk(2, -1))
        del merged
        activation = torch.empty_like(gate)
        example_swiglu_fwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            gate, up, activation, gate.numel(), BLOCK=BLOCK)
        out = bnb.matmul_4bit(activation, down, down_state)
        ctx.save_for_backward(x, packed, down)
        ctx.states = state, down_state
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        import bitsandbytes as bnb
        x, packed, down = ctx.saved_tensors
        state, down_state = ctx.states
        # Explicit half operands also preserve the configured AMP boundaries
        # when autograd executes outside the forward's autocast context.
        da = torch.matmul(grad, _decode(down, down_state))
        merged = bnb.matmul_4bit(x, packed, state)
        gate, up = (part.contiguous() for part in merged.chunk(2, -1))
        del merged
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        example_swiglu_bwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            da.contiguous(), gate, up, dgate, dup, gate.numel(), BLOCK=BLOCK)
        combined = torch.cat((dgate, dup), -1)
        dx = torch.matmul(combined, _decode(packed, state))
        return dx, None, None, None, None


def patch_recomputed_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from bitsandbytes.nn import Linear4bit
    original = Qwen2MLP.forward
    if getattr(original, "_soup_recomputed_mlp", False):
        return

    def forward(self, x):
        layers = (self.gate_proj, self.up_proj, self.down_proj)
        if (not _HALF_MODEL.get() or not self.training or x.dtype != torch.float16 or not x.is_cuda
                or self.config.hidden_act != "silu"
                or not all(type(p) is Linear4bit and p.compute_dtype == torch.float16
                    and p.bias is None and getattr(p.weight, "quant_state", None) is not None
                    and not any(w.requires_grad for w in p.parameters()) for p in layers)):
            return original(self, x)
        # Respect the earlier pointwise padding compaction wrapper. This new
        # wrapper sees full inputs, and performs the same gather/scatter once.
        from .padded_mlp import _TOKENS
        tokens = _TOKENS.get()
        indices = tokens[0] if tokens is not None and x.ndim == 3 and x.shape[:2] == tokens[1] else None
        flat = x.reshape(-1, x.shape[-1])
        selected = flat.index_select(0, indices) if indices is not None else flat
        if selected.shape[0] == 0:
            return x * 0
        cached = packed_gate_up(self)
        if cached is None:
            return original(self, x)
        out = _RecomputedMLP.apply(selected, cached[1], cached[2], self.down_proj.weight,
                                   self.down_proj.weight.quant_state)
        if indices is not None:
            out = torch.zeros_like(flat).index_copy_(0, indices, out)
        return out.view_as(x)

    forward._soup_recomputed_mlp = True
    Qwen2MLP.forward = forward
