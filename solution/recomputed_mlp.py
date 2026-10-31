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
from .projection_io import _gather, _scatter

# Retain the two wide half activations on alternating layers. The other half
# recomputes them; both paths use identical arithmetic and original batches.
RECOMPUTE_EVERY_LAYER = False


def _decode(packed, state):
    from bitsandbytes.functional import dequantize_4bit
    from .nf4_scales import decoded_scales
    state = copy(decoded_scales(state))
    state.dtype = torch.float16
    return dequantize_4bit(packed, state)


class _RecomputedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, down, down_state, indices=None, inverse=None):
        import bitsandbytes as bnb
        ctx.nargs = len(ctx.needs_input_grad)
        ctx.shape = x.shape
        x = _gather(x, indices, torch.float16) if indices is not None else x.reshape(-1, x.shape[-1])
        merged = bnb.matmul_4bit(x, packed, state)
        gate, up = (part.contiguous() for part in merged.chunk(2, -1))
        del merged
        activation = torch.empty_like(gate)
        example_swiglu_fwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            gate, up, activation, gate.numel(), BLOCK=BLOCK)
        out = bnb.matmul_4bit(activation, down, down_state)
        ctx.save_for_backward(x, packed, down, indices, inverse)
        ctx.states = state, down_state
        return (_scatter(out, inverse, ctx.shape, x.dtype) if indices is not None
                else out.view(ctx.shape))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        import bitsandbytes as bnb
        x, packed, down, indices, inverse = ctx.saved_tensors
        state, down_state = ctx.states
        grad = (_gather(grad, indices, torch.float16) if indices is not None
                else grad.reshape(-1, grad.shape[-1]))
        # Explicit half operands also preserve the configured AMP boundaries
        # when autograd executes outside the forward's autocast context.
        da = torch.matmul(grad, _decode(down, down_state))
        merged = bnb.matmul_4bit(x, packed, state)
        gate, up = (part.contiguous() for part in merged.chunk(2, -1))
        del merged
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        example_swiglu_bwd_kernel[(triton.cdiv(gate.numel(), BLOCK),)](
            da.contiguous(), gate, up, dgate, dup, gate.numel(), BLOCK=BLOCK)
        del da, gate, up
        combined = torch.cat((dgate, dup), -1)
        del dgate, dup
        dx = torch.matmul(combined, _decode(packed, state))
        dx = (_scatter(dx, inverse, ctx.shape, x.dtype) if indices is not None else dx.view(ctx.shape))
        return (dx, None, None, None, None, None, None)[:ctx.nargs]


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
        if not RECOMPUTE_EVERY_LAYER and getattr(self, "_soup_layer_index", 0) % 2:
            return original(self, x)
        # Respect the earlier pointwise padding compaction wrapper. This new
        # wrapper sees full inputs, and performs the same gather/scatter once.
        from .padded_mlp import _TOKENS
        from .chunked_head import _ACTIVE
        tokens = _TOKENS.get()
        indices = tokens[0] if tokens is not None and x.ndim == 3 and x.shape[:2] == tokens[1] else None
        if x.numel() == 0 or indices is not None and indices.numel() == 0:
            return x * 0
        cached = packed_gate_up(self)
        if cached is None:
            return original(self, x)
        inverse = None
        if indices is not None:
            frame = _ACTIVE.get()
            io = getattr(frame, "_soup_projection_inverse", None)
            if io is None or io[0] is not indices:
                inverse = torch.full((x.shape[0] * x.shape[1],), -1, device=x.device, dtype=torch.long)
                inverse.index_copy_(0, indices, torch.arange(indices.numel(), device=x.device))
            else:
                inverse = io[1]
        return _RecomputedMLP.apply(x, cached[1], cached[2], self.down_proj.weight,
                                   self.down_proj.weight.quant_state, indices, inverse)

    forward._soup_recomputed_mlp = True
    Qwen2MLP.forward = forward
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    original_init = Qwen2DecoderLayer.__init__

    def initialize(self, config, layer_idx):
        original_init(self, config, layer_idx)
        self.mlp._soup_layer_index = layer_idx

    Qwen2DecoderLayer.__init__ = initialize
