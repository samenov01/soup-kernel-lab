"""Share one frozen base Q/K/V matrix product; leave PEFT's LoRA path intact."""

from contextvars import ContextVar
from copy import copy
import torch
from torch.autograd.function import once_differentiable
from .merged_gate_up import _scales, _signature, share_packed_storage
from .padded_mlp import _TOKENS
from .chunked_head import _ACTIVE
from .qkv_io import _FullQKV

_BASE = ContextVar("soup_precomputed_base_qkv", default=None)


class _QKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state, bias, sizes):
        import bitsandbytes as bnb
        output = bnb.matmul_4bit(x.to(torch.float16), packed, state, bias=bias).to(x.dtype)
        ctx.save_for_backward(packed)
        ctx.state, ctx.sizes = state, sizes
        ctx.dtype = x.dtype
        return output.split(sizes, dim=-1)

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk, dv):
        from bitsandbytes import functional as F
        packed, = ctx.saved_tensors
        state = copy(ctx.state)
        state.dtype = torch.float16
        combined = torch.cat((dq, dk, dv), -1).half()
        return torch.matmul(combined, F.dequantize_4bit(packed, state)).to(ctx.dtype), None, None, None, None


def _base(module):
    return module.get_base_layer() if hasattr(module, "get_base_layer") else module


def patch_merged_qkv():
    from bitsandbytes.nn import Linear4bit
    from bitsandbytes.functional import QuantState
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    original_attention = Qwen2Attention.forward
    if getattr(original_attention, "_soup_merged_qkv", False):
        return
    original_linear = Linear4bit.forward

    def linear(self, x):
        frame = _BASE.get()
        if frame is not None and x is frame[0] and self in frame[1]:
            frame[2].add(self)
            return frame[1][self]
        return original_linear(self, x)

    def attention(self, hidden_states, *args, **kwargs):
        layers = tuple(_base(p) for p in (self.q_proj, self.k_proj, self.v_proj))
        supported = (
            self.training and hidden_states.is_cuda and hidden_states.ndim == 3
            and hidden_states.dtype in (torch.float32, torch.float16) and hidden_states.shape[1] > 1
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
            and kwargs.get("past_key_values") is None
            and all(type(p) is Linear4bit and p.compute_dtype == torch.float16 and p.compute_type_is_set
                    and getattr(p.weight, "quant_state", None) is not None
                    and not any(w.requires_grad for w in p.parameters()) for p in layers)
        )
        if not supported:
            return original_attention(self, hidden_states, *args, **kwargs)
        states = [p.weight.quant_state for p in layers]
        supported = all(s.quant_type == "nf4" and s.blocksize == states[0].blocksize
            and len(s.shape) == 2 and s.shape[1] == hidden_states.shape[-1]
            and s.shape[0] * s.shape[1] % s.blocksize == 0
            and s.shape[0] * s.shape[1] % 2 == 0 for s in states)
        if not supported:
            return original_attention(self, hidden_states, *args, **kwargs)
        signature = tuple((_signature(p.weight), None if p.bias is None else
                           (p.bias.data_ptr(), p.bias._version)) for p in layers)
        cached = getattr(self, "_soup_merged_qkv_cache", None)
        if cached is None or cached[0] != signature:
            packed = torch.cat([p.weight.view(-1, 1) for p in layers])
            sizes = tuple(s.shape[0] for s in states)
            state = QuantState(torch.cat([_scales(s) for s in states]),
                shape=torch.Size((sum(sizes), hidden_states.shape[-1])), code=states[0].code,
                blocksize=states[0].blocksize, quant_type="nf4", dtype=torch.float32)
            bias = None
            if any(p.bias is not None for p in layers):
                bias = torch.cat([p.bias.half() if p.bias is not None else
                    torch.zeros(n, device=hidden_states.device, dtype=torch.float16)
                    for p, n in zip(layers, sizes)])
            share_packed_storage([p.weight for p in layers], packed)
            signature = tuple((_signature(p.weight), None if p.bias is None else
                               (p.bias.data_ptr(), p.bias._version)) for p in layers)
            cached = (signature, packed, state, bias, sizes)
            self._soup_merged_qkv_cache = cached
        tokens = _TOKENS.get()
        x = hidden_states
        if tokens is not None:
            frame = _ACTIVE.get()
            io = getattr(frame, "_soup_projection_inverse", None)
            if io is None or io[0] is not tokens[0]:
                inverse = torch.full((tokens[1].numel(),), -1, device=x.device, dtype=torch.long)
                inverse.index_copy_(0, tokens[0], torch.arange(tokens[0].numel(), device=x.device))
                io = (tokens[0], inverse)
                if hasattr(frame, "attention_mask"):
                    frame._soup_projection_inverse = io
            outputs = _FullQKV.apply(x, *cached[1:], tokens[0], io[1])
        else:
            outputs = _QKV.apply(x, *cached[1:])
        consumed = set()
        frame = _BASE.set((hidden_states, dict(zip(layers, outputs)), consumed))
        try:
            result = original_attention(self, hidden_states, *args, **kwargs)
            if consumed != set(layers):
                raise RuntimeError("PEFT base projection contract changed: precomputed QKV not fully consumed")
            return result
        finally:
            _BASE.reset(frame)

    attention._soup_merged_qkv = True
    Linear4bit.forward = linear
    Qwen2Attention.forward = attention
