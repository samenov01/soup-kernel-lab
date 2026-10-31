"""Compact pointwise attention projections while preserving dropout RNG shapes."""

from contextvars import ContextVar
import torch
from .padded_mlp import _TOKENS

_PROJECTIONS = ContextVar("soup_attention_projection_tokens", default=None)


def patch_padded_attention():
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    original_attention = Qwen2Attention.forward
    if getattr(original_attention, "_soup_projection_padding", False):
        return
    original_linear = torch.nn.Linear.forward
    original_quantized = Linear4bit.forward

    def compact(original, module, x, *args, **kwargs):
        tokens = _PROJECTIONS.get()
        supported = (
            tokens is not None and x.is_cuda and x.ndim == 3
            and x.shape[:2] == tokens[1] and x.dtype in (torch.float32, torch.float16)
        )
        if not supported:
            return original(module, x, *args, **kwargs)
        indices, _ = tokens
        flat = x.reshape(-1, x.shape[-1])
        result = original(module, flat.index_select(0, indices), *args, **kwargs)
        full = result.new_zeros((flat.shape[0], result.shape[-1]))
        return full.index_copy_(0, indices, result).view(*x.shape[:-1], result.shape[-1])

    def linear(self, x):
        # LoRA dropout runs before this call, with its original full shape.
        return compact(original_linear, self, x)

    def quantized(self, x):
        if self.weight.requires_grad:
            return original_quantized(self, x)
        return compact(original_quantized, self, x)

    def attention(self, hidden_states, *args, **kwargs):
        tokens = _TOKENS.get()
        if kwargs.get("past_key_values") is not None or kwargs.get("output_attentions", False):
            tokens = None
        token = _PROJECTIONS.set(tokens)
        try:
            return original_attention(self, hidden_states, *args, **kwargs)
        finally:
            _PROJECTIONS.reset(token)

    attention._soup_projection_padding = True
    torch.nn.Linear.forward = linear
    Linear4bit.forward = quantized
    Qwen2Attention.forward = attention
