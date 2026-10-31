"""Compact pointwise attention projections while preserving dropout RNG shapes."""

from contextvars import ContextVar
import torch
from .padded_mlp import _TOKENS
from .chunked_head import _ACTIVE
from .projection_io import PackedInput, FullOutput

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
        indices = tokens[0]
        fast_io = (
            indices.numel() > 0 and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
            and (type(module) is torch.nn.Linear
                 or type(module) is Linear4bit and module.compute_dtype == torch.float16
                    and module.compute_type_is_set)
        )
        if fast_io:
            result = original(module, PackedInput.apply(x, indices, tokens[2]), *args, **kwargs)
            dtype = x.dtype if type(module) is Linear4bit else result.dtype
            return FullOutput.apply(result, indices, tokens[2], (*x.shape[:-1], result.shape[-1]), dtype)
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
        if tokens is not None:
            indices, shape = tokens
            frame = _ACTIVE.get()
            cached = getattr(frame, "_soup_projection_inverse", None)
            if cached is None or cached[0] is not indices:
                inverse = torch.full((shape[0] * shape[1],), -1, device=indices.device, dtype=torch.long)
                inverse.index_copy_(0, indices, torch.arange(indices.numel(), device=indices.device))
                cached = (indices, inverse)
                if hasattr(frame, "attention_mask"):
                    frame._soup_projection_inverse = cached
            tokens = (indices, shape, cached[1])
        token = _PROJECTIONS.set(tokens)
        try:
            return original_attention(self, hidden_states, *args, **kwargs)
        finally:
            _PROJECTIONS.reset(token)

    attention._soup_projection_padding = True
    torch.nn.Linear.forward = linear
    Linear4bit.forward = quantized
    Qwen2Attention.forward = attention
