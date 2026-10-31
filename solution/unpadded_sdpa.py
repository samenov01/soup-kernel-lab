"""Run native causal SDPA on valid contiguous rows, without changing batches."""

import torch
import torch.nn.functional as F
from .chunked_head import _ACTIVE
from .padded_mlp import _TOKENS


def patch_unpadded_sdpa():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_soup_unpadded_sdpa", False):
        return

    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        frame = _ACTIVE.get()
        mask = getattr(frame, "attention_mask", None)
        supported = (
            type(module) is Qwen2Attention and module.training and module.is_causal
            and dropout == 0. and module.sliding_window is None and _TOKENS.get() is not None
            and query.is_cuda and query.ndim == key.ndim == value.ndim == 4
            and query.shape[0] == key.shape[0] and query.shape[2:] == key.shape[2:]
            and key.shape == value.shape and query.shape[1] % key.shape[1] == 0
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
            and isinstance(mask, torch.Tensor) and mask.shape == (query.shape[0], query.shape[2])
            and kwargs.get("is_causal") in (None, True)
            and kwargs.get("position_bias") is None and not kwargs.get("output_attentions", False)
        )
        if not supported:
            return original(module, query, key, value, attention_mask,
                            dropout=dropout, scaling=scaling, **kwargs)
        signature = (mask.data_ptr(), mask._version)
        cached = getattr(frame, "_soup_sdpa_ranges", None)
        if cached is None or cached[0] != signature:
            ranges = []
            for row in mask.detach().cpu().tolist():
                live = [i for i, v in enumerate(row) if v]
                if not live or len(live) != live[-1] - live[0] + 1:
                    ranges = None
                    break
                ranges.append((live[0], live[-1] + 1))
            cached = (signature, ranges)
            frame._soup_sdpa_ranges = cached
        if cached[1] is None:
            return original(module, query, key, value, attention_mask,
                            dropout=dropout, scaling=scaling, **kwargs)
        batch, heads, length, dim = query.shape
        groups = heads // key.shape[1]
        output = torch.zeros((batch, length, heads, dim), device=query.device, dtype=torch.float16)
        for i, (start, end) in enumerate(cached[1]):
            q = query[i:i + 1, :, start:end]
            k = key[i:i + 1, :, start:end]
            v = value[i:i + 1, :, start:end]
            if groups > 1:
                # Explicit repetition keeps T4's memory-efficient SDPA path;
                # enable_gqa can otherwise select the math backend on sm75.
                k = k[:, :, None].expand(1, key.shape[1], groups, end - start, dim).reshape(1, heads, end - start, dim)
                v = v[:, :, None].expand(1, value.shape[1], groups, end - start, dim).reshape(1, heads, end - start, dim)
            part = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True,
                                                  scale=dim ** -0.5 if scaling is None else scaling)
            output[i:i + 1, start:end] = part.transpose(1, 2)
        return output, None

    attention._soup_unpadded_sdpa = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
