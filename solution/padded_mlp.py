"""Skip padding positions in frozen pointwise MLPs, preserving attention batches.

No token is moved across batches for attention; Q/V and dropout retain original
shapes/RNG. Only deterministic frozen per-token feed-forward work is compacted.
"""

from contextvars import ContextVar
import torch
import torch.nn.functional as F

from .chunked_head import _ACTIVE

_TOKENS = ContextVar("soup_valid_mlp_tokens", default=None)


def patch_padded_mlp():
    from bitsandbytes.nn import Linear4bit
    from transformers.models.qwen2 import modeling_qwen2

    original_body = modeling_qwen2.Qwen2Model.forward
    original_mlp = modeling_qwen2.Qwen2MLP.forward
    if getattr(original_mlp, "_soup_padding_patch", False):
        return

    def body(self, input_ids=None, attention_mask=None, position_ids=None,
             past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        supported = (
            self.training and use_cache is False and past_key_values is None
            and not self.gradient_checkpointing
            and not kwargs.get("output_hidden_states", False) and not kwargs.get("output_attentions", False)
            and isinstance(attention_mask, torch.Tensor) and attention_mask.is_cuda
            and attention_mask.ndim == 2
        )
        active = None
        if supported:
            frame = _ACTIVE.get()
            if hasattr(frame, "labels") and frame.labels.shape == attention_mask.shape:
                shifted = F.pad(frame.labels, (0, 1), value=-100)[:, 1:]
                # An unusual supervised query in left padding must retain all
                # original pad states because SDPA may unmask unattended rows.
                supported = not bool(((attention_mask == 0) & (shifted != -100)).any())
        if supported:
            indices = attention_mask.reshape(-1).ne(0).nonzero(as_tuple=False).flatten()
            if indices.numel() < attention_mask.numel():
                active = (indices, attention_mask.shape)
        token = _TOKENS.set(active)
        try:
            return original_body(self, input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        finally:
            _TOKENS.reset(token)

    def mlp(self, x):
        tokens = _TOKENS.get()
        supported = (
            tokens is not None and x.ndim == 3 and x.shape[:2] == tokens[1]
            and x.dtype == torch.float32
            and all(type(layer) is Linear4bit and layer.compute_dtype == torch.float16
                    and layer.compute_type_is_set and layer.bias is None
                    and not any(p.requires_grad for p in layer.parameters())
                    for layer in (self.gate_proj, self.up_proj, self.down_proj))
        )
        if not supported:
            return original_mlp(self, x)
        indices, _ = tokens
        flat = x.reshape(-1, x.shape[-1])
        # Empty examples still have a differentiable zero contribution.
        if indices.numel() == 0:
            return x * 0
        compact = original_mlp(self, flat.index_select(0, indices))
        out = torch.zeros_like(flat)
        out.index_copy_(0, indices, compact)
        return out.view_as(x)

    mlp._soup_padding_patch = True
    modeling_qwen2.Qwen2Model.forward = body
    modeling_qwen2.Qwen2MLP.forward = mlp
