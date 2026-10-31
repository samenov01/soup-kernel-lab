"""Experimental fp16 activation storage inside ordinary fp16 SFT decoder calls.

Model parameters, trainer flags and final head inputs retain their dtypes.
Norm arithmetic stays fp32. Internal residual/activation rounding changes and
must be validated against the task's actual curve and quality checks.
"""

from contextvars import ContextVar
import torch
import triton
from torch.autograd.function import once_differentiable
from .chunked_head import _ACTIVE
from .rmsnorm import rmsnorm_forward_kernel, rmsnorm_backward_kernel

_HALF_MODEL = ContextVar("soup_fp16_activation_model", default=False)
_HALF_LAYER = ContextVar("soup_fp16_activation_layer", default=False)


class _ActivationNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, output_dtype):
        x = x.contiguous()
        width = x.shape[-1]
        out = torch.empty(x.shape, device=x.device, dtype=output_dtype)
        inv = torch.empty(x.numel() // width, device=x.device, dtype=torch.float32)
        rmsnorm_forward_kernel[(x.numel() // width,)](
            x, weight, out, inv, width, eps, triton.next_power_of_2(width), num_warps=4)
        ctx.save_for_backward(x, weight, inv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        width = x.shape[-1]
        dx = torch.empty_like(x)
        rmsnorm_backward_kernel[(x.numel() // width,)](
            grad.contiguous(), x, weight, inv, dx, width, triton.next_power_of_2(width), num_warps=4)
        return dx, None, None, None


def patch_half_activations():
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    original_body = qwen.Qwen2Model.forward
    original_layer = qwen.Qwen2DecoderLayer.forward
    original_norm = qwen.Qwen2RMSNorm.forward
    if getattr(original_body, "_soup_half_activations", False):
        return

    def body(self, *args, **kwargs):
        supported = (
            bool(_ACTIVE.get()) and self.training and not self.gradient_checkpointing
            and kwargs.get("use_cache") is False and kwargs.get("past_key_values") is None
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
            and self.config._attn_implementation == "sdpa"
            and not kwargs.get("output_attentions", False) and not kwargs.get("output_hidden_states", False)
            and not getattr(self.config, "output_attentions", False)
            and not getattr(self.config, "output_hidden_states", False)
            and all(n.weight.is_cuda and n.weight.dtype == torch.float32 and not n.weight.requires_grad
                    for n in [self.norm, *[norm for layer in self.layers
                        for norm in (layer.input_layernorm, layer.post_attention_layernorm)]])
        )
        token = _HALF_MODEL.set(supported)
        try:
            return original_body(self, *args, **kwargs)
        finally:
            _HALF_MODEL.reset(token)

    def layer(self, hidden_states, *args, **kwargs):
        if not _HALF_MODEL.get():
            return original_layer(self, hidden_states, *args, **kwargs)
        token = _HALF_LAYER.set(True)
        try:
            return original_layer(self, hidden_states.to(torch.float16), *args, **kwargs)
        finally:
            _HALF_LAYER.reset(token)

    def norm(self, x):
        if (_HALF_MODEL.get() and x.is_cuda and x.dtype == torch.float16 and x.numel() > 0
                and self.weight.dtype == torch.float32 and not self.weight.requires_grad):
            dtype = torch.float16 if _HALF_LAYER.get() else torch.float32
            return _ActivationNorm.apply(x, self.weight, self.variance_epsilon, dtype)
        return original_norm(self, x)

    body._soup_half_activations = True
    qwen.Qwen2Model.forward = body
    qwen.Qwen2DecoderLayer.forward = layer
    qwen.Qwen2RMSNorm.forward = norm
