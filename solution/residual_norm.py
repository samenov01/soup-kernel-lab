"""Fuse half residual addition with frozen FP32 RMSNorm arithmetic."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def residual_norm_forward_kernel(A, B, W, R, Y, INV, WIDTH: tl.constexpr,
                                 EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    a = tl.load(A + row * WIDTH + col, mask, other=0).to(tl.float32)
    b = tl.load(B + row * WIDTH + col, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.float16).to(tl.float32)
    w = tl.load(W + col, mask, other=0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    tl.store(R + row * WIDTH + col, x, mask)
    tl.store(Y + row * WIDTH + col, (x * inv) * w, mask)
    tl.store(INV + row, inv)


@triton.jit
def residual_norm_backward_kernel(DR, DY, R, W, INV, DX, WIDTH: tl.constexpr,
                                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(R + row * WIDTH + col, mask, other=0).to(tl.float32)
    w = tl.load(W + col, mask, other=0)
    dy = tl.load(DY + row * WIDTH + col, mask, other=0).to(tl.float32)
    dr = tl.load(DR + row * WIDTH + col, mask, other=0).to(tl.float32)
    inv = tl.load(INV + row)
    normalized = x * inv
    weighted = dy * w
    projection = tl.sum(weighted * normalized, 0) / WIDTH
    dn = (inv * (weighted - normalized * projection)).to(tl.float16).to(tl.float32)
    tl.store(DX + row * WIDTH + col, dr + dn, mask)


class ResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, weight, eps):
        a, b = a.contiguous(), b.contiguous()
        residual, out = torch.empty_like(a), torch.empty_like(a)
        width = a.shape[-1]
        inv = torch.empty(a.numel() // width, device=a.device, dtype=torch.float32)
        residual_norm_forward_kernel[(inv.numel(),)](
            a, b, weight, residual, out, inv, width, eps, triton.next_power_of_2(width), num_warps=4)
        ctx.save_for_backward(residual, weight, inv)
        return residual, out

    @staticmethod
    @once_differentiable
    def backward(ctx, dr, dy):
        residual, weight, inv = ctx.saved_tensors
        width = residual.shape[-1]
        dx = torch.empty_like(residual)
        residual_norm_backward_kernel[(inv.numel(),)](
            dr.contiguous(), dy.contiguous(), residual, weight, inv, dx, width,
            triton.next_power_of_2(width), num_warps=4)
        return dx, dx, None, None


def patch_residual_norm():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from .half_activations import _HALF_MODEL, _HALF_LAYER
    original = Qwen2DecoderLayer.forward
    if getattr(original, "_soup_residual_norm", False):
        return

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, **kwargs):
        if (not _HALF_MODEL.get() or not self.training or past_key_values is not None
                or use_cache or not hidden_states.is_cuda):
            return original(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        token = _HALF_LAYER.set(True)
        try:
            hidden_states = hidden_states.half()
            attention, _ = self.self_attn(hidden_states=self.input_layernorm(hidden_states),
                attention_mask=attention_mask, position_ids=position_ids, past_key_values=None,
                use_cache=False, position_embeddings=position_embeddings, **kwargs)
            residual, normalized = ResidualNorm.apply(hidden_states, attention,
                self.post_attention_layernorm.weight, self.post_attention_layernorm.variance_epsilon)
            return residual + self.mlp(normalized)
        finally:
            _HALF_LAYER.reset(token)

    forward._soup_residual_norm = True
    Qwen2DecoderLayer.forward = forward
