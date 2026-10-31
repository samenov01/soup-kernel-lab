"""Independent masked causal GQA kernels, with bounded per-tile storage.

Mathematics: online softmax (FlashAttention), as described by the papers linked
from https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html.
No tutorial/kernel implementation is imported or copied. Tensor shapes, grouped
gradient rounding and padding semantics here follow the pinned Qwen2 SDPA path.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def causal_gqa_forward_kernel(Q, K, V, MASK, O, STATS, END,
        H: tl.constexpr, KV: tl.constexpr, L, D: tl.constexpr,
        SCALE: tl.constexpr, BLOCK: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    channels = tl.arange(0, D)
    active = (rows < L) & tl.load(MASK + batch * L + rows, rows < L, other=0).to(tl.int1)
    qbase = (batch * H + head) * L * D
    kvbase = (batch * KV + head // (H // KV)) * L * D
    q = tl.load(Q + qbase + rows[:, None] * D + channels[None, :], active[:, None], other=0)
    maximum = tl.full((BLOCK,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK,), tl.float32)
    numerator = tl.zeros((BLOCK, D), tl.float32)
    if tl.sum(active.to(tl.int32)) > 0:
        limit = tl.minimum((tile + 1) * BLOCK, tl.load(END + batch))
        for start in range(0, limit, BLOCK):
            keys = start + cols
            valid = (keys < L) & tl.load(MASK + batch * L + keys, keys < L, other=0).to(tl.int1)
            k = tl.load(K + kvbase + channels[:, None] + keys[None, :] * D, valid[None, :], other=0)
            value = tl.load(V + kvbase + keys[:, None] * D + channels[None, :], valid[:, None], other=0)
            scores = tl.dot(q, k) * SCALE
            permitted = active[:, None] & valid[None, :] & (rows[:, None] >= keys[None, :])
            scores = tl.where(permitted, scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(next_max == -float("inf"), 0., next_max)
            correction = tl.exp(maximum - safe_max)
            probabilities = tl.exp(scores - safe_max[:, None])
            numerator = numerator * correction[:, None] + tl.dot(probabilities.to(tl.float16), value)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = next_max
    normalizer = tl.where(denominator > 0, denominator, 1.)
    result = numerator / normalizer[:, None]
    tl.store(O + qbase + rows[:, None] * D + channels[None, :], result, rows[:, None] < L)
    statbase = (batch * H + head) * L * 2 + rows * 2
    tl.store(STATS + statbase, tl.where(active, maximum, 0.), rows < L)
    tl.store(STATS + statbase + 1, normalizer, rows < L)


@triton.jit
def causal_gqa_delta_kernel(O, DO, DELTA, ROWS, D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    o = tl.load(O + row * D + d).to(tl.float32)
    grad = tl.load(DO + row * D + d).to(tl.float32)
    tl.store(DELTA + row, tl.sum(o * grad))


@triton.jit
def causal_gqa_query_gradient_kernel(Q, K, V, DO, MASK, STATS, DELTA, DQ, END,
        H: tl.constexpr, KV: tl.constexpr, L, D: tl.constexpr,
        SCALE: tl.constexpr, BLOCK: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    columns = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    live = (rows < L) & tl.load(MASK + batch * L + rows, rows < L, other=0).to(tl.int1)
    base = (batch * H + head) * L * D
    kb = (batch * KV + head // (H // KV)) * L * D
    q = tl.load(Q + base + rows[:, None] * D + d[None, :], live[:, None], other=0)
    go = tl.load(DO + base + rows[:, None] * D + d[None, :], live[:, None], other=0)
    sb = (batch * H + head) * L * 2 + rows * 2
    m = tl.load(STATS + sb, rows < L, other=0)
    norm = tl.load(STATS + sb + 1, rows < L, other=1)
    delta = tl.load(DELTA + (batch * H + head) * L + rows, rows < L, other=0)
    gradient = tl.zeros((BLOCK, D), tl.float32)
    if tl.sum(live.to(tl.int32)) > 0:
        limit = tl.minimum((tile + 1) * BLOCK, tl.load(END + batch))
        for start in range(0, limit, BLOCK):
            keys = start + columns
            valid = (keys < L) & tl.load(MASK + batch * L + keys, keys < L, other=0).to(tl.int1)
            k = tl.load(K + kb + d[:, None] + keys[None, :] * D, valid[None, :], other=0)
            v = tl.load(V + kb + d[:, None] + keys[None, :] * D, valid[None, :], other=0)
            scores = tl.dot(q, k) * SCALE
            permitted = live[:, None] & valid[None, :] & (rows[:, None] >= keys[None, :])
            probability = tl.where(permitted, tl.exp(scores - m[:, None]) / norm[:, None], 0.)
            dp = tl.dot(go, v)
            ds = probability * (dp - delta[:, None]) * SCALE
            gradient += tl.dot(ds.to(tl.float16), tl.trans(k))
    tl.store(DQ + base + rows[:, None] * D + d[None, :], gradient, rows[:, None] < L)


@triton.jit
def causal_gqa_key_value_gradient_kernel(Q, K, V, DO, MASK, STATS, DELTA, DK, DV, END,
        H: tl.constexpr, KV: tl.constexpr, L, D: tl.constexpr,
        SCALE: tl.constexpr, BLOCK: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    keys = tile * BLOCK + tl.arange(0, BLOCK)
    rr = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = (keys < L) & tl.load(MASK + batch * L + keys, keys < L, other=0).to(tl.int1)
    base = (batch * H + head) * L * D
    kb = (batch * KV + head // (H // KV)) * L * D
    k = tl.load(K + kb + keys[:, None] * D + d[None, :], valid[:, None], other=0)
    v = tl.load(V + kb + keys[:, None] * D + d[None, :], valid[:, None], other=0)
    gk = tl.zeros((BLOCK, D), tl.float32)
    gv = tl.zeros((BLOCK, D), tl.float32)
    if tl.sum(valid.to(tl.int32)) > 0:
        for start in range(tile * BLOCK, tl.load(END + batch), BLOCK):
            rows = start + rr
            live = (rows < L) & tl.load(MASK + batch * L + rows, rows < L, other=0).to(tl.int1)
            q = tl.load(Q + base + d[:, None] + rows[None, :] * D, live[None, :], other=0)
            go = tl.load(DO + base + rows[:, None] * D + d[None, :], live[:, None], other=0)
            sb = (batch * H + head) * L * 2 + rows * 2
            m = tl.load(STATS + sb, rows < L, other=0)
            norm = tl.load(STATS + sb + 1, rows < L, other=1)
            delta = tl.load(DELTA + (batch * H + head) * L + rows, rows < L, other=0)
            score_t = tl.dot(k, q) * SCALE
            allowed = valid[:, None] & live[None, :] & (keys[:, None] <= rows[None, :])
            pt = tl.where(allowed, tl.exp(score_t - m[None, :]) / norm[None, :], 0.)
            gv += tl.dot(pt.to(tl.float16), go)
            dp_t = tl.dot(v, tl.trans(go))
            ds_t = pt * (dp_t - delta[None, :]) * SCALE
            gk += tl.dot(ds_t.to(tl.float16), tl.trans(q))
    tl.store(DK + base + keys[:, None] * D + d[None, :], gk, keys[:, None] < L)
    tl.store(DV + base + keys[:, None] * D + d[None, :], gv, keys[:, None] < L)


class _CausalGQA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale):
        q, k, v = (x.to(torch.float16).contiguous() for x in (query, key, value))
        b, h, length, dim = q.shape
        kv = k.shape[1]
        mask = mask.contiguous()
        end = (mask * torch.arange(1, length + 1, device=mask.device)).amax(-1).int()
        out = torch.empty_like(q)
        stats = torch.empty((b, h, length, 2), device=q.device, dtype=torch.float32)
        causal_gqa_forward_kernel[(triton.cdiv(length, 32), h, b)](
            q, k, v, mask, out, stats, end, h, kv, length, dim, scale, 32, num_warps=4, num_stages=1)
        ctx.save_for_backward(q, k, v, mask, out, stats, end)
        ctx.scale = scale
        ctx.dtypes = (query.dtype, key.dtype, value.dtype)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, mask, out, stats, end = ctx.saved_tensors
        b, h, length, dim = q.shape
        kv = k.shape[1]
        grad = grad.contiguous()
        delta = torch.empty((b, h, length), device=q.device, dtype=torch.float32)
        dq, dk_heads, dv_heads = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
        causal_gqa_delta_kernel[(b * h * length,)](out, grad, delta, b * h * length, dim, num_warps=4)
        grid = (triton.cdiv(length, 32), h, b)
        causal_gqa_query_gradient_kernel[grid](q, k, v, grad, mask, stats, delta, dq, end,
            h, kv, length, dim, ctx.scale, 32, num_warps=4, num_stages=1)
        causal_gqa_key_value_gradient_kernel[grid](q, k, v, grad, mask, stats, delta, dk_heads, dv_heads, end,
            h, kv, length, dim, ctx.scale, 32, num_warps=4, num_stages=1)
        # The original SDPA path repeats FP32 K/V first, then casts each head
        # to FP16. Its backward rounds per-head gradients before FP32 summation.
        dk = dk_heads.to(ctx.dtypes[1]).view(b, kv, h // kv, length, dim).sum(2)
        dv = dv_heads.to(ctx.dtypes[2]).view(b, kv, h // kv, length, dim).sum(2)
        return dq.to(ctx.dtypes[0]), dk, dv, None, None


def patch_causal_gqa():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    from .chunked_head import _ACTIVE
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_soup_causal_gqa", False):
        return

    def attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        frame = _ACTIVE.get()
        mask = getattr(frame, "attention_mask", None)
        supported = (
            isinstance(module, Qwen2Attention) and module.training and dropout == 0.
            and query.is_cuda and query.dtype == key.dtype == value.dtype == torch.float32
            and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
            and query.ndim == key.ndim == value.ndim == 4 and query.shape[-1] in (64, 128)
            and query.shape[0] == key.shape[0] and query.shape[2:] == key.shape[2:] and key.shape == value.shape
            and query.shape[1] % key.shape[1] == 0 and module.sliding_window is None
            and isinstance(mask, torch.Tensor) and mask.shape == (query.shape[0], query.shape[2])
            and not kwargs.get("output_attentions", False) and kwargs.get("position_bias") is None
        )
        # The model context only permits this path for ordinary full-sequence
        # SFT. Supervised padding-boundary queries must retain native semantics.
        from .padded_mlp import _TOKENS
        if not supported or _TOKENS.get() is None:
            return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        result = _CausalGQA.apply(query, key, value, mask, scaling or query.shape[-1] ** -0.5)
        return result.transpose(1, 2), None

    attention._soup_causal_gqa = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
