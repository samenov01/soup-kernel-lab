"""Native PyTorch memory-efficient attention with explicit ragged autograd.

Each original batch row remains a separate causal sequence. Only small K/V
heads are saved; their deterministic repetition is reconstructed in backward.
No external attention implementation or generated attention kernel is used.
"""

import torch
from torch.autograd.function import once_differentiable
from .attention_io import pack_attention
from .projection_io import _scatter


def _native_forward(q, k, v, cumulative, maximum, scale):
    return torch.ops.aten._efficient_attention_forward(
        q, k, v, None, cumulative, cumulative, maximum, maximum,
        0., 1, True, scale=scale)


def _repeat(x, groups):
    return x.repeat_interleave(groups, dim=2) if groups > 1 else x


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, inverse, cumulative, maximum, scale):
        batch, heads, length, dim = q.shape
        kvheads = k.shape[1]
        qp, kp, vp = (pack_attention(x, indices) for x in (q, k, v))
        groups = heads // kvheads
        out, lse, seed, offset, mq, mk = _native_forward(qp, _repeat(kp, groups), _repeat(vp, groups),
                                                       cumulative, maximum, scale)
        ctx.save_for_backward(qp, kp, vp, out, lse, seed, offset, indices, inverse, cumulative)
        ctx.shape = q.shape
        ctx.dtypes = (q.dtype, k.dtype, v.dtype)
        ctx.maximum, ctx.scale = (mq, mk), scale
        return _scatter(out.view(-1, heads * dim), inverse, (batch, length, heads, dim), out.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset, indices, inverse, cumulative = ctx.saved_tensors
        batch, heads, length, dim = ctx.shape
        groups = heads // k.shape[2]
        dy = pack_attention(grad.transpose(1, 2), indices)
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            dy, q, _repeat(k, groups), _repeat(v, groups), None, out,
            cumulative, cumulative, *ctx.maximum, lse, 0., seed, offset, 1, False,
            scale=ctx.scale)
        if groups > 1:
            dk = dk.view(1, -1, k.shape[2], groups, dim).sum(3, dtype=ctx.dtypes[1])
            dv = dv.view(1, -1, v.shape[2], groups, dim).sum(3, dtype=ctx.dtypes[2])
        def scatter(x, dtype):
            shape = (batch, length, x.shape[2], dim)
            return _scatter(x.reshape(-1, x.shape[2] * dim), inverse, shape, dtype).transpose(1, 2)
        return (scatter(dq, ctx.dtypes[0]), scatter(dk, ctx.dtypes[1]), scatter(dv, ctx.dtypes[2]),
                None, None, None, None, None)


def varlen_attention(q, k, v, frame, ranges, scale):
    cached = getattr(frame, "_soup_varlen_layout", None)
    if cached is None:
        lengths = [end - start for start, end in ranges]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        indices = frame.attention_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        inverse = torch.full((frame.attention_mask.numel(),), -1, device=q.device, dtype=torch.long)
        inverse.index_copy_(0, indices, torch.arange(indices.numel(), device=q.device))
        cumulative = torch.tensor(offsets, device=q.device, dtype=torch.int32)
        cached = (indices, inverse, cumulative, max(lengths))
        frame._soup_varlen_layout = cached
    return _VarlenAttention.apply(q, k, v, *cached, scale)
