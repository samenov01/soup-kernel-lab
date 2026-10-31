"""Native PyTorch memory-efficient attention with explicit ragged autograd.

Each original batch row remains a separate causal sequence. Only small K/V
heads are saved; their deterministic repetition is reconstructed in backward.
No external attention implementation or generated attention kernel is used.
"""

import torch
from torch.autograd.function import once_differentiable


def _native_forward(q, k, v, cumulative, maximum, scale):
    return torch.ops.aten._efficient_attention_forward(
        q, k, v, None, cumulative, cumulative, maximum, maximum,
        0., 1, True, scale=scale)


def _repeat(x, groups):
    return x.repeat_interleave(groups, dim=2) if groups > 1 else x


class _VarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, cumulative, maximum, scale):
        batch, heads, length, dim = q.shape
        kvheads = k.shape[1]
        def gather(x):
            return x.transpose(1, 2).reshape(batch * length, -1).index_select(0, indices).half().view(1, -1, x.shape[1], dim)
        qp, kp, vp = gather(q), gather(k), gather(v)
        groups = heads // kvheads
        out, lse, seed, offset, mq, mk = _native_forward(qp, _repeat(kp, groups), _repeat(vp, groups),
                                                       cumulative, maximum, scale)
        ctx.save_for_backward(qp, kp, vp, out, lse, seed, offset, indices, cumulative)
        ctx.shape = q.shape
        ctx.dtypes = (q.dtype, k.dtype, v.dtype)
        ctx.maximum, ctx.scale = (mq, mk), scale
        full = out.new_zeros((batch * length, heads, dim))
        full.index_copy_(0, indices, out[0])
        return full.view(batch, length, heads, dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset, indices, cumulative = ctx.saved_tensors
        batch, heads, length, dim = ctx.shape
        groups = heads // k.shape[2]
        dy = grad.reshape(batch * length, heads * dim).index_select(0, indices).half().view_as(out)
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            dy, q, _repeat(k, groups), _repeat(v, groups), None, out,
            cumulative, cumulative, *ctx.maximum, lse, 0., seed, offset, 1, False,
            scale=ctx.scale)
        if groups > 1:
            dk = dk.view(1, -1, k.shape[2], groups, dim).sum(3, dtype=ctx.dtypes[1])
            dv = dv.view(1, -1, v.shape[2], groups, dim).sum(3, dtype=ctx.dtypes[2])
        def scatter(x, dtype):
            full = torch.zeros((batch * length, x.shape[2], dim), device=x.device, dtype=dtype)
            full.index_copy_(0, indices, x[0].to(dtype))
            return full.view(batch, length, x.shape[2], dim).transpose(1, 2)
        return (scatter(dq, ctx.dtypes[0]), scatter(dk, ctx.dtypes[1]), scatter(dv, ctx.dtypes[2]),
                None, None, None, None)


def varlen_attention(q, k, v, frame, ranges, scale):
    cached = getattr(frame, "_soup_varlen_layout", None)
    if cached is None:
        lengths = [end - start for start, end in ranges]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        indices = frame.attention_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        cumulative = torch.tensor(offsets, device=q.device, dtype=torch.int32)
        cached = (indices, cumulative, max(lengths))
        frame._soup_varlen_layout = cached
    return _VarlenAttention.apply(q, k, v, *cached, scale)
