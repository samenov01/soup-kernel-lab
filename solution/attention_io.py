"""Gather BHLD attention operands directly into native variable-length layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def attention_pack_kernel(X, INDICES, OUT, ROWS, LENGTH, SB, SH, SL, SD,
                          HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // (HEADS * DIM)
    head = (offset // DIM) % HEADS
    channel = offset % DIM
    full = tl.load(INDICES + row, row < ROWS, other=0).to(tl.int32)
    source = (full // LENGTH) * SB + head * SH + (full % LENGTH) * SL + channel * SD
    value = tl.load(X + source, row < ROWS, other=0)
    tl.store(OUT + offset, value, row < ROWS)


def pack_attention(x, indices):
    heads, dim = x.shape[1], x.shape[-1]
    out = torch.empty((1, indices.numel(), heads, dim), device=x.device, dtype=torch.float16)
    attention_pack_kernel[(triton.cdiv(out.numel(), 1024),)](
        x, indices, out, indices.numel(), x.shape[2], *x.stride(), heads, dim, 1024, num_warps=4)
    return out
