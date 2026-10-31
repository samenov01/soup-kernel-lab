"""Merge two frozen MLP projections, retaining only packed NF4 metadata.

The packed concatenation is built during the measured run. It is neither a
dequantized-weight cache nor a model parameter; the saved adapter is untouched.
"""

from copy import copy
import torch
from torch.autograd.function import once_differentiable


class _GateUp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, state):
        import bitsandbytes as bnb
        output = bnb.matmul_4bit(x.to(torch.float16), packed, state)
        ctx.save_for_backward(packed)
        ctx.state = state
        ctx.dtype = x.dtype
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        from bitsandbytes import functional as F
        packed, = ctx.saved_tensors
        state = copy(ctx.state)
        state.dtype = torch.float16
        weight = F.dequantize_4bit(packed, state)
        return torch.matmul(grad, weight).to(ctx.dtype), None, None


def _scales(state):
    from bitsandbytes import functional as F
    if state.nested:
        return F.dequantize_blockwise(state.absmax, state.state2) + state.offset
    return state.absmax


def _signature(weight):
    state = weight.quant_state
    tensors = [weight, state.absmax]
    if state.nested:
        tensors.extend((state.state2.absmax, state.state2.code, state.offset))
    return (tuple(state.shape), state.blocksize, state.quant_type,
            tuple((t.data_ptr(), t._version) for t in tensors))


def share_packed_storage(weights, packed):
    """Give frozen original parameters views of their unchanged packed bytes.

    Parameters and quantization metadata remain intact. This avoids retaining
    both the original packed weights and a second concatenated copy per layer.
    """
    offset = 0
    with torch.no_grad():
        for weight in weights:
            size = weight.numel()
            weight.data = packed[offset:offset + size].view_as(weight)
            offset += size


def packed_gate_up(module):
    from bitsandbytes.functional import QuantState
    gate, up = module.gate_proj.weight, module.up_proj.weight
    a, b = gate.quant_state, up.quant_state
    supported = (
        a.quant_type == b.quant_type == "nf4" and a.shape == b.shape
        and len(a.shape) == 2 and a.shape[0] * a.shape[1] % a.blocksize == 0
        and a.blocksize == b.blocksize and a.shape[0] * a.shape[1] % 2 == 0
    )
    if not supported:
        return None
    signature = (_signature(gate), _signature(up))
    cached = getattr(module, "_soup_merged_gate_up", None)
    if cached is None or cached[0] != signature:
        packed = torch.cat((gate.view(-1, 1), up.view(-1, 1)))
        scales = torch.cat((_scales(a), _scales(b)))
        state = QuantState(scales, shape=torch.Size((2 * a.shape[0], a.shape[1])),
                           code=a.code, blocksize=a.blocksize, quant_type="nf4", dtype=torch.float32)
        share_packed_storage((gate, up), packed)
        signature = (_signature(gate), _signature(up))
        cached = (signature, packed, state)
        module._soup_merged_gate_up = cached
    return cached


def merged_gate_up(module, x):
    cached = packed_gate_up(module)
    if cached is None:
        return module.gate_proj(x.to(torch.float16)), module.up_proj(x.to(torch.float16))
    merged = _GateUp.apply(x, cached[1], cached[2])
    return merged.chunk(2, dim=-1)


def patch_merged_gate_up():
    from . import compact_mlp
    compact_mlp._gate_up = merged_gate_up
