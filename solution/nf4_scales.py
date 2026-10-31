"""Reuse decoded frozen NF4 scale metadata, never decoded weight matrices."""

from copy import copy
import torch


def decoded_scales(state):
    from bitsandbytes import functional as F
    if not state.nested:
        return state
    tensors = (state.absmax, state.state2.absmax, state.state2.code, state.offset)
    signature = tuple((t.data_ptr(), t._version) for t in tensors)
    cached = getattr(state, "_soup_decoded_scales", None)
    if cached is None or cached[0] != signature:
        plain = copy(state)
        if hasattr(plain, "_soup_decoded_scales"):
            del plain._soup_decoded_scales
        plain.absmax = F.dequantize_blockwise(state.absmax, state.state2) + state.offset
        plain.nested, plain.state2, plain.offset = False, None, None
        cached = (signature, plain)
        state._soup_decoded_scales = cached
    return cached[1]


def patch_nf4_scales():
    from bitsandbytes.autograd._functions import MatMul4Bit
    original = MatMul4Bit.forward
    if getattr(original, "_soup_decoded_scales", False):
        return

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        if (quant_state is not None and quant_state.nested and quant_state.quant_type == "nf4"
                and A.is_cuda and A.dtype == torch.float16 and not B.requires_grad):
            quant_state = decoded_scales(quant_state)
        return original(ctx, A, B, out, bias, quant_state)

    forward._soup_decoded_scales = True
    MatMul4Bit.forward = staticmethod(forward)
