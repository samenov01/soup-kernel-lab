"""Decode frozen NF4 backward operands directly to their consumed dtype."""

from copy import copy
import torch


def patch_nf4_backward():
    from bitsandbytes.autograd._functions import MatMul4Bit
    original = MatMul4Bit.backward
    if getattr(original, "_soup_direct_half", False):
        return

    def backward(ctx, grad_output):
        state = getattr(ctx, "state", None)
        if (ctx.is_empty or not ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
                or not grad_output.is_cuda or grad_output.dtype != torch.float16
                or state is None or state.quant_type != "nf4" or state.dtype != torch.float32):
            return original(ctx, grad_output)
        half_state = copy(state)
        half_state.dtype = torch.float16
        ctx.state = half_state
        try:
            return original(ctx, grad_output)
        finally:
            ctx.state = state

    backward._soup_direct_half = True
    MatMul4Bit.backward = staticmethod(backward)
