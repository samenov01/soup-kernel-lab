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
        supported = (
            not ctx.is_empty and ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
            and grad_output.is_cuda and grad_output.dtype == torch.float16
            and state is not None and state.quant_type == "nf4" and state.dtype == torch.float32
        )
        if not supported:
            return original(ctx, grad_output)
        # The original path decodes to fp32, then immediately casts to fp16.
        # Our unpack kernel multiplies in fp32 and rounds at its store, yielding
        # the same half operand without the temporary fp32 weight matrix.
        # A shallow metadata view leaves the model's actual quantization state
        # and all packed weights/scales untouched; no decoded weights are kept.
        half_state = copy(state)
        half_state.dtype = torch.float16
        ctx.state = half_state
        try:
            return original(ctx, grad_output)
        finally:
            ctx.state = state

    backward._soup_direct_half = True
    MatMul4Bit.backward = staticmethod(backward)
