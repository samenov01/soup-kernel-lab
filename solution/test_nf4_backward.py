"""The direct-half operand and real input gradient must be bitwise identical."""

import pytest
import torch

from solution.nf4 import patch_nf4_dequantization
from solution.nf4_backward import patch_nf4_backward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("rows", [32, 1600])
def test_backward_direct_half(nested, rows):
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from bitsandbytes.backends.cuda import ops

    original_backward = MatMul4Bit.backward
    original_decode = ops._dequantize_4bit_impl
    torch.manual_seed(94)
    weight = torch.randn(128, 256, device="cuda", dtype=torch.float32) * 0.03
    packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=nested)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    upstream = torch.randn(rows, 128, device="cuda", dtype=torch.float16)
    output = bnb.matmul_4bit(x, packed, state)
    reference, = torch.autograd.grad(output, x, upstream)
    seen = []
    try:
        patch_nf4_dequantization()
        decode = ops._dequantize_4bit_impl

        def record(A, absmax, blocksize, quant_type, dtype, out):
            seen.append(dtype)
            return decode(A, absmax, blocksize, quant_type, dtype, out=out)

        ops._dequantize_4bit_impl = record
        patch_nf4_backward()
        output2 = bnb.matmul_4bit(x, packed, state)
        seen.clear()
        actual, = torch.autograd.grad(output2, x, upstream)
        torch.testing.assert_close(output2, output, atol=0, rtol=0)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        assert seen == [torch.float16]
        assert state.dtype == torch.float32
        patched = MatMul4Bit.backward
        patch_nf4_backward()
        assert MatMul4Bit.backward is patched
    finally:
        MatMul4Bit.backward = staticmethod(original_backward)
        ops._dequantize_4bit_impl = original_decode
