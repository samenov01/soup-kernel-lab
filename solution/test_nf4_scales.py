import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_decoded_scale_metadata_reuse_and_invalidation():
    from bitsandbytes import functional as F
    from solution.nf4_scales import decoded_scales
    torch.manual_seed(15)
    w = torch.randn(256, 128, device="cuda")
    packed, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    original = F.dequantize_4bit(packed, state)
    plain = decoded_scales(state)
    torch.testing.assert_close(F.dequantize_4bit(packed, plain), original, atol=0, rtol=0)
    assert decoded_scales(state) is plain
    assert state.nested and state.absmax.dtype == torch.uint8
    state.absmax[0] = (state.absmax[0].int() + 1).clamp(max=255).to(torch.uint8)
    newer = decoded_scales(state)
    assert newer is not plain
    torch.testing.assert_close(F.dequantize_4bit(packed, newer), F.dequantize_4bit(packed, state), atol=0, rtol=0)


def test_cached_scale_matmul_forward_and_gradient():
    import bitsandbytes as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from solution.nf4_scales import patch_nf4_scales
    torch.manual_seed(64)
    packed, state = bnb.functional.quantize_4bit(torch.randn(256, 128, device="cuda") * 0.02,
                                               quant_type="nf4", compress_statistics=True)
    x = torch.randn(33, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    dy = torch.randn(33, 256, device="cuda", dtype=torch.float16) * 256
    ref = bnb.matmul_4bit(x, packed, state)
    dr, = torch.autograd.grad(ref, x, dy)
    original = MatMul4Bit.forward
    try:
        patch_nf4_scales()
        got = bnb.matmul_4bit(x, packed, state)
        dg, = torch.autograd.grad(got, x, dy)
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        torch.testing.assert_close(dg, dr, atol=0, rtol=0)
        assert state.nested and state.absmax.dtype == torch.uint8
        assert not got.grad_fn.state.nested
    finally:
        MatMul4Bit.forward = staticmethod(original)
