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
