import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_full_qkv_compact_backward_and_padding(dtype):
    from bitsandbytes import functional as F
    from solution.qkv_io import _FullQKV
    from solution.merged_qkv import _QKV
    torch.manual_seed(82)
    width, sizes = 128, (128, 32, 32)
    w = torch.randn(sum(sizes), width, device="cuda", dtype=torch.float32) * 0.02
    packed, state = F.quantize_4bit(w, quant_type="nf4", compress_statistics=True)
    bias = torch.randn(sum(sizes), device="cuda", dtype=torch.float16) * 0.02
    x = torch.randn(2, 33, width, device="cuda", dtype=dtype, requires_grad=True)
    indices = torch.arange(5, 60, device="cuda")
    inverse = torch.full((66,), -1, device="cuda", dtype=torch.long)
    inverse[indices] = torch.arange(indices.numel(), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        small = _QKV.apply(x.flatten(0, 1).index_select(0, indices), packed, state, bias, sizes)
        ref = tuple(y.new_zeros(66, n).index_copy(0, indices, y).view(2, 33, n) for y, n in zip(small, sizes))
    dy = tuple(torch.randn_like(y) * 256 for y in ref)
    dr, = torch.autograd.grad(ref, x, dy)
    with torch.autocast("cuda", dtype=torch.float16):
        got = _FullQKV.apply(x, packed, state, bias, sizes, indices, inverse)
    dg, = torch.autograd.grad(got, x, dy)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
