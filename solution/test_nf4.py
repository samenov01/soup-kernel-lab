"""Compare unpacking and real bitsandbytes matmul/autograd before/after patch."""

import pytest
import torch

from solution.nf4 import patch_nf4_dequantization

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def backend(monkeypatch):
    from bitsandbytes.backends.cuda import ops
    from solution import nf4
    original = ops._dequantize_4bit_impl
    launches = []
    original_run = nf4.nf4_unpack_scale_kernel.run

    def record_launch(*args, **kwargs):
        launches.append(1)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(nf4.nf4_unpack_scale_kernel, "run", record_launch)
    yield ops, original, launches
    ops._dequantize_4bit_impl = original


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape,blocksize", [((19, 65), 64), ((128, 256), 128), ((8960, 1536), 64)])
def test_nf4_dequantization(backend, dtype, nested, shape, blocksize):
    _, _, launches = backend
    from bitsandbytes import functional as F
    torch.manual_seed(73)
    weights = torch.randn(shape, dtype=dtype, device="cuda")
    packed, state = F.quantize_4bit(weights, blocksize=blocksize,
                                  compress_statistics=nested, quant_type="nf4")
    reference = F.dequantize_4bit(packed, state)
    packed_before, scales_before = packed.clone(), state.absmax.clone()
    patch_nf4_dequantization()
    actual = F.dequantize_4bit(packed, state)
    # A lookup and one fp32 multiplication, with the same decoded scales.
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    target = torch.empty_like(reference)
    result = F.dequantize_4bit(packed, state, out=target)
    assert result.data_ptr() == target.data_ptr()
    torch.testing.assert_close(target, reference, atol=0, rtol=0)
    torch.testing.assert_close(F.dequantize_4bit(packed.t(), state), reference.t(), atol=0, rtol=0)
    assert torch.equal(packed, packed_before) and torch.equal(state.absmax, scales_before)
    assert torch.isfinite(actual).all()
    assert len(launches) == 3  # plain, caller-provided out, transposed packed input


def test_all_codes_odd_tail_and_scales(backend):
    ops, original, launches = backend
    # All 16 nibble values, including repeated codes, zero and negative scales.
    packed = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(3)
    n = packed.numel() * 2 - 1
    scales = torch.linspace(-2, 2, (n + 63) // 64, device="cuda")
    reference = torch.empty(n, device="cuda")
    original(packed, scales, 64, "nf4", torch.float32, out=reference)
    patch_nf4_dequantization()
    actual = torch.empty_like(reference)
    ops._dequantize_4bit_impl(packed, scales, 64, "nf4", torch.float32, out=actual)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    first = ops._dequantize_4bit_impl
    patch_nf4_dequantization()
    assert ops._dequantize_4bit_impl is first
    assert len(launches) == 1


def test_fp4_fallback(backend):
    _, _, launches = backend
    from bitsandbytes import functional as F
    x = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(x, quant_type="fp4")
    expected = F.dequantize_4bit(packed, state)
    patch_nf4_dequantization()
    torch.testing.assert_close(F.dequantize_4bit(packed, state), expected, atol=0, rtol=0)
    assert not launches


@pytest.mark.parametrize("nested", [False, True])
def test_matmul_forward_and_input_gradient(backend, nested):
    _, _, launches = backend
    import bitsandbytes as bnb
    torch.manual_seed(17)
    # >1536 rows selects bnb's dequantize + linear path on its CUDA backend.
    x = torch.randn(2, 800, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    weights = torch.randn(128, 256, device="cuda", dtype=torch.float16) * 0.02
    packed, state = bnb.functional.quantize_4bit(weights, quant_type="nf4", compress_statistics=nested)
    bias = torch.randn(128, device="cuda", dtype=torch.float16)
    upstream = torch.randn(2, 800, 128, device="cuda", dtype=torch.float16)
    ref = bnb.matmul_4bit(x, packed, state, bias=bias)
    ref_grad, = torch.autograd.grad(ref, x, upstream)
    patch_nf4_dequantization()
    actual = bnb.matmul_4bit(x, packed, state, bias=bias)
    actual_grad, = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, ref_grad, atol=0, rtol=0)
    assert torch.isfinite(actual_grad).all()
    assert len(launches) == 2  # both forward and input-gradient really use our kernel
