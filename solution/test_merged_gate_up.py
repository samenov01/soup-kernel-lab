import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width,intermediate,rows", [(128, 256, 33), (1536, 8960, 512)])
def test_merged_frozen_gate_up(width, intermediate, rows):
    from bitsandbytes.nn import Linear4bit
    from types import SimpleNamespace
    from solution.merged_gate_up import merged_gate_up
    from solution.nf4 import patch_nf4_dequantization
    from bitsandbytes.backends.cuda import ops
    original_decode = ops._dequantize_4bit_impl
    torch.manual_seed(42)
    layers = []
    for _ in range(2):
        layer = Linear4bit(width, intermediate, bias=False, compute_dtype=torch.float16,
                          quant_type="nf4", compress_statistics=True)
        layer.weight.data.normal_(std=0.02)
        layers.append(layer.cuda().requires_grad_(False))
    module = SimpleNamespace(gate_proj=layers[0], up_proj=layers[1])
    x = torch.randn(rows, width, device="cuda", requires_grad=True)
    upstream = torch.randn(rows, 2 * intermediate, device="cuda", dtype=torch.float16) * 0.01
    with torch.autocast("cuda", dtype=torch.float16):
        reference = torch.cat([layer(x.to(torch.float16)) for layer in layers], -1)
    dr, = torch.autograd.grad(reference, x, upstream * 65536)
    before = [layer.weight.clone() for layer in layers]
    try:
        patch_nf4_dequantization()
        with torch.autocast("cuda", dtype=torch.float16):
            actual = torch.cat(merged_gate_up(module, x), -1)
        dg, = torch.autograd.grad(actual, x, upstream * 65536)
        torch.testing.assert_close(actual, reference, rtol=3e-3, atol=2e-3)
        torch.testing.assert_close(dg / 65536, dr / 65536, rtol=3e-3, atol=1e-4)
        packed = module._soup_merged_gate_up[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            merged_gate_up(module, x)
        assert module._soup_merged_gate_up[1] is packed
        assert packed.dtype == torch.uint8
        for layer, old in zip(layers, before):
            assert torch.equal(layer.weight, old)
    finally:
        ops._dequantize_4bit_impl = original_decode
