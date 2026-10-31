import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width,out_width", [(128, 32), (1536, 1536)])
def test_lora_autograd_exact_scaled_gradients(dtype, width, out_width):
    from solution.lora_autograd import CompactLoRA
    from solution.compact_lora import SparseLoRAAdd
    torch.manual_seed(252)
    base = torch.randn(2, 33, out_width, device="cuda", dtype=dtype, requires_grad=True)
    x = torch.randn(2, 33, width, device="cuda", dtype=dtype, requires_grad=True)
    a = torch.randn(16, width, device="cuda", requires_grad=True) * 0.02
    b = torch.randn(out_width, 16, device="cuda", requires_grad=True) * 0.02
    indices = torch.arange(5, 60, device="cuda")
    inverse = torch.full((66,), -1, device="cuda", dtype=torch.long)
    inverse[indices] = torch.arange(indices.numel(), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = torch.nn.functional.linear(x.flatten(0, 1).index_select(0, indices), a)
        delta = torch.nn.functional.linear(hidden, b)
        ref = SparseLoRAAdd.apply(base, delta, indices, inverse, 2.)
        got = CompactLoRA.apply(base, x, a, b, indices, inverse, 2.)
    dy = torch.randn_like(ref) * 256
    dr = torch.autograd.grad(ref, (base, x, a, b), dy)
    dg = torch.autograd.grad(got, (base, x, a, b), dy)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    for actual, expected in zip(dg, dr):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
