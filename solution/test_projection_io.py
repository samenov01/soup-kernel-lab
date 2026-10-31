import pytest
import torch
from solution.projection_io import PackedInput, FullOutput

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width", [16, 257, 1536])
def test_projection_io_and_scaled_gradient(width):
    torch.manual_seed(88)
    x = torch.randn(2, 33, width, device="cuda", requires_grad=True)
    indices = torch.arange(0, 66, 2, device="cuda")
    inverse = torch.full((66,), -1, device="cuda", dtype=torch.long)
    inverse[indices] = torch.arange(indices.numel(), device="cuda")
    reference_packed = x.reshape(-1, width).index_select(0, indices).half()
    reference = torch.zeros_like(x).flatten(0, 1).index_copy(0, indices, reference_packed.float()).view_as(x)
    actual_packed = PackedInput.apply(x, indices, inverse)
    actual = FullOutput.apply(actual_packed, indices, inverse, x.shape, x.dtype)
    grad = torch.randn_like(x) * 0.01 * 65536
    dr, = torch.autograd.grad(reference, x, grad)
    dg, = torch.autograd.grad(actual, x, grad)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
