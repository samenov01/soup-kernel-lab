import pytest
import torch
from solution.compact_lora import SparseLoRAAdd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width,scale", [(256, 0.5), (1536, 2.)])
def test_sparse_lora_scaling_and_amp_gradient(width, scale):
    torch.manual_seed(98)
    base = torch.randn(2, 33, width, device="cuda", requires_grad=True)
    indices = torch.arange(0, 66, 2, device="cuda")
    inverse = torch.full((66,), -1, device="cuda", dtype=torch.long)
    inverse[indices] = torch.arange(indices.numel(), device="cuda")
    delta = torch.randn(indices.numel(), width, device="cuda", dtype=torch.float16, requires_grad=True)
    full = torch.zeros(66, width, device="cuda", dtype=torch.float16).index_copy(0, indices, delta * scale)
    reference = base + full.view_as(base)
    actual = SparseLoRAAdd.apply(base, delta, indices, inverse, scale)
    gradient = torch.randn_like(base) * 0.01 * 65536
    dr = torch.autograd.grad(reference, (base, delta), gradient)
    dg = torch.autograd.grad(actual, (base, delta), gradient)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    for a, b in zip(dg, dr):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
