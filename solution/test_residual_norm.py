import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width", [64, 1536])
def test_fused_residual_norm_scaled_gradient(width):
    from solution.residual_norm import ResidualNorm
    from solution.half_activations import _ActivationNorm
    torch.manual_seed(921)
    a = torch.randn(2, 33, width, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    w = torch.randn(width, device="cuda", dtype=torch.float32)
    r = a + b
    y = _ActivationNorm.apply(r, w, 1e-6, torch.float16)
    dr, dy = torch.randn_like(r) * 256, torch.randn_like(y) * 256
    reference = torch.autograd.grad((r, y), (a, b), (dr, dy))
    rr, yy = ResidualNorm.apply(a, b, w, 1e-6)
    actual = torch.autograd.grad((rr, yy), (a, b), (dr, dy))
    torch.testing.assert_close(rr, r, atol=0, rtol=0)
    torch.testing.assert_close(yy, y, atol=0, rtol=0)
    for got, ref in zip(actual, reference):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
