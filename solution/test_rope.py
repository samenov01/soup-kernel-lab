import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("length,dim,cos_batch", [(17, 64, 1), (513, 128, 2)])
def test_rope_forward_backward(length, dim, cos_batch):
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from solution.rope import patch_rope
    original = qwen.apply_rotary_pos_emb
    torch.manual_seed(751)
    q = torch.randn(2, length, 12, dim, device="cuda").transpose(1, 2).requires_grad_()
    k = torch.randn(2, length, 2, dim, device="cuda").transpose(1, 2).requires_grad_()
    # Deliberately use distinct halves to verify the general transpose derivative.
    cos = torch.randn(cos_batch, length, dim, device="cuda")
    sin = torch.randn_like(cos)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    ref = original(q, k, cos, sin)
    gradients = torch.autograd.grad(ref, (q, k), (gq, gk))
    try:
        patch_rope()
        actual = qwen.apply_rotary_pos_emb(q, k, cos, sin)
        grad = torch.autograd.grad(actual, (q, k), (gq, gk))
        for a, b in zip(actual, ref):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(grad, gradients):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    finally:
        qwen.apply_rotary_pos_emb = original
