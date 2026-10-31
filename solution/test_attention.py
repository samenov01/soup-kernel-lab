import pytest
import torch
import torch.nn.functional as F
from solution.attention import _CausalGQA

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("length,dim,side", [(33, 64, "left"), (129, 128, "right"), (512, 128, "right")])
@pytest.mark.parametrize("amp_scale", [1., 65536.])
def test_causal_gqa_outputs_and_scaled_gradients(length, dim, side, amp_scale):
    torch.manual_seed(38)
    q = torch.randn(2, 12, length, dim, device="cuda", requires_grad=True) * 0.3
    k = torch.randn(2, 2, length, dim, device="cuda", requires_grad=True) * 0.3
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.ones(2, length, device="cuda", dtype=torch.long)
    if side == "left": mask[0, :length // 2] = 0
    else: mask[0, length // 2:] = 0
    causal = torch.ones(length, length, device="cuda", dtype=torch.bool).tril()
    allowed = causal[None, None] & mask[:, None, None, :].bool()
    with torch.autocast("cuda", dtype=torch.float16):
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(6, 1), v.repeat_interleave(6, 1), attn_mask=allowed)
    upstream = torch.randn_like(ref) * mask[:, None, :, None] * 0.01
    dr = torch.autograd.grad(ref, (q, k, v), upstream * amp_scale)
    actual = _CausalGQA.apply(q, k, v, mask, dim ** -0.5)
    dg = torch.autograd.grad(actual, (q, k, v), upstream * amp_scale)
    live = mask[:, None, :, None].expand_as(ref).bool()
    torch.testing.assert_close(actual[live], ref[live], atol=2e-3, rtol=3e-3)
    for a, b in zip(dg, dr):
        torch.testing.assert_close(a / amp_scale, b / amp_scale, atol=3e-5, rtol=2e-2)
        assert torch.isfinite(a).all()
