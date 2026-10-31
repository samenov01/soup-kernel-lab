import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("side,dtype", [("left", torch.float32), ("right", torch.float16)])
def test_native_unpadded_causal_sdpa(side, dtype, monkeypatch):
    from types import SimpleNamespace
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from solution.chunked_head import _ACTIVE
    from solution.padded_mlp import _TOKENS
    from solution.unpadded_sdpa import patch_unpadded_sdpa

    torch.manual_seed(323)
    module = Qwen2Attention(Qwen2Config(hidden_size=1536, num_attention_heads=12,
                                       num_key_value_heads=2), 0).train()
    q = torch.randn(2, 12, 129, 128, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 129, 128, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.ones(2, 129, device="cuda", dtype=torch.long)
    if side == "left": mask[0, :64] = 0
    else: mask[0, 65:] = 0
    allowed = torch.ones(129, 129, device="cuda", dtype=torch.bool).tril()[None, None] & mask[:, None, None].bool()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    with torch.autocast("cuda", dtype=torch.float16):
        reference, _ = original(module, q, k, v, allowed, scaling=128 ** -0.5)
    grad = torch.randn_like(reference) * mask[:, :, None, None] * 0.01 * 65536
    dr = torch.autograd.grad(reference, (q, k, v), grad)
    token = _ACTIVE.set(SimpleNamespace(attention_mask=mask))
    tok = _TOKENS.set((mask.flatten().nonzero().flatten(), mask.shape))
    from solution import varlen_attention as varlen
    native = varlen._native_forward
    calls = []

    def record(query, *args, **kwargs):
        calls.append(query.shape)
        return native(query, *args, **kwargs)

    try:
        monkeypatch.setattr(varlen, "_native_forward", record)
        patch_unpadded_sdpa()
        with torch.autocast("cuda", dtype=torch.float16):
            actual, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, allowed, scaling=128 ** -0.5)
        dg = torch.autograd.grad(actual, (q, k, v), grad)
        live = mask.bool()
        torch.testing.assert_close(actual[live], reference[live], atol=2e-3, rtol=3e-3)
        for a, b in zip(dg, dr):
            torch.testing.assert_close(a.float() / 65536, b.float() / 65536, atol=3e-5, rtol=5e-3)
            assert torch.isfinite(a).all()
        assert len(calls) == 1 and calls[0] == (1, 194, 12, 128)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        _ACTIVE.reset(token)
        _TOKENS.reset(tok)
