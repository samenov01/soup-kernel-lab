"""Frozen Qwen2 RMSNorm vs the exact pinned Transformers implementation."""

import pytest
import torch

from solution.rmsnorm import patch_qwen2_rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def norm_patch(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution import rmsnorm

    original = Qwen2RMSNorm.forward
    calls = []
    original_run = rmsnorm.rmsnorm_forward_kernel.run

    def record(*args, **kwargs):
        calls.append(1)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(rmsnorm.rmsnorm_forward_kernel, "run", record)
    yield Qwen2RMSNorm, original, calls
    Qwen2RMSNorm.forward = original


@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 17, 1001), (1, 1), (2, 4097)])
@pytest.mark.parametrize("scale", [0.0, 1e-5, 1.0, 1000.0])
def test_frozen_norm_forward_backward(norm_patch, shape, scale):
    cls, original, calls = norm_patch
    torch.manual_seed(31)
    norm = cls(shape[-1], eps=1e-6).cuda().requires_grad_(False)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    x = (torch.randn(shape, device="cuda") * scale).requires_grad_()
    before = x.detach().clone()
    # Exercise noncontiguous upstream gradients while inputs remain contiguous.
    grad = torch.randn((*shape, 2), device="cuda")[..., 0]
    ref = original(norm, x)
    dr, = torch.autograd.grad(ref, x, grad)
    patch_qwen2_rmsnorm()
    got = norm(x)
    dg, = torch.autograd.grad(got, x, grad)
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(dg, dr, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
    assert norm.weight.grad is None
    assert len(calls) == 1
    assert torch.isfinite(got).all() and torch.isfinite(dg).all()


@pytest.mark.parametrize("mode", ["trainable", "fp16", "noncontiguous", "cpu", "empty"])
def test_norm_fallbacks(norm_patch, mode):
    cls, original, calls = norm_patch
    torch.manual_seed(5)
    device = "cpu" if mode == "cpu" else "cuda"
    norm = cls(17, eps=1e-6).to(device).requires_grad_(mode == "trainable")
    x = torch.randn(3, 17, device=device)
    if mode == "fp16":
        x = x.half()
    elif mode == "noncontiguous":
        x = torch.randn(3, 34, device=device)[:, ::2]
    elif mode == "empty":
        x = x[:0]
    x.requires_grad_()
    params = (x, norm.weight) if mode == "trainable" else (x,)
    ref = original(norm, x)
    grads_ref = torch.autograd.grad(ref.sum(), params)
    patch_qwen2_rmsnorm()
    first = cls.forward
    patch_qwen2_rmsnorm()
    assert cls.forward is first
    out = norm(x)
    grads_out = torch.autograd.grad(out.sum(), params)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    for got, expected in zip(grads_out, grads_ref):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
    assert not calls


def test_tiny_qwen_model_integration(norm_patch):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    _, _, calls = norm_patch
    torch.manual_seed(44)
    config = Qwen2Config(vocab_size=101, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32)
    model = Qwen2ForCausalLM(config).cuda().eval().requires_grad_(False)
    x = torch.randn(2, 7, 64, device="cuda", requires_grad=True)
    ref = model(inputs_embeds=x, use_cache=False).logits
    ref_grad, = torch.autograd.grad(ref.square().mean(), x)
    patch_qwen2_rmsnorm()
    got = model(inputs_embeds=x, use_cache=False).logits
    grad, = torch.autograd.grad(got.square().mean(), x)
    torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(grad, ref_grad, rtol=3e-4, atol=2e-6)
    assert len(calls) == 5  # two norms per decoder layer and the final norm
