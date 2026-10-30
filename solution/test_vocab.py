"""Numerical and call-site tests. Run with the pinned Kaggle dependencies."""

import pytest
import torch
import torch.nn.functional as F

from solution.vocab import token_cross_entropy, entropy_forward, patch_vocab_reductions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and Triton")


@pytest.mark.parametrize("vocab", [1, 17, 4095, 4096, 4097, 151936])
@pytest.mark.parametrize("strided", [False, True])
def test_ce_forward_backward(vocab, strided):
    torch.manual_seed(42)
    # Row padding tests stride support without a contiguous copy of logits.
    width = vocab + 7 if strided else vocab
    storage = torch.randn(9, width, device="cuda", requires_grad=True)
    x = storage[:, :vocab]
    target_storage = torch.randint(vocab, (18,), device="cuda")
    labels = target_storage[::2]
    labels[::3] = -100
    labels[-1] = vocab - 1
    upstream = torch.randn(18, device="cuda")[::2]
    expected = F.cross_entropy(x, labels, reduction="none")
    expected_grad, = torch.autograd.grad(expected, storage, upstream)
    actual = token_cross_entropy(x, labels)
    actual_grad, = torch.autograd.grad(actual, storage, upstream)
    # fp32 reductions and approximate exp/log: small absolute and relative error.
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-5)
    assert torch.isfinite(actual).all() and torch.isfinite(actual_grad).all()
    assert (actual[labels == -100] == 0).all()
    assert (actual_grad[labels == -100] == 0).all()


@pytest.mark.parametrize("mode", ["uniform", "peaked", "shifted", "all_ignored"])
def test_extreme_finite_logits(mode):
    x = torch.zeros(3, 8193, device="cuda")
    if mode == "peaked":
        x[:, 100] = 80
        x[:, 200] = -80
    if mode == "shifted":
        x += 1000
    x.requires_grad_()
    labels = torch.tensor([0, 100, -100], device="cuda")
    if mode == "all_ignored":
        labels.fill_(-100)
    ref = F.cross_entropy(x, labels, reduction="none")
    got = token_cross_entropy(x, labels)
    ref_grad, = torch.autograd.grad(ref.sum(), x)
    got_grad, = torch.autograd.grad(got.sum(), x)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(got_grad, ref_grad, atol=2e-6, rtol=1e-4)
    with torch.no_grad():
        lp = F.log_softmax(x, -1)
        entropy = entropy_forward(x)
        torch.testing.assert_close(entropy, -(lp.exp() * lp).sum(-1), atol=3e-5, rtol=3e-6)
        assert torch.isfinite(entropy).all()


@pytest.mark.parametrize("shape", [(2, 11, 4097), (1, 512, 151936)])
def test_entropy(shape):
    torch.manual_seed(19)
    x = torch.randn(shape, device="cuda")
    before = x.clone()
    with torch.no_grad():
        got = entropy_forward(x)
        # Chunk the reference to keep this test comfortably below T4 memory.
        ref = torch.cat([
            -(p.exp() * p).sum(-1)
            for p in (F.log_softmax(c, -1) for c in x.flatten(0, -2).split(16))
        ]).reshape(shape[:-1])
    torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-6)
    torch.testing.assert_close(x, before, atol=0, rtol=0)


def test_patched_call_sites_and_fallbacks():
    from transformers.loss import loss_utils
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf

    original_ce = loss_utils.fixed_cross_entropy
    original_entropy = sft_trainer.entropy_from_logits
    original_rows = rewind_hf.row_losses
    try:
        patch_vocab_reductions()
        first = loss_utils.fixed_cross_entropy
        patch_vocab_reductions()
        assert loss_utils.fixed_cross_entropy is first
        torch.manual_seed(28)
        logits = torch.randn(2, 7, 4097, device="cuda", requires_grad=True)
        labels = torch.randint(4097, (2, 7), device="cuda")
        labels[0, :3] = -100
        labels[1] = -100
        # Includes label shift and externally supplied normalization in HF's real loss.
        for divisor in [None, 4, torch.tensor(4)]:
            shifted = F.pad(labels, (0, 1), value=-100)[:, 1:].flatten()
            ref = original_ce(logits.flatten(0, 1), shifted, divisor)
            got = loss_utils.ForCausalLMLoss(logits, labels, 4097, divisor)
            torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-6)
            dg, = torch.autograd.grad(got, logits, retain_graph=True)
            dr, = torch.autograd.grad(ref, logits, retain_graph=True)
            torch.testing.assert_close(dg, dr, atol=2e-6, rtol=3e-5)
        ref_means, ref_counts = original_rows(logits, labels)
        means, counts = rewind_hf.row_losses(logits, labels)
        assert counts == ref_counts
        assert means == pytest.approx(ref_means, abs=3e-5, rel=3e-6)
        with torch.no_grad():
            torch.testing.assert_close(sft_trainer.entropy_from_logits(logits),
                                       original_entropy(logits), atol=3e-5, rtol=3e-6)
        # Differentiable entropy and unsupported dtype/layout keep the original.
        for x in [logits, logits.detach().half(), logits.detach()[:, ::2]]:
            torch.testing.assert_close(sft_trainer.entropy_from_logits(x), original_entropy(x))
        cpu_x = torch.randn(3, 9, requires_grad=True)
        cpu_y = torch.tensor([1, -100, 7])
        torch.testing.assert_close(first(cpu_x, cpu_y), original_ce(cpu_x, cpu_y))
        ignored = torch.full((14,), -100, device="cuda", dtype=torch.long)
        loss = first(logits.flatten(0, 1), ignored)
        assert torch.isnan(loss)
        grad, = torch.autograd.grad(loss, logits)
        assert (grad == 0).all()
    finally:
        loss_utils.fixed_cross_entropy = original_ce
        sft_trainer.entropy_from_logits = original_entropy
        rewind_hf.row_losses = original_rows
