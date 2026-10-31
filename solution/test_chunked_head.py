"""Bounded logits loss, actual AMP scale, all metrics and Soup row-loss protocol."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from solution.chunked_head import _ChunkedHeadLoss, patch_chunked_head

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("shape,width,vocab", [((2, 7), 64, 257), ((2, 257), 128, 4097), ((1, 9), 128, 151936)])
@pytest.mark.parametrize("scale", [1.0, 65536.0])
def test_chunk_loss_gradient_and_metrics(shape, width, vocab, scale):
    torch.manual_seed(83)
    x = torch.randn((*shape, width), device="cuda", requires_grad=True)
    w = torch.randn(vocab, width, device="cuda") * 0.02
    labels = torch.randint(vocab, shape, device="cuda")
    labels[:, :2] = -100
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:].contiguous()
    denominator = (shifted != -100).sum()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = F.linear(x, w).float()
        token_loss = F.cross_entropy(logits.flatten(0, 1), shifted.flatten(), reduction="none").view(shape)
        reference = token_loss.sum() / denominator
        logps = F.log_softmax(logits, -1)
        expected_entropy = -(logps.exp() * logps).sum(-1)
        expected_predictions = logits.argmax(-1)
    dr, = torch.autograd.grad(reference * scale, x)
    saved = []

    def pack(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autocast("cuda", dtype=torch.float16):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            actual, tokens, entropy, predictions = _ChunkedHeadLoss.apply(x, w, labels, denominator)
    dg, = torch.autograd.grad(actual * scale, x)
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(tokens, token_loss, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(entropy, expected_entropy, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(dg / scale, dr / scale, atol=3e-6, rtol=3e-3)
    torch.testing.assert_close(predictions, expected_predictions, atol=0, rtol=0)
    assert not tokens.requires_grad and not entropy.requires_grad
    assert not any(len(s) == 2 and s[-1] == vocab for s in saved)
    assert torch.isfinite(dg).all()


def test_sft_metrics_and_rewind_protocol():
    from accelerate.utils.operations import convert_to_fp32
    from soup_cli.trainer.rewind_hf import row_losses, row_losses_from_token_losses
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTTrainer

    original_model = Qwen2ForCausalLM.forward
    original_compute = SFTTrainer.compute_loss
    torch.manual_seed(19)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=257, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2)).cuda().train().requires_grad_(False)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def get_base_model(self):
            return self.model

        def forward(self, **kwargs):
            # Match Accelerate: trainer.compute_loss itself is outside AMP.
            with torch.autocast("cuda", dtype=torch.float16):
                return convert_to_fp32(self.model(**kwargs))

    model = Wrapper(base)
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(num_processes=1, gather_for_metrics=lambda t: t)
    trainer.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll", packing=False,
        padding_free=False, torch_compile=False, past_index=-1, average_tokens_across_devices=False,
        fp16=True, bf16=False)
    trainer.num_virtual_tokens = 0
    trainer.aux_loss_enabled = False
    trainer.label_smoother = None
    trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = True
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    trainer._total_train_tokens = 0
    x = torch.randn(2, 7, 64, device="cuda", requires_grad=True)
    labels = torch.randint(257, (2, 7), device="cuda")
    labels[0, :3] = -100
    labels[1] = -100
    mask = torch.ones_like(labels)
    count = (labels[:, 1:] != -100).sum()
    inputs = {"inputs_embeds": x, "labels": labels, "attention_mask": mask}
    try:
        reference, ref_outputs = original_compute(trainer, model, dict(inputs), True, count)
        dr, = torch.autograd.grad(reference, x)
        expected_metrics = dict(trainer._metrics["train"])
        expected_rows = row_losses(ref_outputs.logits, labels)
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer._total_train_tokens = 0
        patch_chunked_head()
        first = SFTTrainer.compute_loss
        patch_chunked_head()
        assert SFTTrainer.compute_loss is first
        assert not torch.is_autocast_enabled("cuda")
        actual, outputs = trainer.compute_loss(model, dict(inputs), True, count)
        dg, = torch.autograd.grad(actual, x)
        assert outputs.logits is None
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=3e-5)
        torch.testing.assert_close(dg, dr, rtol=5e-3, atol=3e-6)
        for name, values in expected_metrics.items():
            assert trainer._metrics["train"][name] == pytest.approx(values, abs=3e-5, rel=3e-4)
        rows = row_losses_from_token_losses(getattr(trainer, STEP_ATTR).token_loss, labels)
        assert rows[0] == pytest.approx(expected_rows[0], abs=3e-5, rel=3e-4)
        assert rows[1] == expected_rows[1]
        # Ordinary model calls outside the SFT loss context retain dense logits.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            dense = model(**inputs)
        assert dense.logits.shape == (2, 7, 257)
    finally:
        Qwen2ForCausalLM.forward = original_model
        SFTTrainer.compute_loss = original_compute
