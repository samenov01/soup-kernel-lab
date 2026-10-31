"""Valid outputs and gradients stay equivalent; MLP really skips padding."""

import pytest
import torch

from solution.compact_mlp import patch_compact_mlp
from solution.padded_mlp import patch_padded_mlp
from solution.swiglu import patch_qwen2_mlp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("padding", ["right", "left", "none"])
def test_padding_compaction_model_outputs_and_gradient(padding):
    from bitsandbytes.nn import Linear4bit
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2MLP

    old_body, old_mlp = Qwen2Model.forward, Qwen2MLP.forward
    torch.manual_seed(62)
    config = Qwen2Config(vocab_size=257, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = Qwen2ForCausalLM(config)
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            quantized = Linear4bit(linear.in_features, linear.out_features, bias=False,
                compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True)
            quantized.weight.data.copy_(linear.weight.data)
            setattr(layer.mlp, name, quantized)
    model = model.cuda().train().requires_grad_(False)
    x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
    mask = torch.ones(2, 17, device="cuda", dtype=torch.long)
    if padding == "right":
        mask[0, 9:] = 0
    elif padding == "left":
        mask[0, :8] = 0
    calls = []
    hook = model.model.layers[0].mlp.gate_proj.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape))
    try:
        patch_qwen2_mlp()
        patch_compact_mlp()
        with torch.autocast("cuda", dtype=torch.float16):
            reference = model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits.float()
            reference_loss = (reference.square() * mask[..., None]).sum()
        dr, = torch.autograd.grad(reference_loss, x)
        patch_padded_mlp()
        with torch.autocast("cuda", dtype=torch.float16):
            actual = model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits.float()
            loss = (actual.square() * mask[..., None]).sum()
        dg, = torch.autograd.grad(loss, x)
        live = mask.bool()
        torch.testing.assert_close(actual[live], reference[live], atol=2e-3, rtol=3e-3)
        torch.testing.assert_close(dg, dr, atol=3e-3, rtol=5e-3)
        assert torch.isfinite(actual).all() and torch.isfinite(dg).all()
        if padding == "none":
            assert calls[-1] == (2, 17, 128)
        else:
            assert calls[-1] == (int(mask.sum()), 128)
            assert (dg[~live] == 0).all()
    finally:
        hook.remove()
        Qwen2Model.forward, Qwen2MLP.forward = old_body, old_mlp
