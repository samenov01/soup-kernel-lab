import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_half_activation_norm_fp32_arithmetic(dtype):
    from solution.half_activations import _ActivationNorm
    torch.manual_seed(731)
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(128, device="cuda", dtype=torch.float32)
    eps = 1e-6
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(dtype)
    actual = _ActivationNorm.apply(x, weight, eps, dtype)
    grad = torch.randn_like(actual) * 0.001 * 65536
    dr, = torch.autograd.grad(ref, x, grad)
    dg, = torch.autograd.grad(actual, x, grad)
    torch.testing.assert_close(actual, ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(dg, dr, atol=0.125, rtol=3e-3)
    assert torch.isfinite(dg).all()


def test_decoder_half_storage_keeps_parameters_and_head_dtype():
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from bitsandbytes.nn import Linear4bit
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    from solution.half_activations import patch_half_activations
    from solution.chunked_head import _ACTIVE

    torch.manual_seed(532)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=257, hidden_size=128,
        intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attention_dropout=0.0))
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                source = getattr(parent, name)
                target = Linear4bit(source.in_features, source.out_features,
                    bias=source.bias is not None, compute_dtype=torch.float16,
                    quant_type="nf4", compress_statistics=True)
                target.weight.data.copy_(source.weight)
                if source.bias is not None: target.bias.data.copy_(source.bias)
                setattr(parent, name, target)
    model = model.cuda().requires_grad_(False)
    model.is_loaded_in_4bit = True
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                            target_modules=["q_proj", "v_proj"]))
    for name, p in model.named_parameters():
        if "lora_B" in name: p.data.normal_(std=0.01)
    model.train()
    x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
    mask = torch.ones(2, 17, device="cuda", dtype=torch.long)
    mask[0, 10:] = 0
    labels = torch.randint(257, mask.shape, device="cuda")
    labels[mask == 0] = -100
    params = [p for p in model.parameters() if p.requires_grad]
    originals = qwen.Qwen2Model.forward, qwen.Qwen2DecoderLayer.forward, qwen.Qwen2RMSNorm.forward
    dtypes = {name: p.dtype for name, p in model.named_parameters()}

    def run():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits.float()
            loss = (output.square() * mask[..., None]).mean()
        gradients = torch.autograd.grad(loss * 65536, [x, *params])
        return output, [g / 65536 for g in gradients]

    token = _ACTIVE.set(SimpleNamespace(labels=labels, attention_mask=mask))
    recorded = []
    hooks = []
    try:
        reference, dr = run()
        patch_half_activations()
        base = model.get_base_model().model
        hooks.append(base.layers[0].mlp.register_forward_hook(
            lambda _module, _args, out: recorded.append(("mlp", out.dtype))))
        hooks.append(base.layers[0].input_layernorm.register_forward_pre_hook(
            lambda _module, args: recorded.append(("norm_input", args[0].dtype))))
        hooks.append(base.norm.register_forward_hook(
            lambda _module, _args, out: recorded.append(("final", out.dtype))))
        actual, dg = run()
        torch.testing.assert_close(actual[mask.bool()], reference[mask.bool()], atol=3e-3, rtol=3e-2)
        for a, b in zip(dg, dr):
            torch.testing.assert_close(a, b, atol=5e-6, rtol=5e-2)
            assert (a - b).norm() / b.norm().clamp_min(1e-8) < 0.05
            assert torch.isfinite(a).all()
        assert ("mlp", torch.float16) in recorded and ("final", torch.float32) in recorded
        assert ("norm_input", torch.float16) in recorded
        assert {name: p.dtype for name, p in model.named_parameters()} == dtypes
        # Ordinary model calls outside the SFT context retain their original path.
        _ACTIVE.reset(token)
        token = _ACTIVE.set(False)
        recorded.clear()
        run()
        assert ("final", torch.float32) in recorded
        assert ("norm_input", torch.float32) in recorded
    finally:
        _ACTIVE.reset(token)
        for hook in hooks: hook.remove()
        qwen.Qwen2Model.forward, qwen.Qwen2DecoderLayer.forward, qwen.Qwen2RMSNorm.forward = originals
