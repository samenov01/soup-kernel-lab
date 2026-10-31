"""Valid-token outputs, LoRA gradients and exact dropout RNG consumption."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("side", ["left", "right"])
def test_attention_projection_compaction(side):
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Model, Qwen2MLP
    from solution.padded_mlp import patch_padded_mlp
    from solution.padded_attention import patch_padded_attention

    originals = (Qwen2Attention.forward, Qwen2Model.forward, Qwen2MLP.forward,
                 torch.nn.Linear.forward, Linear4bit.forward)
    torch.manual_seed(952)
    config = Qwen2Config(vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0.0)
    model = Qwen2ForCausalLM(config)
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            old = getattr(layer.self_attn, name)
            quant = Linear4bit(old.in_features, old.out_features, bias=old.bias is not None,
                compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True)
            quant.weight.data.copy_(old.weight)
            if old.bias is not None:
                quant.bias.data.copy_(old.bias)
            setattr(layer.self_attn, name, quant)
    model = model.cuda().requires_grad_(False)
    model.is_loaded_in_4bit = True
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"]))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.01)
    model.train()
    x = torch.randn(2, 33, 128, device="cuda", requires_grad=True)
    mask = torch.ones(2, 33, device="cuda", dtype=torch.long)
    if side == "left": mask[0, :17] = 0
    else: mask[0, 16:] = 0
    params = [p for p in model.parameters() if p.requires_grad]

    def run():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits.float()
            loss = (output.square() * mask[..., None]).sum()
        return output, torch.autograd.grad(loss, [x, *params]), torch.cuda.get_rng_state()

    try:
        patch_padded_mlp()
        rng = torch.cuda.get_rng_state()
        ref, grads, rng_ref = run()
        torch.cuda.set_rng_state(rng)
        patch_padded_attention()
        actual, actual_grads, rng_actual = run()
        torch.testing.assert_close(actual[mask.bool()], ref[mask.bool()], atol=2e-3, rtol=3e-3)
        for a, b in zip(actual_grads, grads):
            torch.testing.assert_close(a, b, atol=5e-3, rtol=1e-2)
        assert torch.equal(rng_ref, rng_actual)
        assert (actual_grads[0][~mask.bool()] == 0).all()
    finally:
        (Qwen2Attention.forward, Qwen2Model.forward, Qwen2MLP.forward,
         torch.nn.Linear.forward, Linear4bit.forward) = originals
