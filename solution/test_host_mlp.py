"""GPU checks for exact NF4 storage and the real local-checkpoint load path."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width,intermediate,rows", [(128, 256, 33), (1536, 8960, 512)])
def test_host_mlp_exact_output_scaled_backward_and_placement(width, intermediate, rows):
    from bitsandbytes.nn import Linear4bit
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution.host_mlp import offload_mlp, resident_mlp
    from solution.merged_gate_up import packed_gate_up
    from solution.recomputed_mlp import _RecomputedMLP
    torch.manual_seed(810)
    m = Qwen2MLP(Qwen2Config(hidden_size=width, intermediate_size=intermediate))
    for name in ("gate_proj", "up_proj", "down_proj"):
        old = getattr(m, name)
        new = Linear4bit(old.in_features, old.out_features, bias=False,
                        compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True)
        new.weight.data.copy_(old.weight.data)
        setattr(m, name, new)
    m = m.cuda().requires_grad_(False)
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16, requires_grad=True)
    cache = packed_gate_up(m)
    parameter_objects = dict(m.named_parameters())
    original = {name: p.data.cpu().clone() for name, p in m.named_parameters()}
    with torch.autocast("cuda", dtype=torch.float16):
        expected = _RecomputedMLP.apply(x, cache[1], cache[2], m.down_proj.weight,
                                        m.down_proj.weight.quant_state)
    dy = torch.randn_like(expected) * 256
    dr, = torch.autograd.grad(expected, x, dy)
    del cache
    offload_mlp(m)
    m.to("cuda")  # Framework placement must not move the host weight matrices.
    for name, p in m.named_parameters():
        assert p is parameter_objects[name]
        assert p.device.type == "cpu" and not p.requires_grad
        torch.testing.assert_close(p.data, original[name], atol=0, rtol=0)
        assert p.quant_state.absmax.device.type == "cpu"
    cache = packed_gate_up(m)
    saved = []
    def pack(t):
        saved.append(t.detach())
        return t.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.float16):
            actual = _RecomputedMLP.apply(x, cache[1], cache[2], m.down_proj.weight,
                                         m._soup_host_down_state)
    dg, = torch.autograd.grad(actual, x, dy)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
    assert not any(t.is_cuda and t.dtype == torch.uint8 for t in saved)
    assert m.down_proj.weight.quant_state.absmax.device.type == "cpu"
    with resident_mlp(m, x.device):
        assert all(p.is_cuda for p in m.parameters())
    assert all(p.device.type == "cpu" for p in m.parameters())


def test_host_load_config_preserves_original_config_and_quantization():
    from transformers import Qwen2Config, BitsAndBytesConfig
    from transformers.modeling_utils import LoadStateDictConfig
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from solution.host_mlp import host_load_config
    quantizer = Bnb4BitHfQuantizer(BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4"))
    config = LoadStateDictConfig(hf_quantizer=quantizer, device_map={"": 0},
                                dtype=torch.float16, offload_buffers=False)
    model = SimpleNamespace(config=Qwen2Config(tie_word_embeddings=True))
    local, device = host_load_config(model, config)
    assert config.device_map == {"": 0}
    assert config.__dataclass_params__.frozen
    assert local is not config and local.device_map is not config.device_map
    assert local.dtype == config.dtype and local.offload_buffers == config.offload_buffers
    assert local.hf_quantizer is quantizer and device == torch.device("cuda:0")
    assert local.device_map == {"": 0, "model.embed_tokens": "cpu", "lm_head": "cpu"}


def test_local_nf4_checkpoint_peft_forward_backward_after_host_loading(tmp_path):
    # No hub request: a tiny locally generated model exercises the same HF
    # loader -> quantizer -> PEFT preparation -> model.to -> LoRA path.
    import peft
    from peft.utils import other
    from transformers import Qwen2Config, Qwen2ForCausalLM, BitsAndBytesConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2MLP
    from solution.host_offload import patch_host_offload
    from solution.host_mlp import patch_host_mlp_loading
    config = Qwen2Config(vocab_size=257, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        tie_word_embeddings=True)
    torch.manual_seed(729)
    Qwen2ForCausalLM(config).save_pretrained(tmp_path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    def load():
        return Qwen2ForCausalLM.from_pretrained(tmp_path, quantization_config=deepcopy(quant),
                                               device_map={"": 0}, dtype=torch.float16)
    baseline = load()
    baseline = peft.prepare_model_for_kbit_training(baseline, use_gradient_checkpointing=False)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                           lora_dropout=0.05, task_type="CAUSAL_LM")
    torch.manual_seed(83)
    baseline = peft.get_peft_model(baseline, deepcopy(lora)).train()
    ids = torch.arange(34, device="cuda").view(2, 17)
    torch.manual_seed(49)
    with torch.autocast("cuda", dtype=torch.float16):
        ref = baseline(input_ids=ids, labels=ids, use_cache=False).loss
    ref.backward()
    expected = {n: p.grad.detach().clone() for n, p in baseline.named_parameters() if p.requires_grad}
    originals = (peft.prepare_model_for_kbit_training, other.prepare_model_for_kbit_training,
                 torch.nn.Embedding.forward, torch.nn.Linear.forward, Qwen2Model.forward,
                 Qwen2ForCausalLM._apply, Qwen2ForCausalLM._load_pretrained_model, Qwen2MLP.forward)
    try:
        patch_host_offload()
        patch_host_mlp_loading()
        model = load()
        assert model.model.embed_tokens.weight.device.type == "cpu"
        assert model.model.embed_tokens.weight is model.lm_head.weight
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        torch.manual_seed(83)
        model = peft.get_peft_model(model, deepcopy(lora)).train().to("cuda")
        for layer in model.get_base_model().model.layers:
            assert layer.mlp.down_proj.weight.device.type == "cpu"
        assert all(p.is_cuda for p in model.parameters() if p.requires_grad)
        torch.manual_seed(49)
        with torch.autocast("cuda", dtype=torch.float16):
            got = model(input_ids=ids, labels=ids, use_cache=False).loss
        got.backward()
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
        for n, p in model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, expected[n], atol=0, rtol=0)
    finally:
        (peft.prepare_model_for_kbit_training, other.prepare_model_for_kbit_training,
         torch.nn.Embedding.forward, torch.nn.Linear.forward, Qwen2Model.forward,
         Qwen2ForCausalLM._apply, original_load, Qwen2MLP.forward) = originals
        Qwen2ForCausalLM._load_pretrained_model = staticmethod(original_load)
