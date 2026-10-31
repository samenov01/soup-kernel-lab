"""Store frozen MLP NF4 bytes on CPU; compute forward/backward on CUDA."""

from contextlib import contextmanager
from dataclasses import replace
import torch
from .host_storage import pinned_copy, copy_quant_state, keep_weight_on_host
from .merged_gate_up import packed_gate_up, share_packed_storage, _signature
from .nf4_scales import decoded_scales


def offload_mlp(module):
    if getattr(module, "_soup_host_mlp", False):
        return
    from bitsandbytes.nn import Linear4bit
    layers = (module.gate_proj, module.up_proj, module.down_proj)
    if not all(type(p) is Linear4bit and p.bias is None and p.weight.is_cuda
               and not p.weight.requires_grad for p in layers):
        raise RuntimeError("Host MLP requires frozen CUDA NF4 projections without bias")
    cached = packed_gate_up(module)
    if cached is None:
        raise RuntimeError("Host MLP requires compatible gate/up NF4 layouts")
    packed = pinned_copy(cached[1])
    state = copy_quant_state(cached[2], "cpu")
    # Decode nested scales using the original CUDA operation before offload.
    # Both quantized bytes and the resulting FP32 scale values stay exact.
    down_state = copy_quant_state(decoded_scales(module.down_proj.weight.quant_state), "cpu")
    down = pinned_copy(module.down_proj.weight)
    for layer in layers:
        layer.weight.quant_state = copy_quant_state(layer.weight.quant_state, "cpu")
        layer.quant_state = layer.weight.quant_state
    share_packed_storage([module.gate_proj.weight, module.up_proj.weight], packed)
    with torch.no_grad():
        module.down_proj.weight.data = down
    signature = (_signature(module.gate_proj.weight), _signature(module.up_proj.weight))
    module._soup_merged_gate_up = (signature, packed, state)
    module._soup_host_down_state = down_state
    module._soup_host_mlp = True
    for layer in layers:
        keep_weight_on_host(layer)


@contextmanager
def resident_mlp(module, device):
    """Fallback for ordinary model calls outside the optimized Half path."""
    layers = (module.gate_proj, module.up_proj, module.down_proj)
    original = [(p.weight.data, p.weight.quant_state, p.quant_state) for p in layers]
    cache = module._soup_merged_gate_up
    try:
        for layer, (data, state, _) in zip(layers, original):
            with torch.no_grad():
                layer.weight.data = data.to(device, non_blocking=True)
            layer.weight.quant_state = copy_quant_state(state, device)
            layer.quant_state = layer.weight.quant_state
        yield
    finally:
        for layer, (data, state, module_state) in zip(layers, original):
            with torch.no_grad():
                layer.weight.data = data
            layer.weight.quant_state = state
            layer.quant_state = module_state
        module._soup_merged_gate_up = cache


def host_load_config(model, config):
    """Change only embedding/head storage in a private loader config copy.

    The original checkpoint loader, CUDA NF4 quantizer and allocator warmup
    still run. No offload mode or trainer flag is enabled, and the caller's
    device map is not mutated. Our own code owns subsequent placement.
    """
    quantizer = config.hf_quantizer
    mapping = config.device_map
    if (model.config.model_type != "qwen2" or not model.config.tie_word_embeddings
            or quantizer is None or type(quantizer).__name__ != "Bnb4BitHfQuantizer"
            or quantizer.quantization_config.bnb_4bit_quant_type != "nf4"
            or not isinstance(mapping, dict) or set(mapping) != {""}):
        return None
    target = mapping[""]
    device = torch.device("cuda", target) if isinstance(target, int) else torch.device(target)
    if device.type != "cuda":
        return None
    # Transformers LoadStateDictConfig is frozen. Construct a new instance
    # rather than assigning fields on a shallow copy of that frozen object.
    local = replace(config, device_map={**mapping, "model.embed_tokens": "cpu", "lm_head": "cpu"})
    return local, device


def patch_host_mlp_loading():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP
    from .half_activations import _HALF_MODEL
    original_load = Qwen2ForCausalLM._load_pretrained_model
    original_mlp = Qwen2MLP.forward
    if getattr(original_load, "_soup_host_load", False):
        return

    def load(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        plan = host_load_config(model, load_config)
        if plan is None:
            return original_load(model, state_dict, checkpoint_files, load_config, expected_keys)
        local, device = plan
        result = original_load(model, state_dict, checkpoint_files, local, expected_keys)
        model.model.embed_tokens._soup_embedding_device = device
        model._soup_cpu_embedding_load = True
        return result

    def mlp(self, x):
        if not getattr(self, "_soup_host_mlp", False):
            return original_mlp(self, x)
        from bitsandbytes.nn import Linear4bit
        layers = (self.gate_proj, self.up_proj, self.down_proj)
        if (getattr(original_mlp, "_soup_recomputed_mlp", False)
                and _HALF_MODEL.get() and self.training and x.is_cuda and x.dtype == torch.float16
                and self.config.hidden_act == "silu"
                and all(type(p) is Linear4bit and p.compute_dtype == torch.float16
                        and not p.weight.requires_grad for p in layers)):
            return original_mlp(self, x)
        with resident_mlp(self, x.device):
            return original_mlp(self, x)

    load._soup_host_load = True
    Qwen2ForCausalLM._load_pretrained_model = staticmethod(load)
    Qwen2MLP.forward = mlp
