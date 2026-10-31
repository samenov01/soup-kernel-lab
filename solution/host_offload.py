"""Own lossless CPU offload; never enables Soup's built-in offload modes.

Frozen tied embedding/head parameters retain their values, dtype and identity.
Large decoder saved tensors are copied to pinned host memory and restored on
the original CUDA device. All transfers take place during the measured run.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .chunked_head import _ACTIVE
from .streamed_head import StreamedHeadLoss, host_half_weight

# This is storage policy inside our own autograd hooks, not a trainer setting.
# Only activations that backward really consumes are retained; no reserve or
# unused allocation is created. Storage aliases are counted once per forward.
GPU_SAVED_BUDGET = 512 * 2**20


@dataclass
class _HostTensor:
    value: torch.Tensor
    device: torch.device


def pack_activation(tensor):
    if (not tensor.is_cuda or tensor.numel() * tensor.element_size() < 65536
            or tensor.dtype not in (torch.float16, torch.float32, torch.bool)):
        return tensor.detach()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    host.copy_(tensor.detach(), non_blocking=True)
    return _HostTensor(host, tensor.device)


def unpack_activation(packed):
    if not isinstance(packed, _HostTensor):
        return packed
    return packed.value.to(packed.device, non_blocking=True)


def budgeted_packer(budget=GPU_SAVED_BUDGET):
    retained = set()
    used = 0
    def pack(tensor):
        nonlocal used
        if (tensor.is_cuda and tensor.dtype in (torch.float16, torch.float32, torch.bool)
                and tensor.numel() * tensor.element_size() >= 65536):
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in retained:
                return tensor.detach()
            if used + storage.nbytes() <= budget:
                retained.add(key)
                used += storage.nbytes()
                return tensor.detach()
        return pack_activation(tensor)
    return pack


def offload_embedding(model):
    embedding, head = model.model.embed_tokens, model.lm_head
    if embedding.weight is not head.weight:
        raise RuntimeError("Host embedding requires the task's tied head weight")
    weight = embedding.weight
    if weight.is_cuda:
        embedding._soup_embedding_device = weight.device
        with torch.no_grad():
            weight.data = weight.detach().cpu()
    if not hasattr(embedding, "_soup_embedding_device"):
        raise RuntimeError("Embedding must originate on the configured CUDA device")
    embedding._soup_host_embedding = True
    head._soup_host_head = True
    head._soup_streamed_loss = StreamedHeadLoss
    model.model._soup_saved_offload = True


def patch_host_offload():
    import peft
    from peft.utils import other
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
    original_prepare = peft.prepare_model_for_kbit_training
    if getattr(original_prepare, "_soup_host_offload", False):
        return
    original_embedding = torch.nn.Embedding.forward
    original_linear = torch.nn.Linear.forward
    original_body = Qwen2Model.forward

    def prepare(model, *args, **kwargs):
        supported = (type(model) is Qwen2ForCausalLM and getattr(model, "is_loaded_in_4bit", False)
                     and model.model.embed_tokens.weight is model.lm_head.weight
                     and model.model.embed_tokens.weight.is_cuda)
        if supported:
            # Move before PEFT widens this frozen parameter to FP32, avoiding
            # a simultaneous full Half + Float CUDA copy at model setup.
            offload_embedding(model)
        result = original_prepare(model, *args, **kwargs)
        if supported and (result.lm_head.weight.dtype != torch.float32 or result.lm_head.weight.requires_grad):
            raise RuntimeError("PEFT frozen head dtype contract changed")
        return result

    def embedding(self, ids):
        if not getattr(self, "_soup_host_embedding", False):
            return original_embedding(self, ids)
        if self.weight.device.type != "cpu":
            raise RuntimeError("Frozen embedding unexpectedly moved from host storage")
        return original_embedding(self, ids.cpu()).to(self._soup_embedding_device, non_blocking=False)

    def linear(self, x):
        if not getattr(self, "_soup_host_head", False):
            return original_linear(self, x)
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16:
            weight = host_half_weight(self.weight).to(x.device, non_blocking=True)
        else:
            weight = self.weight.to(x.device)
        return F.linear(x, weight, self.bias)

    def body(self, *args, **kwargs):
        if not (getattr(self, "_soup_saved_offload", False) and self.training and _ACTIVE.get()):
            return original_body(self, *args, **kwargs)
        with torch.autograd.graph.saved_tensors_hooks(budgeted_packer(), unpack_activation):
            return original_body(self, *args, **kwargs)

    prepare._soup_host_offload = True
    peft.prepare_model_for_kbit_training = prepare
    other.prepare_model_for_kbit_training = prepare
    torch.nn.Embedding.forward = embedding
    torch.nn.Linear.forward = linear
    Qwen2Model.forward = body
