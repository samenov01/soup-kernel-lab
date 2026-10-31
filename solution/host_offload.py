"""Own lossless CPU offload; never enables Soup's built-in offload modes.

Frozen tied embedding/head parameters retain their values, dtype and identity.
Large decoder saved tensors are copied to pinned host memory and restored on
the original CUDA device. All transfers take place during the measured run.
"""

from dataclasses import dataclass
import weakref
import torch
import torch.nn.functional as F
from .chunked_head import _ACTIVE, _ChunkedHeadLoss
from .streamed_head import StreamedHeadLoss

# This is storage policy inside our own autograd hooks, not a trainer setting.
# Only activations that backward really consumes are retained; no reserve or
# unused allocation is created. Storage aliases are counted once per forward.
GPU_SAVED_BUDGET = 192 * 2**20
# Speed candidate: spend VRAM on reusable operands instead of PCIe copies.
# The external <=2000 MiB peak still needs a complete Kaggle measurement.
STREAM_VOCABULARY = False
OFFLOAD_MLP = False


def resident_half_weight(weight, device):
    """Reuse a real frozen head operand, built inside the measured process.

    One GPU copy per parameter; refresh on writes, storage/layout changes or
    a different execution device. Never cache logits, outputs or gradients.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if weight.requires_grad or weight.dtype != torch.float32 or weight.device.type != "cpu":
        raise RuntimeError("Resident head requires a frozen FP32 CPU parameter")
    signature = (weight.data_ptr(), weight._version, tuple(weight.shape),
                 tuple(weight.stride()), device)
    cached = getattr(weight, "_soup_head_device_cache", None)
    if cached is None or cached[0] != signature:
        # Convert the current values directly; don't depend on another cache's
        # invalidation contract. Only the Half operand is materialized on GPU.
        half = weight.detach().half().pin_memory().to(device, non_blocking=True)
        cached = (signature, half)
        weight._soup_head_device_cache = cached
    return cached[1]


class _HostBackedFullHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor, mask):
        # Keep only the original Half matmul operand on GPU; the FP32 model
        # parameter stays on CPU. This retains the original head GEMM shapes.
        half = resident_half_weight(weight, hidden.device)
        return _ChunkedHeadLoss.forward(ctx, hidden, half, labels, divisor, mask)

    @staticmethod
    def backward(ctx, *gradients):
        return _ChunkedHeadLoss.backward(ctx, *gradients)


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
    offloaded = {}
    used = 0
    def pack(tensor):
        nonlocal used
        if (not tensor.is_cuda or tensor.dtype not in (torch.float16, torch.float32, torch.bool)
                or tensor.numel() * tensor.element_size() < 65536):
            return tensor.detach()
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in retained:
            return tensor.detach()
        if used + storage.nbytes() <= budget:
            retained.add(key)
            used += storage.nbytes()
            return tensor.detach()
        # The same activation can be saved by several operations. Reuse its
        # host snapshot only for the very same live Tensor and version. Weak
        # references prevent this lookup table from retaining GPU storage.
        key = id(tensor)
        previous = offloaded.get(key)
        if (previous is not None and previous[0]() is tensor
                and previous[1] == tensor._version):
            return previous[2]
        packed = pack_activation(tensor)
        if isinstance(packed, _HostTensor):
            offloaded[key] = (weakref.ref(tensor), tensor._version, packed)
        return packed
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
    head._soup_streamed_loss = StreamedHeadLoss if STREAM_VOCABULARY else _HostBackedFullHead
    model.model._soup_saved_offload = True
    if not weight.requires_grad:
        from .host_storage import keep_weight_on_host
        keep_weight_on_host(embedding)
        keep_weight_on_host(head)


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
    original_apply = Qwen2ForCausalLM._apply

    def prepare(model, *args, **kwargs):
        supported = (type(model) is Qwen2ForCausalLM and getattr(model, "is_loaded_in_4bit", False)
                     and model.model.embed_tokens.weight is model.lm_head.weight
                     and (model.model.embed_tokens.weight.is_cuda
                          or getattr(model, "_soup_cpu_embedding_load", False)))
        if supported:
            # Move before PEFT widens this frozen parameter to FP32, avoiding
            # a simultaneous full Half + Float CUDA copy at model setup.
            offload_embedding(model)
        result = original_prepare(model, *args, **kwargs)
        if supported and (result.lm_head.weight.dtype != torch.float32 or result.lm_head.weight.requires_grad):
            raise RuntimeError("PEFT frozen head dtype contract changed")
        if supported:
            offload_embedding(result)
            if OFFLOAD_MLP and getattr(model, "_soup_cpu_embedding_load", False):
                from .host_mlp import offload_mlp
                for layer in result.model.layers:
                    offload_mlp(layer.mlp)
        return result

    def embedding(self, ids):
        if not getattr(self, "_soup_host_embedding", False):
            return original_embedding(self, ids)
        if self.weight.device.type != "cpu":
            raise RuntimeError("Frozen embedding unexpectedly moved from host storage")
        return original_embedding(self, ids.cpu()).to(self._soup_embedding_device, non_blocking=False)

    def apply_model(self, fn, recurse=True):
        marked = getattr(self.model.embed_tokens, "_soup_host_embedding", False)
        result = original_apply(self, fn, recurse=recurse)
        if marked:
            # Framework model.to(device) can run after PEFT setup. Restore our
            # storage policy after that ordinary placement, without changing
            # its requested execution device or training arguments.
            if self.model.embed_tokens.weight is not self.lm_head.weight:
                self.tie_weights()
            offload_embedding(self)
        return result

    def linear(self, x):
        if not getattr(self, "_soup_host_head", False):
            return original_linear(self, x)
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16:
            weight = resident_half_weight(self.weight, x.device)
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
    Qwen2ForCausalLM._apply = apply_model
