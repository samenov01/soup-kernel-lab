"""Lossless storage helpers for frozen parameters, without allocator changes."""

from copy import copy
from types import MethodType
import torch


def pinned_copy(tensor):
    # Copy the tensor bytes, not Params4bit.to(cpu)'s backend-specific packing.
    return tensor.data.detach().to(device="cpu", copy=True).pin_memory()


def copy_quant_state(state, device):
    result = copy(state)
    # A cached decoded state may hold CUDA arrays even after the visible state
    # moves. Recreate that optional cache on the device that consumes it.
    result.__dict__.pop("_soup_decoded_scales", None)
    for name in ("absmax", "code", "offset"):
        tensor = getattr(state, name, None)
        if isinstance(tensor, torch.Tensor):
            value = pinned_copy(tensor) if torch.device(device).type == "cpu" else tensor.to(device, non_blocking=True)
            setattr(result, name, value)
    if state.state2 is not None:
        result.state2 = copy_quant_state(state.state2, device)
    return result


def device_operands(packed, state, device):
    if packed.device == torch.device(device):
        return packed, state
    if packed.device.type != "cpu":
        raise RuntimeError("Frozen NF4 storage must be on CPU or its execution device")
    return packed.data.to(device, non_blocking=True), copy_quant_state(state, device)


def keep_weight_on_host(module):
    """Keep a frozen weight on CPU through Module.to(), preserving dtype casts.

    The module's other parameters/buffers still receive the ordinary fn. A
    zero-element probe resolves fn's target dtype/device without moving the
    full host weight. The parameter's real values are retained on CPU.
    """
    if getattr(module, "_soup_keep_host_weight", False):
        return
    original = module._apply

    def apply_host(self, fn, recurse=True):
        weight = self.weight
        if weight.device.type != "cpu" or weight.requires_grad:
            raise RuntimeError("Host placement requires a frozen CPU weight")

        def place(tensor):
            if tensor is not weight:
                return fn(tensor)
            probe = fn(torch.empty(0, dtype=tensor.dtype, device="cpu"))
            if hasattr(self, "_soup_embedding_device"):
                self._soup_embedding_device = probe.device
            # Use a plain Tensor so Params4bit cannot repack or requantize it.
            return tensor.data.detach().to(device="cpu", dtype=probe.dtype)

        return original(place, recurse=recurse)

    module._apply = MethodType(apply_host, module)
    module._soup_keep_host_weight = True
