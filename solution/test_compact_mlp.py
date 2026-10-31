"""Lossless internal dtype storage relative to the shipped SwiGLU solution."""

import pytest
import torch

from solution.compact_mlp import patch_compact_mlp
from solution.swiglu import patch_qwen2_mlp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def mlp_class():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    original = Qwen2MLP.forward
    patch_qwen2_mlp()
    yield Qwen2MLP, Qwen2MLP.forward
    Qwen2MLP.forward = original


@pytest.mark.parametrize("shape,width,intermediate", [((2, 19), 128, 256), ((4, 512), 1536, 8960)])
def test_compact_mlp_bitwise_and_saved_storage(mlp_class, shape, width, intermediate):
    from bitsandbytes.nn import Linear4bit
    from transformers import Qwen2Config
    cls, reference_forward = mlp_class
    torch.manual_seed(14)
    config = Qwen2Config(hidden_size=width, intermediate_size=intermediate, hidden_act="silu")
    model = cls(config)
    for name in ("gate_proj", "up_proj", "down_proj"):
        source = getattr(model, name)
        target = Linear4bit(source.in_features, source.out_features, bias=False,
                            compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True)
        target.weight.data.copy_(source.weight.data)
        setattr(model, name, target)
    model = model.cuda().requires_grad_(False)
    x = torch.randn((*shape, width), device="cuda", requires_grad=True)
    grad = torch.randn_like(x)
    ref_saved, actual_saved = [], []

    def record(destination):
        def pack(t):
            if t.shape == (*shape, intermediate):
                destination.append((t.dtype, t.numel() * t.element_size()))
            return t
        return pack

    with torch.autograd.graph.saved_tensors_hooks(record(ref_saved), lambda t: t):
        ref = reference_forward(model, x)
    dr, = torch.autograd.grad(ref, x, grad)
    patch_compact_mlp()
    with torch.autograd.graph.saved_tensors_hooks(record(actual_saved), lambda t: t):
        got = model(x)
    dg, = torch.autograd.grad(got, x, grad)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
    assert got.dtype == torch.float32 and dg.dtype == torch.float32
    assert len(ref_saved) == len(actual_saved) == 2
    assert all(dtype == torch.float32 for dtype, _ in ref_saved)
    assert all(dtype == torch.float16 for dtype, _ in actual_saved)
    assert sum(n for _, n in actual_saved) * 2 == sum(n for _, n in ref_saved)


def test_unquantized_mlp_fallback(mlp_class):
    from transformers import Qwen2Config
    cls, original = mlp_class
    model = cls(Qwen2Config(hidden_size=64, intermediate_size=128)).cuda()
    x = torch.randn(2, 13, 64, device="cuda", requires_grad=True)
    ref = original(model, x)
    dr, = torch.autograd.grad(ref.sum(), x)
    patch_compact_mlp()
    first = cls.forward
    patch_compact_mlp()
    assert cls.forward is first
    got = model(x)
    dg, = torch.autograd.grad(got.sum(), x)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
