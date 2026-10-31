import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width,intermediate,rows,padding", [(128, 256, 33, False), (1536, 8960, 512, False), (128, 256, 66, True)])
def test_recomputed_mlp_values_scaled_gradient_and_storage(width, intermediate, rows, padding):
    from bitsandbytes.nn import Linear4bit
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from solution.merged_gate_up import merged_gate_up, packed_gate_up
    from solution.compact_mlp import _CompactSwiGLU
    from solution.recomputed_mlp import _RecomputedMLP
    torch.manual_seed(929)
    m = Qwen2MLP(Qwen2Config(hidden_size=width, intermediate_size=intermediate))
    for name in ("gate_proj", "up_proj", "down_proj"):
        old = getattr(m, name)
        new = Linear4bit(old.in_features, old.out_features, bias=False,
                        compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True)
        new.weight.data.copy_(old.weight.data)
        setattr(m, name, new)
    m = m.cuda().requires_grad_(False)
    x = torch.randn(rows, width, device="cuda", dtype=torch.float16, requires_grad=True)
    indices, inverse = None, None
    if padding:
        x = x.detach().view(2, 33, width).requires_grad_(True)
        indices = torch.arange(5, 60, device="cuda")
        inverse = torch.full((66,), -1, device="cuda", dtype=torch.long)
        inverse[indices] = torch.arange(indices.numel(), device="cuda")
    selected = x.flatten(0, 1).index_select(0, indices) if padding else x
    with torch.autocast("cuda", dtype=torch.float16):
        ref = m.down_proj(_CompactSwiGLU.apply(*merged_gate_up(m, selected)))
        if padding:
            ref = torch.zeros_like(x.flatten(0, 1)).index_copy(0, indices, ref).view_as(x)
    dy = torch.randn_like(ref) * 256
    dr, = torch.autograd.grad(ref, x, dy)
    cache = packed_gate_up(m)
    saved = []
    def pack(t):
        saved.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cuda", dtype=torch.float16):
            got = _RecomputedMLP.apply(x, cache[1], cache[2], m.down_proj.weight,
                                       m.down_proj.weight.quant_state, indices, inverse)
    dg, = torch.autograd.grad(got, x, dy)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
    assert (indices.numel() if padding else rows, width) in saved
    assert not any(len(s) == 2 and s[0] == rows and s[1] >= intermediate for s in saved)
