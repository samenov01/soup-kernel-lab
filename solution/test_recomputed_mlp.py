import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("width,intermediate,rows", [(128, 256, 33), (1536, 8960, 512)])
def test_recomputed_mlp_values_scaled_gradient_and_storage(width, intermediate, rows):
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
    with torch.autocast("cuda", dtype=torch.float16):
        ref = m.down_proj(_CompactSwiGLU.apply(*merged_gate_up(m, x)))
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
                                       m.down_proj.weight.quant_state)
    dg, = torch.autograd.grad(got, x, dy)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)
    torch.testing.assert_close(dg, dr, atol=0, rtol=0)
    assert (rows, width) in saved
    assert not any(len(s) == 2 and s[0] == rows and s[1] >= intermediate for s in saved)
