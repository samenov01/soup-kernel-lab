import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("heads", [2, 12])
@pytest.mark.parametrize("layout", ["BHLD", "BLHD", "broadcast"])
def test_attention_pack_strides_and_half_cast(heads, layout):
    from solution.attention_io import pack_attention
    torch.manual_seed(9)
    x = torch.randn(2, heads, 129, 128, device="cuda")
    if layout == "BLHD": x = x.transpose(1, 2).contiguous().transpose(1, 2)
    if layout == "broadcast": x = x[:, :, :1].expand_as(x)
    indices = torch.arange(13, 240, 2, device="cuda")
    ref = x.transpose(1, 2).reshape(258, heads, 128).index_select(0, indices).half()[None]
    torch.testing.assert_close(pack_attention(x, indices), ref, atol=0, rtol=0)
