import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("vocab", [257, 9001, 151936])
@pytest.mark.parametrize("scale", [1., 65536.])
def test_host_streamed_loss_and_actual_scaled_gradient(vocab, scale):
    from solution.streamed_head import StreamedHeadLoss
    torch.manual_seed(818)
    x = torch.randn(2, 9, 128, device="cuda", requires_grad=True)
    w = torch.randn(vocab, 128, device="cuda") * 0.02
    labels = torch.randint(vocab, (2, 9), device="cuda")
    labels[:, :3] = -100
    mask = torch.ones_like(labels)
    target = F.pad(labels, (0, 1), value=-100)[:, 1:]
    count = (target != -100).sum()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = F.linear(x, w).float()
        tokens = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").view_as(labels)
        ref = tokens.sum() / count
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp() * logp).sum(-1)
    dr, = torch.autograd.grad(ref * scale, x)
    with torch.autocast("cuda", dtype=torch.float16):
        got, losses, ent, pred = StreamedHeadLoss.apply(x, w.cpu(), labels, count, mask)
    dg, = torch.autograd.grad(got * scale, x)
    torch.testing.assert_close(got, ref, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(losses, tokens, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(ent, entropy, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(pred, logits.argmax(-1), atol=0, rtol=0)
    torch.testing.assert_close(dg / scale, dr / scale, atol=5e-6, rtol=5e-3)
    assert torch.isfinite(dg).all()


def test_host_activation_exact_values_and_backward():
    from solution.host_offload import pack_activation, unpack_activation, _HostTensor
    torch.manual_seed(72)
    x = torch.randn(2, 33, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    seen = []
    def pack(t):
        result = pack_activation(t)
        seen.append(result)
        return result
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack_activation):
        y = (x * x).sum()
    grad, = torch.autograd.grad(y, x)
    torch.testing.assert_close(grad, x * 2, atol=0, rtol=0)
    assert any(isinstance(t, _HostTensor) and t.value.device.type == "cpu" for t in seen)


def test_host_tied_embedding_identity_dtype_and_values():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from solution.host_offload import patch_host_offload, offload_embedding
    import peft
    from peft.utils import other
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    originals = (peft.prepare_model_for_kbit_training, other.prepare_model_for_kbit_training,
                 torch.nn.Embedding.forward, torch.nn.Linear.forward, Qwen2Model.forward)
    torch.manual_seed(19)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=257, hidden_size=128, intermediate_size=256,
          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)).cuda()
    m.requires_grad_(False)
    ids = torch.randint(257, (2, 17), device="cuda")
    expected = m.model.embed_tokens(ids)
    weight = m.lm_head.weight
    try:
        patch_host_offload()
        offload_embedding(m)
        actual = m.model.embed_tokens(ids)
        assert m.lm_head.weight is weight and m.model.embed_tokens.weight is weight
        assert weight.dtype == torch.float32 and not weight.requires_grad
        assert weight.device.type == "cpu"
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        (peft.prepare_model_for_kbit_training, other.prepare_model_for_kbit_training,
         torch.nn.Embedding.forward, torch.nn.Linear.forward, Qwen2Model.forward) = originals
