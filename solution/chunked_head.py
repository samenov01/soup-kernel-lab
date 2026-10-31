"""Exact training outputs with bounded logits storage and AMP-aware backward.

Enabled only in the ordinary single-device SFT path that consumes the loss,
entropy, token accuracy and rewind values. Other model calls keep dense logits.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from torch.autograd.function import once_differentiable

from .vocab import (_partials, vocab_ce_finish_kernel, vocab_ce_backward_kernel,
                    vocab_entropy_finish_kernel, TILE)

_ACTIVE = ContextVar("soup_chunked_training_head", default=False)
CHUNK = 256


class _ChunkedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor):
        batch, length, width = hidden.shape
        vocab = weight.shape[0]
        # These are precisely the operand conversions of the original autocast linear.
        xhalf = hidden.reshape(-1, width).to(torch.float16)
        whalf = weight.to(torch.float16)
        target = F.pad(labels, (0, 1), value=-100)[:, 1:].contiguous().view(-1)
        count = (target != -100).sum() if divisor is None else divisor
        denominator = torch.as_tensor(count, device=hidden.device, dtype=torch.float32)
        rows = xhalf.shape[0]
        losses = torch.empty(rows, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(losses)
        predictions = torch.empty(rows, device=hidden.device, dtype=torch.long)
        norm = torch.empty((rows, 2), device=hidden.device, dtype=torch.float32)
        with torch.cuda.device(hidden.device):
            for start in range(0, rows, CHUNK):
                end = min(start + CHUNK, rows)
                logits = F.linear(xhalf[start:end], whalf).float()
                maxima, sums, moments, parts = _partials(logits, entropy=True)
                vocab_ce_finish_kernel[(end - start,)](
                    logits, target[start:end], maxima, sums, norm[start:end], losses[start:end],
                    vocab, vocab, parts, -100, triton.next_power_of_2(parts), num_warps=4, debug=True)
                vocab_entropy_finish_kernel[(end - start,)](
                    maxima, sums, moments, entropy[start:end], parts,
                    triton.next_power_of_2(parts), num_warps=4)
                predictions[start:end] = logits.argmax(dim=-1)
                del logits
        ctx.save_for_backward(xhalf, whalf, target, norm, denominator)
        ctx.shape = hidden.shape
        tokens = losses.view(batch, length)
        entropy = entropy.view(batch, length)
        predictions = predictions.view(batch, length)
        ctx.mark_non_differentiable(tokens, entropy, predictions)
        return losses.sum() / denominator, tokens, entropy, predictions

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_tokens, _grad_entropy, _grad_predictions):
        xhalf, whalf, target, norm, denominator = ctx.saved_tensors
        rows, width = xhalf.shape
        vocab = whalf.shape[0]
        dx = torch.empty((rows, width), device=xhalf.device, dtype=torch.float32)
        # GradScaler's actual upstream scale is applied BEFORE the fp16 cast,
        # matching the dense loss -> cast -> linear backward sequence.
        scale = grad_loss.float() / denominator
        with torch.cuda.device(xhalf.device):
            for start in range(0, rows, CHUNK):
                end = min(start + CHUNK, rows)
                logits = F.linear(xhalf[start:end], whalf).float()
                dlogits = torch.empty((end - start, vocab), device=xhalf.device, dtype=torch.float16)
                upstream = scale.expand(end - start).contiguous()
                vocab_ce_backward_kernel[(end - start, triton.cdiv(vocab, TILE))](
                    logits, target[start:end], norm[start:end], upstream, dlogits,
                    vocab, vocab, -100, TILE, num_warps=4)
                del logits
                dx[start:end] = torch.matmul(dlogits, whalf)
                del dlogits
        return dx.view(ctx.shape), None, None, None


def patch_chunked_head():
    from transformers import Trainer
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from trl import SFTTrainer
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR

    if getattr(SFTTrainer.compute_loss, "_soup_chunk_head_patch", False):
        return
    original_model = modeling_qwen2.Qwen2ForCausalLM.forward
    original_compute = SFTTrainer.compute_loss

    @dataclass
    class ChunkedOutput(CausalLMOutputWithPast):
        token_loss: torch.Tensor | None = None
        token_entropy: torch.Tensor | None = None
        token_predictions: torch.Tensor | None = None

    def head_supported(base):
        return (isinstance(base, modeling_qwen2.Qwen2ForCausalLM)
                and type(base.lm_head) is torch.nn.Linear and base.lm_head.bias is None
                and base.lm_head.weight.is_cuda and base.lm_head.weight.dtype == torch.float32
                and not base.lm_head.weight.requires_grad)

    def model_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      past_key_values=None, inputs_embeds=None, labels=None,
                      use_cache=None, logits_to_keep=0, **kwargs):
        if not _ACTIVE.get():
            return original_model(self, input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
        if not head_supported(self) or labels is None or logits_to_keep != 0:
            raise RuntimeError("Chunked training head contract changed")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
            raise RuntimeError("Chunked head requires the model's configured fp16 autocast")
        divisor = kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        if hidden.dtype != torch.float32:
            raise RuntimeError("Chunked training head expects the task's fp32 final normalization")
        loss, tokens, entropy, predictions = _ChunkedHeadLoss.apply(hidden, self.lm_head.weight, labels, divisor)
        return ChunkedOutput(loss=loss, logits=None, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            token_loss=tokens, token_entropy=entropy, token_predictions=predictions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        labels = inputs.get("labels")
        mask = inputs.get("attention_mask")
        supported = (
            head_supported(base) and model.training and self.model.training
            # Accelerate enters autocast inside the model's wrapped forward,
            # after this trainer method. Validate it there, not before the call.
            and self.args.fp16 and not self.args.bf16
            and self.label_smoother is None and self.compute_loss_func is None
            and self.num_virtual_tokens == 0 and not self.aux_loss_enabled
            and not self.args.use_liger_kernel and self.args.loss_type == "nll"
            and not self.args.packing and not self.args.padding_free and not self.args.torch_compile
            and self.accelerator.num_processes == 1 and self.args.past_index < 0
            and isinstance(labels, torch.Tensor) and labels.is_cuda and labels.dtype == torch.long
            and labels.ndim == 2 and labels.shape[1] > 0
            and isinstance(mask, torch.Tensor) and mask.device == labels.device and mask.shape == labels.shape
            and "shift_labels" not in inputs
            and isinstance(inputs.get("logits_to_keep", 0), int) and inputs.get("logits_to_keep", 0) == 0
            and inputs.get("return_dict", True) is not False
        )
        if not supported:
            return original_compute(self, model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        active = _ACTIVE.set(True)
        try:
            # Keep HF's original loss normalization and scaling logic intact.
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            _ACTIVE.reset(active)
        with torch.no_grad():
            mask = inputs["attention_mask"]
            entropy = (outputs.token_entropy * mask).sum() / mask.sum()
            self._metrics["train"]["entropy"].append(self.accelerator.gather_for_metrics(entropy).mean().item())
            n_tokens = self.accelerator.gather_for_metrics(mask.sum()).sum().item()
            self._total_train_tokens += n_tokens
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
            supervised = labels[:, 1:] != -100
            correct = ((outputs.token_predictions[:, :-1] == labels[:, 1:]) & supervised).sum()
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(supervised.sum()).sum()
            accuracy = (correct.sum() / total).item() if total > 0 else 0.0
            self._metrics["train"]["mean_token_accuracy"].append(accuracy)
        # Existing Soup rewind protocol consumes actual per-token losses when
        # no dense logits are retained; this does not enable a Soup mode/flag.
        setattr(self, STEP_ATTR, SimpleNamespace(token_loss=outputs.token_loss))
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_chunk_head_patch = True
    modeling_qwen2.Qwen2ForCausalLM.forward = model_forward
    SFTTrainer.compute_loss = compute_loss
