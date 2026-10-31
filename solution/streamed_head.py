"""Vocabulary-streamed frozen head; FP32 weights live on CPU, GEMMs on CUDA.

All classes and original CE/entropy/accuracy outputs are evaluated. No model
weight is quantized or trained differently. The vocabulary split changes GEMM
reduction order; full training quality must pass the original harness.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

VOCAB_CHUNK = 8192


@triton.jit
def head_stream_stats_kernel(X, Y, STATE, ROWS, START, WIDTH: tl.constexpr,
                              FIRST: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    delta = x - maximum
    p = tl.exp(delta)
    total = tl.sum(p, 0)
    moment = tl.sum(p * tl.where(col < WIDTH, delta, 0.), 0)
    prediction = START + tl.argmax(x, 0, tie_break_left=True)
    label = tl.load(Y + row)
    here = (label >= START) & (label < START + WIDTH)
    target = tl.load(X + row * WIDTH + label - START, here, other=0).to(tl.float32)
    if not FIRST:
        old_max = tl.load(STATE + row * 5)
        old_sum = tl.load(STATE + row * 5 + 1)
        old_moment = tl.load(STATE + row * 5 + 2)
        old_target = tl.load(STATE + row * 5 + 3)
        old_prediction = tl.load(STATE + row * 5 + 4)
        new_max = tl.maximum(old_max, maximum)
        old_delta, new_delta = old_max - new_max, maximum - new_max
        old_scale, new_scale = tl.exp(old_delta), tl.exp(new_delta)
        moment = old_scale * (old_moment + old_sum * old_delta) + new_scale * (moment + total * new_delta)
        total = old_scale * old_sum + new_scale * total
        prediction = tl.where(old_max >= maximum, old_prediction, prediction)
        maximum = new_max
        target = tl.where(here, target, old_target)
    tl.store(STATE + row * 5, maximum)
    tl.store(STATE + row * 5 + 1, total)
    tl.store(STATE + row * 5 + 2, moment)
    tl.store(STATE + row * 5 + 3, target)
    tl.store(STATE + row * 5 + 4, prediction)


@triton.jit
def head_stream_finish_kernel(STATE, Y, LOSS, ENTROPY, PREDICTIONS, NORM,
                              VOCAB: tl.constexpr):
    row = tl.program_id(0)
    maximum = tl.load(STATE + row * 5)
    total = tl.load(STATE + row * 5 + 1)
    moment = tl.load(STATE + row * 5 + 2)
    target = tl.load(STATE + row * 5 + 3)
    prediction = tl.load(STATE + row * 5 + 4).to(tl.int64)
    label = tl.load(Y + row)
    tl.device_assert((label == -100) | ((label >= 0) & (label < VOCAB)), "target out of bounds")
    log_sum = tl.log(total)
    tl.store(LOSS + row, tl.where(label != -100, (maximum - target) + log_sum, 0.))
    tl.store(ENTROPY + row, log_sum - moment / total)
    tl.store(PREDICTIONS + row, prediction)
    tl.store(NORM + row * 2, maximum)
    tl.store(NORM + row * 2 + 1, log_sum)


@triton.jit
def head_stream_gradient_kernel(X, Y, NORM, SCALE, DY, START, WIDTH: tl.constexpr,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=-float("inf")).to(tl.float32)
    label = tl.load(Y + row)
    maximum = tl.load(NORM + row * 2)
    log_sum = tl.load(NORM + row * 2 + 1)
    scale = tl.load(SCALE)
    grad = (tl.exp((x - maximum) - log_sum) - (START + col == label).to(tl.float32)) * scale
    tl.store(DY + row * WIDTH + col, grad, col < WIDTH)


def host_half_weight(weight):
    signature = (weight.data_ptr(), weight._version)
    cached = getattr(weight, "_soup_head_host_cache", None)
    if cached is None or cached[0] != signature:
        if weight.device.type != "cpu" or weight.requires_grad or weight.dtype != torch.float32:
            raise RuntimeError("Streamed head requires the unchanged frozen FP32 CPU weight")
        cached = (signature, weight.detach().half().pin_memory())
        weight._soup_head_host_cache = cached
    return cached[1]


class StreamedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor, attention_mask):
        batch, length, width = hidden.shape
        target = F.pad(labels, (0, 1), value=-100)[:, 1:].contiguous().flatten()
        denominator = torch.as_tensor((target != -100).sum() if divisor is None else divisor,
                                      device=hidden.device, dtype=torch.float32)
        needed = (attention_mask.flatten().ne(0) | target.ne(-100)) if attention_mask is not None else torch.ones_like(target, dtype=torch.bool)
        selected = needed.nonzero(as_tuple=False).flatten()
        xhalf = hidden.reshape(-1, width).index_select(0, selected).half()
        target = target.index_select(0, selected)
        rows, vocab = xhalf.shape[0], weight.shape[0]
        state = torch.empty((rows, 5), device=hidden.device, dtype=torch.float32)
        cpu_weight = host_half_weight(weight)
        for start in range(0, vocab, VOCAB_CHUNK):
            w = cpu_weight[start:start + VOCAB_CHUNK].to(hidden.device, non_blocking=True)
            logits = F.linear(xhalf, w)
            head_stream_stats_kernel[(rows,)](
                logits, target, state, rows, start, w.shape[0], start == 0,
                triton.next_power_of_2(w.shape[0]), num_warps=8)
            del logits, w
        losses = torch.empty(rows, device=hidden.device)
        entropy = torch.empty_like(losses)
        predictions = torch.empty(rows, device=hidden.device, dtype=torch.long)
        norm = torch.empty((rows, 2), device=hidden.device)
        head_stream_finish_kernel[(rows,)](state, target, losses, entropy, predictions, norm, vocab, debug=True)
        ctx.save_for_backward(xhalf, weight, target, norm, denominator, selected)
        ctx.shape = hidden.shape
        full_rows = batch * length
        def full(x):
            return x.new_zeros(full_rows).index_copy_(0, selected, x).view(batch, length)
        tokens, entropy, predictions = full(losses), full(entropy), full(predictions)
        ctx.mark_non_differentiable(tokens, entropy, predictions)
        return tokens.sum() / denominator, tokens, entropy, predictions

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _tokens, _entropy, _predictions):
        x, weight, target, norm, denominator, selected = ctx.saved_tensors
        supervised = target.ne(-100).nonzero(as_tuple=False).flatten()
        x, target, norm = (t.index_select(0, supervised) for t in (x, target, norm))
        positions = selected.index_select(0, supervised)
        full = torch.zeros((ctx.shape[0] * ctx.shape[1], ctx.shape[2]), device=x.device)
        if x.shape[0] == 0:
            return full.view(ctx.shape), None, None, None, None
        dx = torch.zeros_like(x, dtype=torch.float32)
        scale = grad_loss.float() / denominator
        cpu_weight = host_half_weight(weight)
        for start in range(0, weight.shape[0], VOCAB_CHUNK):
            w = cpu_weight[start:start + VOCAB_CHUNK].to(x.device, non_blocking=True)
            logits = F.linear(x, w)
            dy = torch.empty_like(logits)
            head_stream_gradient_kernel[(x.shape[0], triton.cdiv(w.shape[0], 1024))](
                logits, target, norm, scale, dy, start, w.shape[0], 1024, num_warps=4)
            del logits
            # Each tile accumulates in FP32, then the complete input gradient
            # rounds once to Half, matching the original input-cast boundary.
            with torch.autocast("cuda", enabled=False):
                dx.add_(torch.mm(dy, w, out_dtype=torch.float32))
            del dy, w
        full.index_copy_(0, positions, dx.half().float())
        return full.view(ctx.shape), None, None, None, None
