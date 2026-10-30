"""Tiled vocabulary reductions, written for the challenge (no copied kernels).

Only the fp32, CUDA, dense-class-index path is accelerated. Other inputs retain
the original library implementation. Full logits remain available to TRL.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

TILE = 4096


@triton.jit
def vocab_partial_kernel(X, M, S, W, V: tl.constexpr, STRIDE: tl.constexpr,
                         PARTS: tl.constexpr, ENTROPY: tl.constexpr,
                         BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + col, col < V, other=-float("inf"))
    maximum = tl.max(x, 0)
    delta = x - maximum
    if not ENTROPY:
        # A whole tile of masked classes contributes zero, not exp(-inf - -inf).
        # Keep entropy's original log_softmax semantics for non-finite logits.
        delta = tl.where(maximum == -float("inf"), -float("inf"), delta)
    p = tl.exp(delta)
    slot = row * PARTS + part
    tl.store(M + slot, maximum)
    tl.store(S + slot, tl.sum(p, 0))
    if ENTROPY:
        # 0 * -inf from padding must not contaminate the moment.
        moment = tl.sum(p * tl.where(col < V, delta, 0.0), 0)
        tl.store(W + slot, moment)


@triton.jit
def vocab_ce_finish_kernel(X, Y, M, S, NORM, LOSS, V: tl.constexpr,
                           STRIDE: tl.constexpr, PARTS: tl.constexpr,
                           IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    maxima = tl.load(M + row * PARTS + part, part < PARTS, other=-float("inf"))
    sums = tl.load(S + row * PARTS + part, part < PARTS, other=0.0)
    maximum = tl.max(maxima, 0)
    log_sum = tl.log(tl.sum(sums * tl.exp(maxima - maximum), 0))
    label = tl.load(Y + row)
    valid = label != IGNORE
    tl.device_assert((~valid) | ((label >= 0) & (label < V)), "target out of bounds")
    target = tl.load(X + row * STRIDE + label, valid & (label >= 0) & (label < V), other=0.0)
    # Do not round maximum + log_sum into one fp32 number: a large common
    # offset would destroy the small normalization term used by backward.
    tl.store(NORM + 2 * row, maximum)
    tl.store(NORM + 2 * row + 1, log_sum)
    tl.store(LOSS + row, tl.where(valid, (maximum - target) + log_sum, 0.0))


@triton.jit
def vocab_ce_backward_kernel(X, Y, NORM, DLOSS, DX, V: tl.constexpr,
                             STRIDE: tl.constexpr, IGNORE: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    label = tl.load(Y + row)
    x = tl.load(X + row * STRIDE + col, col < V, other=-float("inf"))
    maximum = tl.load(NORM + 2 * row)
    log_sum = tl.load(NORM + 2 * row + 1)
    upstream = tl.load(DLOSS + row)
    grad = (tl.exp((x - maximum) - log_sum) - (col == label).to(tl.float32)) * upstream
    tl.store(DX + row * V + col, tl.where(label != IGNORE, grad, 0.0), col < V)


@triton.jit
def vocab_entropy_finish_kernel(M, S, W, OUT, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    maxima = tl.load(M + row * PARTS + part, part < PARTS, other=-float("inf"))
    sums = tl.load(S + row * PARTS + part, part < PARTS, other=0.0)
    moments = tl.load(W + row * PARTS + part, part < PARTS, other=0.0)
    maximum = tl.max(maxima, 0)
    delta = tl.where(part < PARTS, maxima - maximum, 0.0)
    scale = tl.exp(delta)
    denominator = tl.sum(scale * sums, 0)
    centered_moment = tl.sum(scale * (moments + sums * delta), 0)
    tl.store(OUT + row, tl.log(denominator) - centered_moment / denominator)


def _partials(x, entropy=False):
    rows, vocab = x.shape
    parts = triton.cdiv(vocab, TILE)
    maxima = torch.empty((rows, parts), device=x.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    moments = torch.empty_like(maxima) if entropy else sums
    vocab_partial_kernel[(rows, parts)](
        x, maxima, sums, moments, vocab, x.stride(0), parts, entropy, TILE,
        num_warps=4,
    )
    return maxima, sums, moments, parts


class _TokenCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, labels, ignore_index):
        labels = labels.contiguous()
        maxima, sums, _, parts = _partials(x)
        norm = torch.empty((x.shape[0], 2), device=x.device, dtype=torch.float32)
        loss = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        vocab_ce_finish_kernel[(x.shape[0],)](
            x, labels, maxima, sums, norm, loss, x.shape[1], x.stride(0),
            parts, ignore_index, triton.next_power_of_2(parts), num_warps=4,
            debug=True,
        )
        ctx.save_for_backward(x, labels, norm)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, labels, norm = ctx.saved_tensors
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        vocab_ce_backward_kernel[(x.shape[0], triton.cdiv(x.shape[1], TILE))](
            x, labels, norm, grad.contiguous(), dx, x.shape[1], x.stride(0),
            ctx.ignore_index, TILE, num_warps=4,
        )
        return dx, None, None


def supported(x, labels=None):
    ok = (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32
          and x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0 and x.stride(1) == 1)
    if labels is not None:
        ok = (ok and isinstance(labels, torch.Tensor) and labels.device == x.device
              and labels.dtype == torch.long and labels.shape == x.shape[:1])
    return ok


def token_cross_entropy(x, labels, ignore_index=-100):
    """Unreduced fp32 CE; first-order derivatives, no mutation of input logits."""
    if not supported(x, labels):
        return torch.nn.functional.cross_entropy(x, labels, reduction="none", ignore_index=ignore_index)
    with torch.cuda.device(x.device):
        return _TokenCrossEntropy.apply(x, labels, ignore_index)


def entropy_forward(logits):
    """No-grad entropy for contiguous fp32 logits; callers guard this contract."""
    x = logits.view(-1, logits.shape[-1])
    with torch.cuda.device(x.device):
        maxima, sums, moments, parts = _partials(x, entropy=True)
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        vocab_entropy_finish_kernel[(x.shape[0],)](
            maxima, sums, moments, out, parts, triton.next_power_of_2(parts), num_warps=4,
        )
    return out.view(logits.shape[:-1])


def patch_vocab_reductions():
    """Patch three concrete call sites, without changing torch's global CE."""
    from transformers.loss import loss_utils
    from trl.trainer import sft_trainer
    from soup_cli.trainer import rewind_hf

    if getattr(loss_utils.fixed_cross_entropy, "_soup_vocab_patch", False):
        return
    original_ce = loss_utils.fixed_cross_entropy
    original_entropy = sft_trainer.entropy_from_logits
    original_rows = rewind_hf.row_losses

    def fixed_ce(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
        if not supported(source, target):
            return original_ce(source, target, num_items_in_batch, ignore_index, **kwargs)
        losses = token_cross_entropy(source, target, ignore_index)
        divisor = (target != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
        if torch.is_tensor(divisor):
            divisor = divisor.to(losses.device)
        return losses.sum() / divisor

    def entropy(logits, chunk_size=128):
        if (not isinstance(logits, torch.Tensor) or logits.ndim < 2
                or not logits.is_contiguous() or logits.numel() == 0
                or logits.dtype != torch.float32 or not logits.is_cuda
                or (torch.is_grad_enabled() and logits.requires_grad)):
            return original_entropy(logits, chunk_size)
        return entropy_forward(logits)

    def row_losses(logits, labels):
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                or not isinstance(labels, torch.Tensor) or labels.shape != logits.shape[:2]
                or logits.shape[0] == 0 or logits.shape[1] < 2
                or not supported(logits[0, :-1], labels[0, 1:])):
            return original_rows(logits, labels)
        means, counts = [], []
        with torch.no_grad():
            for row in range(logits.shape[0]):
                target = labels[row, 1:]
                losses = token_cross_entropy(logits[row, :-1].detach(), target)
                mask = target != -100
                count = mask.sum()
                means.append((losses * mask).sum() / count.clamp(min=1))
                counts.append(count)
            # Preserve float32 row reduction and integer counts, but transfer
            # whole lists rather than synchronizing twice for each example.
            return torch.stack(means).tolist(), torch.stack(counts).tolist()

    fixed_ce._soup_vocab_patch = True
    loss_utils.fixed_cross_entropy = fixed_ce
    sft_trainer.entropy_from_logits = entropy
    rewind_hf.row_losses = row_losses
