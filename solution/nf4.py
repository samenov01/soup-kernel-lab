"""NF4 unpack + block scaling, implemented independently for this challenge.

The 16-value codebook comes from bitsandbytes' format definition. Its original
CUDA backend still owns nested-scale decoding, allocation and matrix products.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def nf4_unpack_scale_kernel(PACKED, SCALES, CODEBOOK, OUT, N,
                            GROUP: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = index < N
    packed = tl.load(PACKED + index // 2, live, other=0).to(tl.int32)
    # bitsandbytes packs the first element into the high nibble.
    code = tl.where(index % 2 == 0, packed >> 4, packed & 15)
    value = tl.load(CODEBOOK + code)
    scale = tl.load(SCALES + index // GROUP, live, other=0.0)
    tl.store(OUT + index, value * scale, live)


def patch_nf4_dequantization():
    """Replace only the pinned CUDA backend's NF4 unpacking primitive.

    Both gemm fallback and autograd input-gradient paths call this primitive.
    Unsupported types/layouts continue through the original implementation.
    """
    from bitsandbytes import functional as F
    from bitsandbytes.backends.cuda import ops

    original = ops._dequantize_4bit_impl
    if getattr(original, "_soup_nf4_patch", False):
        return
    codebooks = {}

    def dequantize(A, absmax, blocksize, quant_type, dtype, out):
        supported = (
            quant_type == "nf4" and dtype in (torch.float16, torch.float32)
            and A.is_cuda and A.dtype == torch.uint8 and A.is_contiguous()
            and out.device == A.device and out.dtype == dtype and out.is_contiguous()
            and absmax.device == A.device and absmax.dtype == torch.float32
            and absmax.is_contiguous() and blocksize in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            and out.numel() > 0 and A.numel() * 2 >= out.numel()
            and absmax.numel() >= triton.cdiv(out.numel(), blocksize)
        )
        if not supported:
            return original(A, absmax, blocksize, quant_type, dtype, out=out)
        with torch.cuda.device(A.device):
            if A.device not in codebooks:
                codebooks[A.device] = F.get_4bit_type("nf4", device=A.device)
            nf4_unpack_scale_kernel[(triton.cdiv(out.numel(), 1024),)](
                A, absmax, codebooks[A.device], out, out.numel(), blocksize, 1024,
                num_warps=4,
            )
        # The backend primitive writes out and returns None.

    dequantize._soup_nf4_patch = True
    ops._dequantize_4bit_impl = dequantize
