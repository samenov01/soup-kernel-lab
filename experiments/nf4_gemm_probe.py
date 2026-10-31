"""Independent fused NF4 matmul performance probe; not part of submission kernels.

Benchmarks are diagnostic only. No patch of the measured training path is made.
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def probe_nf4_gemm(A, PACKED, QABS, SUPERABS, ABSCODE, OFFSET, CODEBOOK, OUT,
                    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    lane_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    offset = tl.load(OFFSET)
    c0 = tl.load(CODEBOOK + 0)
    c1 = tl.load(CODEBOOK + 1)
    c2 = tl.load(CODEBOOK + 2)
    c3 = tl.load(CODEBOOK + 3)
    c4 = tl.load(CODEBOOK + 4)
    c5 = tl.load(CODEBOOK + 5)
    c6 = tl.load(CODEBOOK + 6)
    c7 = tl.load(CODEBOOK + 7)
    c8 = tl.load(CODEBOOK + 8)
    c9 = tl.load(CODEBOOK + 9)
    c10 = tl.load(CODEBOOK + 10)
    c11 = tl.load(CODEBOOK + 11)
    c12 = tl.load(CODEBOOK + 12)
    c13 = tl.load(CODEBOOK + 13)
    c14 = tl.load(CODEBOOK + 14)
    c15 = tl.load(CODEBOOK + 15)
    for step in range(tl.cdiv(K, BK)):
        inner = step * BK + lane_k
        a = tl.load(A + rows[:, None] * K + inner[None, :],
                    (rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        # Decode as [N,K] so packed reads are contiguous along K.
        weight_index = cols[:, None] * K + inner[None, :]
        valid = (cols[:, None] < N) & (inner[None, :] < K)
        packed = tl.load(PACKED + weight_index // 2, valid, other=0).to(tl.int32)
        code = tl.where(weight_index % 2 == 0, packed >> 4, packed & 15)
        # BK divides GROUP: one scale per output column, not per tile element.
        group = (cols * K + step * BK) // GROUP
        scale_code = tl.load(QABS + group, cols < N, other=0).to(tl.int32)
        scale_value = tl.load(ABSCODE + scale_code)
        scale_max = tl.load(SUPERABS + group // 256, cols < N, other=0.0)
        scale = scale_value * scale_max + offset
        value = tl.where(code < 8, tl.where(code < 4, tl.where(code < 2, tl.where(code < 1, c0, c1), tl.where(code < 3, c2, c3)), tl.where(code < 6, tl.where(code < 5, c4, c5), tl.where(code < 7, c6, c7))), tl.where(code < 12, tl.where(code < 10, tl.where(code < 9, c8, c9), tl.where(code < 11, c10, c11)), tl.where(code < 14, tl.where(code < 13, c12, c13), tl.where(code < 15, c14, c15))))
        b = tl.where(valid, value * scale[:, None], 0.0).to(tl.float16)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def timed(fn, repetitions=15):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repetitions


def main():
    import bitsandbytes as bnb
    from solution.nf4 import patch_nf4_dequantization
    patch_nf4_dequantization()
    torch.manual_seed(21)
    rows = []
    for m, n, k in [(512, 8960, 1536), (2048, 8960, 1536), (512, 1536, 8960), (2048, 1536, 8960)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
        packed, state = bnb.functional.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)
        del weight
        reference = lambda: bnb.matmul_4bit(a, packed, state)
        expected = reference()
        baseline_ms = timed(reference)
        for bm, bn, bk in [(32, 64, 64), (64, 64, 32), (64, 128, 32)]:
            out = torch.empty(m, n, device="cuda", dtype=torch.float16)
            record = {"implementation": "vector-scale-coalesced", "shape": [m, n, k], "tile": [bm, bn, bk], "baseline_ms": baseline_ms}
            compiled = []

            def candidate():
                kernel = probe_nf4_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                    a, packed, state.absmax, state.state2.absmax, state.state2.code,
                    state.offset, state.code, out, m, n, k, state.blocksize, bm, bn, bk,
                    num_warps=4, num_stages=2, enable_fp_fusion=False)
                if not compiled:
                    compiled.append(kernel)
                return out

            try:
                actual = candidate()
                torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
                candidate_ms = timed(candidate)
                record.update(ok=True, candidate_ms=candidate_ms, speedup=baseline_ms / candidate_ms,
                              max_abs_error=(actual.float() - expected.float()).abs().max().item())
                record.update(registers=getattr(compiled[0], "n_regs", None),
                              spills=getattr(compiled[0], "n_spills", None),
                              shared_bytes=getattr(getattr(compiled[0], "metadata", None), "shared", None))
            except Exception as exc:
                record.update(ok=False, error=str(exc)[:1000])
            rows.append(record)
            print(json.dumps(record), flush=True)
        del a, packed, state, expected, out
    Path("nf4_gemm_probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
