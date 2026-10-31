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
    for step in range(tl.cdiv(K, BK)):
        inner = step * BK + lane_k
        a = tl.load(A + rows[:, None] * K + inner[None, :],
                    (rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        weight_index = cols[None, :] * K + inner[:, None]
        valid = (cols[None, :] < N) & (inner[:, None] < K)
        packed = tl.load(PACKED + weight_index // 2, valid, other=0).to(tl.int32)
        code = tl.where(weight_index % 2 == 0, packed >> 4, packed & 15)
        block = weight_index // GROUP
        scale_code = tl.load(QABS + block, valid, other=0).to(tl.int32)
        scale_value = tl.load(ABSCODE + scale_code)
        scale_max = tl.load(SUPERABS + block // 256, valid, other=0.0)
        scale = scale_value * scale_max + offset
        value = tl.load(CODEBOOK + code)
        b = (value * scale).to(tl.float16)
        b = tl.where(valid, b, 0).to(tl.float16)
        acc = tl.dot(a, b, acc)
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
            record = {"shape": [m, n, k], "tile": [bm, bn, bk], "baseline_ms": baseline_ms}

            def candidate():
                probe_nf4_gemm[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                    a, packed, state.absmax, state.state2.absmax, state.state2.code,
                    state.offset, state.code, out, m, n, k, state.blocksize, bm, bn, bk,
                    num_warps=4, num_stages=2, enable_fp_fusion=False)
                return out

            try:
                actual = candidate()
                torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
                candidate_ms = timed(candidate)
                record.update(ok=True, candidate_ms=candidate_ms, speedup=baseline_ms / candidate_ms,
                              max_abs_error=(actual.float() - expected.float()).abs().max().item())
            except Exception as exc:
                record.update(ok=False, error=str(exc)[:1000])
            rows.append(record)
            print(json.dumps(record), flush=True)
        del a, packed, state, expected, out
    Path("nf4_gemm_probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
