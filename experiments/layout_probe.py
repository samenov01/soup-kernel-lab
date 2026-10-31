"""Independent T4 GEMM layout probe; never part of scored measurements."""

import json
from pathlib import Path
import torch
import subprocess
import sys


def timing(fn, repeats=25):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    torch.manual_seed(18)
    result = {"device": torch.cuda.get_device_name(), "diagnostic_only": True, "rows": []}
    for rows in (512, 1024, 1536):
        for inner, output in ((1536, 8960), (17920, 1536), (1792, 1536), (1536, 1536)):
            a = torch.randn(rows, inner, device="cuda", dtype=torch.float16) * 0.01
            w = torch.randn(inner, output, device="cuda", dtype=torch.float16) * 0.01
            wt = w.T.contiguous()
            ref = torch.matmul(a, w)
            alt = torch.nn.functional.linear(a, wt)
            diff = (ref.float() - alt.float()).abs().max().item()
            record = {"m": rows, "k": inner, "n": output, "max_abs_diff": diff,
                      "nn_ms": timing(lambda: torch.matmul(a, w)),
                      "tn_ms": timing(lambda: torch.nn.functional.linear(a, wt))}
            result["rows"].append(record)
            print(json.dumps(record), flush=True)
            del a, w, wt, ref, alt
    command = [sys.executable, "-X", "importtime", "-c",
               "import torch; torch.empty(1,device='cuda'); import transformers; import solution; solution.apply()"]
    imports = subprocess.run(command, text=True, capture_output=True, check=True)
    timings = []
    for line in imports.stderr.splitlines():
        if line.startswith("import time:"):
            try:
                own, cumulative, name = line[len("import time:"):].split("|", 2)
                timings.append({"self_us": int(own), "total_us": int(cumulative), "module": name.strip()})
            except ValueError: pass
    result["imports_by_cumulative"] = sorted(timings, key=lambda x: -x["total_us"])[:30]
    result["imports_by_self"] = sorted(timings, key=lambda x: -x["self_us"])[:30]
    print(json.dumps({"imports": result["imports_by_cumulative"]}), flush=True)
    Path("nf4_gemm_probe.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
