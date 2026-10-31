"""Executed by a private Kaggle kernel, outside the unchanged challenge harness."""

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZipFile, ZIP_DEFLATED


CONFIG = json.loads(Path(__file__).with_name("run-config.json").read_text())
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
RESULTS = WORK / "runs"
LOGS = WORK / "automation_logs"
LOGS.mkdir(exist_ok=True)
status = {"commit": CONFIG["commit"], "stage": "initializing", "ok": False}


def run(stage, command, cwd=WORK, env=None):
    status["stage"] = stage
    print(f"[automation] {stage}", flush=True)
    log = LOGS / f"{stage}.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-16000:], flush=True)
        raise RuntimeError(f"{stage} failed (exit {result.returncode}); see {log.name}")
    print(log.read_text(encoding="utf-8", errors="replace")[-2000:], flush=True)


try:
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    status["gpus"] = gpus.splitlines()
    assert gpus.splitlines() and "T4" in gpus.splitlines()[0], "Expected T4 GPU 0"
    run("dependencies", [sys.executable, "-m", "pip", "install", "-q",
        "soup-cli[train]==0.77.0", "transformers==5.16.1", "peft==0.20.0",
        "trl==0.29.1", "bitsandbytes==0.50.2", "accelerate==1.14.0", "pytest"])
    if REPO.exists():
        raise RuntimeError("repo already exists; start a fresh saved run to preserve previous files")
    run("clone", ["git", "clone", "-q", CONFIG["repo_url"], str(REPO)])
    run("checkout", ["git", "checkout", "-q", CONFIG["commit"]], cwd=REPO)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert actual == CONFIG["commit"], (actual, CONFIG["commit"])
    if CONFIG.get("probe"):
        status["diagnostic_only"] = True
        run("probe", [sys.executable, CONFIG["probe"]], cwd=REPO,
            env=dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="0"))
        import shutil
        shutil.copyfile(REPO / "nf4_gemm_probe.json", WORK / "probe_results.json")
    else:
        run("data", [sys.executable, "task/prepare_data.py"], cwd=REPO)
        run("pytest_solution", [sys.executable, "-m", "pytest", "solution", "-q"], cwd=REPO)
        run("pytest_regression", [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=REPO, env=dict(os.environ, SOUP_APPLY_SOLUTION="1"))
        run("training", [sys.executable, "harness/run_task.py", "--repeats", str(CONFIG.get("repeats", 2)),
            "--profile", "--out", str(RESULTS)], cwd=REPO)
        run("checks", [sys.executable, "harness/check_rules.py", "--result",
            str(RESULTS / "result.json")], cwd=REPO)
        run("score", [sys.executable, "harness/score.py", str(RESULTS / "result.json")], cwd=REPO)
    status.update(ok=True, stage="complete")
except Exception as exc:
    status["error"] = str(exc)
    (LOGS / "exception.log").write_text(traceback.format_exc(), encoding="utf-8")
    raise
finally:
    (WORK / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    archive = WORK / "kernel_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        output.write(WORK / "run_status.json", "automation/run_status.json")
        if (WORK / "probe_results.json").exists():
            output.write(WORK / "probe_results.json", "automation/probe_results.json")
        for file in LOGS.glob("*.log"):
            output.write(file, "automation/" + file.name)
        if RESULTS.exists():
            for folder, dirs, names in os.walk(RESULTS):
                dirs[:] = [d for d in dirs if d not in {"adapter", "triton_cache", "inductor_cache"}]
                for name in names:
                    file = Path(folder) / name
                    if file.suffix in {".json", ".jsonl", ".log", ".yaml"}:
                        output.write(file, "runs/" + file.relative_to(RESULTS).as_posix())
    print(f"[automation] Saved {archive.name}, {archive.stat().st_size} bytes", flush=True)
