"""Local Kaggle CLI orchestration. Authentication stays in Kaggle's user config."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "diassamenov/soup-kernel-lab-auto"
STAGING = ROOT / "artifacts" / "kaggle_auto" / "source"


def cli_path():
    if os.environ.get("KAGGLE_CLI"):
        return os.environ["KAGGLE_CLI"]
    found = shutil.which("kaggle")
    if found:
        return found
    import site
    user_scripts = Path(site.getuserbase()) / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
    if user_scripts.exists():
        return str(user_scripts)
    raise SystemExit("Kaggle CLI not found. Set KAGGLE_CLI to its executable path.")


def call(*args):
    subprocess.run([cli_path(), *args], check=True)


def prepare(commit, repeats=2):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("Use the full 40-character Git commit SHA")
    STAGING.mkdir(parents=True, exist_ok=True)
    job = (ROOT / "automation" / "kaggle_job.py").read_text(encoding="utf-8")
    config = {"commit": commit, "repo_url": "https://github.com/samenov01/soup-kernel-lab.git", "repeats": repeats}
    # Embed config into the submitted notebook: no external input attachment needed.
    job = job.replace('CONFIG = json.loads(Path(__file__).with_name("run-config.json").read_text())',
                      "CONFIG = " + repr(config))
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": job.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (STAGING / "experiment.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {
        "id": KERNEL, "title": "Soup Kernel Lab Auto", "code_file": "experiment.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (STAGING / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {KERNEL} for {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "submit"):
        command = sub.add_parser(name)
        command.add_argument("--commit", required=True)
        command.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    sub.add_parser("status")
    command = sub.add_parser("download")
    command.add_argument("--out", required=True)
    command.add_argument("--version", type=int)
    args = parser.parse_args()
    if args.action in ("prepare", "submit"):
        prepare(args.commit, args.repeats)
        if args.action == "submit":
            call("kernels", "push", "-p", str(STAGING), "--accelerator", "NvidiaTeslaT4", "--timeout", "3600")
    elif args.action == "status":
        call("kernels", "status", KERNEL)
    elif args.action == "download":
        out = Path(args.out).resolve()
        if out.exists() and any(out.iterdir()):
            raise SystemExit("Choose a new output directory to preserve previous runs")
        out.mkdir(parents=True, exist_ok=True)
        ref = KERNEL + (f"/{args.version}" if args.version else "")
        call("kernels", "output", ref, "-p", str(out), "--file-pattern", r"^(kernel_results\.zip|run_status\.json)$")


if __name__ == "__main__":
    main()
