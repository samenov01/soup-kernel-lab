"""Archive actual run results/logs for sharing, never adapters or caches.

Kaggle cell: %run /kaggle/working/repo/automation/archive_results.py
Does not start training, run checks, or change existing measurement files.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


def archive_results(work, repo, destination):
    work, repo, destination = work.resolve(), repo.resolve(), destination.resolve()
    roots = {p for parent in (work, repo) for p in parent.glob("runs*") if p.is_dir()}
    roots.add(work / "automation_logs")
    files = set()
    for root in sorted(roots):
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d != "adapter" and "cache" not in d.lower()]
            for name in names:
                p = Path(folder) / name
                if p.suffix.lower() in {".json", ".jsonl", ".log", ".yaml", ".yml", ".txt"}:
                    resolved = p.resolve()
                    if resolved.is_relative_to(work) and p.is_file():
                        files.add(p)
    status = work / "run_status.json"
    if status.is_file():
        files.add(status)
    if not files:
        raise RuntimeError("No run results/logs found. Finish the training cell first; no empty ZIP was created.")

    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": len(files)}
    try:
        provenance["checkout_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        h = hashlib.sha256()
        # Same ordering and path prefix as the official solution checksum.
        for p in sorted((repo / "solution").rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            h.update(p.relative_to(repo / "solution").as_posix().encode())
            h.update(p.read_bytes())
        provenance["checkout_solution_sha256"] = h.hexdigest()
        provenance["tracked_changes"] = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        provenance["git_metadata_error"] = type(exc).__name__
    # This is metadata about the checkout at export time, not a claim that
    # every older runs* directory was produced by that checkout.
    provenance["run_checks"] = []
    for p in sorted(files):
        if p.name != "checks.json":
            continue
        try:
            checks = json.loads(p.read_text(encoding="utf-8"))
            result = p.with_name("result.json")
            provenance["run_checks"].append({
                "directory": p.parent.relative_to(work).as_posix(),
                "recorded_solution_sha256": checks.get("solution_sha256"),
                "matches_checkout": checks.get("solution_sha256") == provenance.get("checkout_solution_sha256"),
                "result_hash_matches": result.is_file() and hashlib.sha256(result.read_bytes()).hexdigest()
                    == checks.get("result_sha256"),
            })
        except (OSError, ValueError) as exc:
            provenance["run_checks"].append({"path": str(p.relative_to(work)), "error": type(exc).__name__})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(work).as_posix())
        z.writestr("export_provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2))
    with ZipFile(destination) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP verification failed: {bad}")
    print(f"Saved {destination}: {len(files)} result/log files, {destination.stat().st_size:,} bytes")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive_results(args.work, args.repo or args.work / "repo", args.output or args.work / "kernel_results.zip")
