"""Result records. Layout (see results/SCHEMA.md):

results/sessions/<session_id>.json              one per sitting at a GPU: idle power, versions, GPU info
results/runs/<experiment>/<run_id>/run.json     what was run: config, model, git commit, command
results/runs/<experiment>/<run_id>/measurements.jsonl   one raw row per trial, appended as it is measured
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SCHEMA_VERSION = 1


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ["torch", "transformers", "nvidia-ml-py", "zeus", "datasets", "math-verify"]:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


class Recorder:
    def __init__(self, experiment: str, config: dict, model: dict, meter, idle: dict, resume_dir: str | None = None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        host = socket.gethostname()
        chash = config_hash(config)
        gpu_tag = f"gpu{meter.nvml_index}"

        session_id = f"{stamp}_{host}_{gpu_tag}"
        session = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "host": host,
            **meter.info(),
            "idle": idle,
            "other_processes_at_start": meter.other_processes(),
            "versions": versions(),
        }
        (RESULTS / "sessions").mkdir(parents=True, exist_ok=True)
        (RESULTS / "sessions" / f"{session_id}.json").write_text(json.dumps(session, indent=2))

        if resume_dir:
            self.dir = Path(resume_dir)
            run = json.loads((self.dir / "run.json").read_text())
            if run["config_hash"] != chash:
                sys.exit(f"Refusing to resume {self.dir}: config differs ({run['config_hash']} vs {chash})")
            run.setdefault("resumed_in_sessions", []).append(session_id)
            run["status"] = "running"
        else:
            run_id = f"{stamp}_{experiment}_{gpu_tag}_{chash}"
            self.dir = RESULTS / "runs" / experiment / run_id
            self.dir.mkdir(parents=True)
            run = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "session_id": session_id,
                "experiment": experiment,
                "git_commit": _git("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                "command": " ".join([sys.executable, *sys.argv]),
                "cwd": os.getcwd(),
                "model": model,
                "config": config,
                "config_hash": chash,
                "status": "running",
                "started": stamp,
            }
        self.run = run
        self.session_id = session_id
        self._write_run()
        self._jsonl = open(self.dir / "measurements.jsonl", "a")

    def _write_run(self):
        (self.dir / "run.json").write_text(json.dumps(self.run, indent=2))

    def done_keys(self, key_fields: list[str]) -> set[tuple]:
        """Keys already measured in this run (for resume)."""
        path = self.dir / "measurements.jsonl"
        if not path.exists():
            return set()
        with open(path) as f:
            return {tuple(json.loads(line)[k] for k in key_fields) for line in f if line.strip()}

    def write(self, row: dict):
        row = {"run_id": self.run["run_id"], "session_id": self.session_id, **row}
        self._jsonl.write(json.dumps(row) + "\n")
        self._jsonl.flush()
        os.fsync(self._jsonl.fileno())

    def finish(self, status: str = "complete"):
        self.run["status"] = status
        self.run["finished"] = time.strftime("%Y%m%d-%H%M%S")
        self._write_run()
        self._jsonl.close()
