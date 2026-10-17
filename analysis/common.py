"""Loading and plotting helpers shared by the B2/B3 analysis scripts."""

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("ENERGY_RESULTS_DIR", REPO / "results"))

# Categorical slots in fixed order (identity), validated reference palette; text stays in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_run(run_dir: Path) -> tuple[dict, dict, pd.DataFrame]:
    """run.json, its session (GPU, idle power) and measurements.jsonl as a DataFrame."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    session = json.loads((RESULTS / "sessions" / f"{run['session_id']}.json").read_text())
    rows = pd.read_json(run_dir / "measurements.jsonl", lines=True)
    rows["gpu"] = session["gpu_name"]
    rows["idle_W"] = session["idle"]["mean_W"]
    if "E_end_mJ" in rows:
        rows["dur_s"] = rows.t_end_s - rows.t_start_s
    return run, session, rows


def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    return pd.concat([load_run(Path(d))[2] for d in run_dirs], ignore_index=True)


def style(ax, xlabel: str, ylabel: str):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def write_md(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {path}")
