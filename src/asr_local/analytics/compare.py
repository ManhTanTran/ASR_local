from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def load_results(main_root: Path, run_id: str) -> dict | None:
    path = Path(main_root) / "artifacts" / "runs" / run_id / "results.json"
    if not path.exists():
        print("Missing results:", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def result_table(*results: dict | None) -> pd.DataFrame:
    return pd.DataFrame([row for row in results if row])


def artifact_manifest(main_root: Path, run_id: str) -> pd.DataFrame:
    run_dir = Path(main_root) / "artifacts" / "runs" / run_id
    rows = []
    if not run_dir.exists():
        return pd.DataFrame([{"run_id": run_id, "missing": str(run_dir)}])
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "run_id": run_id,
                    "path": str(path.relative_to(run_dir)),
                    "size_mb": round(path.stat().st_size / 1_000_000, 3),
                }
            )
    return pd.DataFrame(rows)


def _uv_module(module: str, *args: str) -> list[str]:
    return ["uv", "run", "python", "-m", module, *args]


def run_main_report_commands(main_root: Path, base_run_id: str, candidate_run_id: str) -> None:
    commands = [
        _uv_module("asr_lab.analytics.report", "--run-id", base_run_id),
        _uv_module("asr_lab.analytics.report", "--run-id", candidate_run_id),
        _uv_module("asr_lab.analytics.compare", "--base", base_run_id, "--cand", candidate_run_id),
        _uv_module("asr_lab.registry.build_scoreboard", "--print"),
    ]
    for command in commands:
        subprocess.run(command, cwd=main_root, text=True)

