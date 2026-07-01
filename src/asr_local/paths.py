from __future__ import annotations

import sys
from pathlib import Path


def find_local_root(start: Path | None = None) -> Path:
    """Return the ASR_local root containing src/asr_local."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "asr_local").is_dir():
            return candidate
    raise RuntimeError("Could not find ASR_local root containing src/asr_local")


def find_main_root(start: Path | None = None) -> Path:
    """Return the sibling/main ASR repo containing src/asr_lab."""
    local_root = find_local_root(start)
    candidates = [
        local_root.parent / "ASR",
        Path(r"C:\Users\Admin\OneDrive\Documents\Documents\ASR"),
        Path.cwd(),
        *Path.cwd().parents,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "asr_lab").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find main ASR repo containing src/asr_lab")


def add_src(path: Path) -> None:
    src = str((path / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def bootstrap(start: Path | None = None) -> tuple[Path, Path]:
    """Add ASR_local/src and main ASR/src to sys.path, returning both roots."""
    local_root = find_local_root(start)
    main_root = find_main_root(local_root)
    add_src(local_root)
    add_src(main_root)
    return local_root, main_root


def notebook_work_root(main_root: Path, run_family: str) -> Path:
    """Default local artifact root for notebook-driven runs."""
    return main_root / "artifacts" / "notebook_runs" / run_family

