from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root containing src/asr_lab."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "asr_lab").is_dir():
            return candidate
    raise RuntimeError("Could not find project root containing src/asr_lab")


def find_local_root(start: Path | None = None) -> Path:
    """Compatibility alias for notebooks that still name the repo LOCAL_ROOT."""
    return find_project_root(start)


def find_main_root(start: Path | None = None) -> Path:
    """Compatibility alias: ASR_local now carries the asr_lab package itself."""
    return find_project_root(start)


def add_src(path: Path) -> None:
    src = str((path / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def bootstrap(start: Path | None = None) -> tuple[Path, Path]:
    """Add project src to sys.path, returning (local_root, main_root)."""
    root = find_project_root(start)
    add_src(root)
    return root, root


def notebook_work_root(main_root: Path, run_family: str) -> Path:
    """Default local artifact root for notebook-driven runs."""
    return main_root / "artifacts" / "notebook_runs" / run_family
