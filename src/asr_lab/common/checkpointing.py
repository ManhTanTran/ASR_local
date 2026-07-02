"""Reusable Lightning checkpoint helpers for Kaggle/interrupted runs."""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def checkpoint_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _existing_ckpts(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.is_file() and path.suffix == ".ckpt"]


def _latest(paths: Iterable[Path]) -> Path | None:
    candidates = _existing_ckpts(paths)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _glob_existing(pattern: str) -> list[Path]:
    return _existing_ckpts(Path(match) for match in glob.glob(pattern, recursive=True))


def _split_search_dirs(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def find_latest_checkpoint(
    run_dir: str | Path,
    *,
    run_id: str = "",
    explicit: str | Path | None = None,
    search_inputs: bool = True,
    search_dirs: Iterable[str | Path] | None = None,
) -> Path | None:
    """Find the newest checkpoint for this run.

    Search order:
    1. explicit file/glob, if provided;
    2. local ``<run_dir>/checkpoints/*.ckpt``;
    3. user supplied search dirs and ``ASR_CHECKPOINT_SEARCH_DIRS``;
    4. Kaggle input outputs matching ``runs/<run_id>/checkpoints/*.ckpt``.
    """
    if explicit:
        explicit_text = str(explicit)
        explicit_path = Path(explicit_text).expanduser()
        if explicit_path.is_file():
            return explicit_path
        match = _latest(_glob_existing(explicit_text))
        if match:
            return match
        raise FileNotFoundError(f"No checkpoint matched: {explicit_text}")

    run_dir = Path(run_dir)
    local = _latest((run_dir / "checkpoints").glob("*.ckpt"))
    if local:
        return local

    roots: list[Path] = []
    if search_dirs:
        roots.extend(Path(path).expanduser() for path in search_dirs)
    roots.extend(_split_search_dirs(os.environ.get("ASR_CHECKPOINT_SEARCH_DIRS")))
    for root in roots:
        if root.exists():
            match = _latest(root.glob("**/*.ckpt"))
            if match:
                return match

    if search_inputs and run_id:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            pattern = str(kaggle_input / "**" / "runs" / run_id / "checkpoints" / "*.ckpt")
            match = _latest(_glob_existing(pattern))
            if match:
                return match

    return None


def write_checkpoint_manifest(
    run_dir: str | Path,
    *,
    latest: str | Path | None = None,
    removed: list[str] | None = None,
) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    files = []
    for path in sorted(ckpt_dir.glob("*.ckpt"), key=lambda item: item.stat().st_mtime):
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1_000_000, 2),
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    payload = {
        "latest": str(latest) if latest else (str(files[-1]["path"]) if files else None),
        "files": files,
        "removed": removed or [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest = ckpt_dir / "checkpoint_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def make_periodic_checkpoint_callback(
    run_dir: str | Path,
    *,
    every_n_steps: int = 500,
    keep_last: int = 2,
    save_on_epoch_end: bool = True,
    save_on_exception: bool = True,
):
    """Return a Lightning callback that saves resumable checkpoints during training."""
    import lightning.pytorch as pl

    class PeriodicCheckpointCallback(pl.Callback):
        def __init__(self) -> None:
            self.ckpt_dir = checkpoint_dir(run_dir)
            self.every_n_steps = max(0, int(every_n_steps))
            self.keep_last = max(0, int(keep_last))
            self.save_on_epoch_end = bool(save_on_epoch_end)
            self.save_on_exception = bool(save_on_exception)
            self._last_saved_step = -1

        def _rotate(self) -> list[str]:
            if self.keep_last <= 0:
                return []
            files = sorted(self.ckpt_dir.glob("*.ckpt"), key=lambda item: item.stat().st_mtime)
            removed: list[str] = []
            while len(files) > self.keep_last:
                victim = files.pop(0)
                try:
                    victim.unlink()
                    removed.append(victim.name)
                except OSError as exc:
                    print(f"[checkpoint] could not remove old checkpoint {victim}: {exc}", flush=True)
            return removed

        def _save(self, trainer, reason: str, *, force: bool = False) -> None:
            if getattr(trainer, "sanity_checking", False):
                return
            if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
                return
            step = int(getattr(trainer, "global_step", 0) or 0)
            if step <= 0:
                return
            if not force and step == self._last_saved_step:
                return
            epoch = int(getattr(trainer, "current_epoch", 0) or 0)
            safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in reason)
            path = self.ckpt_dir / f"{safe_reason}-epoch{epoch:03d}-step{step:06d}.ckpt"
            print(f"[checkpoint] saving {path}", flush=True)
            trainer.save_checkpoint(str(path))
            self._last_saved_step = step
            removed = self._rotate()
            write_checkpoint_manifest(run_dir, latest=path, removed=removed)
            print(f"[checkpoint] saved {path.name}", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            if self.every_n_steps <= 0:
                return
            step = int(getattr(trainer, "global_step", 0) or 0)
            if step > 0 and step % self.every_n_steps == 0:
                self._save(trainer, f"step-{step:06d}")

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            if self.save_on_epoch_end:
                self._save(trainer, "epoch-end")

        def on_exception(self, trainer, pl_module, exception) -> None:
            if self.save_on_exception:
                try:
                    self._save(trainer, "exception", force=True)
                except Exception as exc:
                    print(f"[checkpoint] exception checkpoint failed: {exc}", flush=True)

        def on_fit_end(self, trainer, pl_module) -> None:
            self._save(trainer, "fit-end")

    return PeriodicCheckpointCallback()
