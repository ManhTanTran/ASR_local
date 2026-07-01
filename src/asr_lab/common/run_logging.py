"""Shared helpers for streaming command output and run progress logs."""
from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


Command = Sequence[object]


def command_text(cmd: Command) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_logged(
    cmd: Command,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command, stream output to the notebook, and optionally append it to a log file."""
    cmd = [str(part) for part in cmd]
    if log_path is None:
        print("$", command_text(cmd), flush=True)
        return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=check)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"\n\n=== {datetime.now(timezone.utc).isoformat()} ===\n$ {command_text(cmd)}\n"
    print(header.strip(), flush=True)

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(header)
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = proc.wait()
        footer = f"\n=== exit code {returncode} | log: {log_path} ===\n"
        log.write(footer)

    print(footer, flush=True)
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def read_log_tail(path: str | Path, max_lines: int = 220) -> str:
    path = Path(path)
    if not path.exists():
        return f"Log file does not exist: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = [
        f"Log path: {path}",
        f"Showing {min(len(lines), max_lines)}/{len(lines)} last lines",
        "-" * 80,
    ]
    return "\n".join(header + lines[-max_lines:])


def print_log_tail(path: str | Path, max_lines: int = 220) -> None:
    print(read_log_tail(path, max_lines=max_lines))


def write_run_status(run_dir: str | Path, run_id: str, state: str, **extra: object) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "status.json"
    payload = {"state": state, "run_id": run_id, **extra}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _metric_value(value) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_metric(name: str, value) -> str | None:
    value = _metric_value(value)
    if value is None:
        return None
    return f"{name}={value:.5g}"


def _optimizer_lr(trainer) -> float | None:
    try:
        optimizers = getattr(trainer, "optimizers", None) or []
        if optimizers and optimizers[0].param_groups:
            return optimizers[0].param_groups[0].get("lr")
    except Exception:
        return None
    return None


def make_lightning_metric_callback(every_n_steps: int = 25):
    """Return a Lightning callback that prints compact train/val metric lines."""
    import lightning.pytorch as pl

    class ConsoleMetricCallback(pl.Callback):
        def __init__(self, steps: int):
            self.every_n_steps = max(0, int(steps))
            self._last_train_step = -1
            self._last_val_epoch = -1

        def _selected_metrics(self, trainer, prefixes: tuple[str, ...]) -> list[str]:
            metrics = getattr(trainer, "callback_metrics", {}) or {}
            parts = []
            for name in sorted(metrics):
                low = str(name).lower()
                if any(low.startswith(prefix) for prefix in prefixes) or low in {"loss", "learning_rate"}:
                    part = _format_metric(str(name), metrics[name])
                    if part:
                        parts.append(part)
            return parts[:8]

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            if self.every_n_steps <= 0:
                return
            step = int(getattr(trainer, "global_step", 0) or 0)
            if step <= 0 or step == self._last_train_step or step % self.every_n_steps != 0:
                return
            self._last_train_step = step
            parts = self._selected_metrics(trainer, ("train",))
            lr = _optimizer_lr(trainer)
            if lr is not None and not any(part.startswith(("lr=", "learning_rate=")) for part in parts):
                parts.append(f"lr={lr:.5g}")
            print(f"[train] epoch={trainer.current_epoch} step={step} " + " ".join(parts), flush=True)

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            epoch = int(getattr(trainer, "current_epoch", 0) or 0)
            if epoch == self._last_val_epoch:
                return
            self._last_val_epoch = epoch
            parts = self._selected_metrics(trainer, ("val",))
            if parts:
                print(f"[val] epoch={epoch} step={trainer.global_step} " + " ".join(parts), flush=True)

    return ConsoleMetricCallback(every_n_steps)
