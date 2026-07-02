"""Shared helpers for streaming command output and run progress logs."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import deque
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
    console_patterns: Sequence[str] | None = None,
    error_tail_lines: int = 0,
    system_exit_on_fail: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command and append full output to a log file.

    ``console_patterns`` keeps Kaggle notebook output small: every line still goes
    to ``log_path``, but only matching lines are printed to the visible notebook.
    """
    cmd = [str(part) for part in cmd]
    if log_path is None:
        print("$", command_text(cmd), flush=True)
        return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=check)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"\n\n=== {datetime.now(timezone.utc).isoformat()} ===\n$ {command_text(cmd)}\n"
    print(header.strip(), flush=True)
    console_re = re.compile("|".join(f"(?:{pattern})" for pattern in console_patterns)) if console_patterns else None
    tail = deque(maxlen=max(0, int(error_tail_lines)))

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
            log.write(line)
            if tail.maxlen:
                tail.append(line)
            if console_re is None or console_re.search(line):
                print(line, end="", flush=True)
        returncode = proc.wait()
        footer = f"\n=== exit code {returncode} | log: {log_path} ===\n"
        log.write(footer)

    print(footer, flush=True)
    if check and returncode != 0:
        if tail:
            print(f"Last {len(tail)} log lines:", flush=True)
            print("".join(tail), end="", flush=True)
        message = f"Command failed with exit code {returncode}. Full log: {log_path}"
        if system_exit_on_fail:
            raise SystemExit(message)
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


def _find_metric(metrics: Mapping[object, object], candidates: Sequence[str]) -> float | None:
    for candidate in candidates:
        if candidate in metrics:
            value = _metric_value(metrics[candidate])
            if value is not None:
                return value
    lowered = {str(key).lower(): key for key in metrics}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            value = _metric_value(metrics[key])
            if value is not None:
                return value
    return None


def _optimizer_lr(trainer) -> float | None:
    try:
        optimizers = getattr(trainer, "optimizers", None) or []
        if optimizers and optimizers[0].param_groups:
            return optimizers[0].param_groups[0].get("lr")
    except Exception:
        return None
    return None


def make_lightning_metric_callback(every_n_steps: int = 25):
    """Return a Lightning callback that prints compact train/val metric lines.

    ``every_n_steps=0`` disables per-step logs but still prints one epoch summary.
    """
    import lightning.pytorch as pl

    class ConsoleMetricCallback(pl.Callback):
        def __init__(self, steps: int):
            self.every_n_steps = max(0, int(steps))
            self._last_train_step = -1
            self._last_val_epoch = -1
            self._last_train_parts: list[str] = []
            self._reset_val_accumulators()

        def _reset_val_accumulators(self) -> None:
            self._val_loss_sum = 0.0
            self._val_loss_count = 0
            self._val_wer_sum = 0.0
            self._val_wer_count = 0
            self._val_wer_num = 0.0
            self._val_wer_denom = 0.0

        def _collect_val_output(self, output) -> None:
            if output is None:
                return
            if isinstance(output, Mapping):
                loss = _find_metric(output, ("val_loss", "validation_loss", "loss"))
                if loss is not None:
                    self._val_loss_sum += loss
                    self._val_loss_count += 1
                wer = _find_metric(output, ("val_wer", "wer"))
                if wer is not None:
                    self._val_wer_sum += wer
                    self._val_wer_count += 1
                wer_num = _find_metric(output, ("val_wer_num", "wer_num"))
                wer_denom = _find_metric(output, ("val_wer_denom", "wer_denom"))
                if wer_num is not None and wer_denom is not None:
                    self._val_wer_num += wer_num
                    self._val_wer_denom += wer_denom
                return
            if isinstance(output, (list, tuple)):
                for item in output:
                    self._collect_val_output(item)

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

        def _epoch_parts(self, trainer) -> list[str]:
            metrics = getattr(trainer, "callback_metrics", {}) or {}
            parts: list[str] = []
            seen: set[str] = set()

            def add_value(label: str, value: float | None) -> None:
                if value is not None and label not in seen:
                    parts.append(f"{label}={value:.5g}")
                    seen.add(label)

            def add(label: str, candidates: tuple[str, ...]) -> None:
                add_value(label, _find_metric(metrics, candidates))

            add("train_loss", ("train_loss", "loss"))
            if "train_loss" not in seen:
                for part in self._last_train_parts:
                    if "loss=" in part:
                        parts.append(part if part.startswith("train_") else f"train_{part}")
                        seen.add("train_loss")
                        break
            add("val_loss", ("val_loss", "validation_loss"))
            if "val_loss" not in seen and self._val_loss_count:
                add_value("val_loss", self._val_loss_sum / self._val_loss_count)
            add("val_wer", ("val_wer", "wer"))
            if "val_wer" not in seen:
                wer_num = _find_metric(metrics, ("val_wer_num", "wer_num"))
                wer_denom = _find_metric(metrics, ("val_wer_denom", "wer_denom"))
                if wer_num is not None and wer_denom:
                    add_value("val_wer", wer_num / wer_denom)
            if "val_wer" not in seen and self._val_wer_denom:
                add_value("val_wer", self._val_wer_num / self._val_wer_denom)
            if "val_wer" not in seen and self._val_wer_count:
                add_value("val_wer", self._val_wer_sum / self._val_wer_count)
            lr = _optimizer_lr(trainer)
            if lr is not None:
                parts.append(f"lr={lr:.5g}")
            return parts

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            parts = self._selected_metrics(trainer, ("train",))
            if parts:
                self._last_train_parts = parts
            if self.every_n_steps <= 0:
                return
            step = int(getattr(trainer, "global_step", 0) or 0)
            if step <= 0 or step == self._last_train_step or step % self.every_n_steps != 0:
                return
            self._last_train_step = step
            lr = _optimizer_lr(trainer)
            if lr is not None and not any(part.startswith(("lr=", "learning_rate=")) for part in parts):
                parts.append(f"lr={lr:.5g}")
            print(f"[train] epoch={trainer.current_epoch} step={step} " + " ".join(parts), flush=True)

        def on_validation_epoch_start(self, trainer, pl_module) -> None:
            self._reset_val_accumulators()

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ) -> None:
            self._collect_val_output(outputs)

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            epoch = int(getattr(trainer, "current_epoch", 0) or 0)
            if epoch == self._last_val_epoch:
                return
            self._last_val_epoch = epoch
            parts = self._epoch_parts(trainer)
            if parts:
                print(f"[epoch] epoch={epoch} step={trainer.global_step} " + " ".join(parts), flush=True)

    return ConsoleMetricCallback(every_n_steps)
