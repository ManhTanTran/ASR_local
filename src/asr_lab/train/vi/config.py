"""Nạp + merge + validate config YAML cho hệ train/vi. Reproducible: snapshot config + git sha vào run-dir.

Merge: _base.yaml (cùng thư mục file exp) <- file exp <- CLI overrides (dotlist, vd optim.lr=5e-5).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(exp_path: str, overrides: list[str] | None = None) -> DictConfig:
    exp_path = Path(exp_path)
    base_path = exp_path.parent / "_base.yaml"
    cfg = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    _validate(cfg)
    return cfg  # type: ignore[return-value]


def _validate(cfg: DictConfig) -> None:
    missing = []
    if not cfg.run.get("id"):
        missing.append("run.id")
    if not cfg.data.get("train_sets"):
        missing.append("data.train_sets")
    if not cfg.model.get("init_from"):
        missing.append("model.init_from")
    if missing:
        raise SystemExit(f"config thiếu bắt buộc: {missing}")
    if cfg.model.get("change_vocabulary") and not cfg.model.tokenizer.get("dir") \
            and not cfg.model.tokenizer.get("vocab_size"):
        raise SystemExit("change_vocabulary=true nhưng thiếu tokenizer.dir hoặc tokenizer.vocab_size")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def snapshot(cfg: DictConfig, run_dir: Path) -> None:
    """Ghi config đã merge + git sha vào run-dir để truy ngược mọi run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "git_sha.txt").write_text(_git_sha())
