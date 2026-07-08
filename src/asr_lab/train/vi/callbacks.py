"""Callback Lightning cho train/vi: ModelCheckpoint top-k theo val_wer + save_last (resume), LR monitor,
EMA (tuỳ chọn, import mềm để CPU smoke không vỡ nếu bản NeMo khác)."""
from __future__ import annotations

from pathlib import Path


def build_callbacks(cfg, ckpt_dir: Path) -> list:
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor=cfg.checkpoint.monitor, mode=cfg.checkpoint.mode,
            save_top_k=int(cfg.checkpoint.save_top_k), save_last=True,
            every_n_train_steps=int(cfg.checkpoint.every_n_train_steps) or None,
            filename="{step}-{val_wer:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    ema = cfg.checkpoint.get("ema")
    if ema:
        try:
            from nemo.collections.common.callbacks import EMA
            cbs.append(EMA(decay=float(ema)))
            print(f"[callbacks] EMA decay={ema}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[callbacks] EMA tắt (import lỗi: {type(e).__name__})", flush=True)
    return cbs
