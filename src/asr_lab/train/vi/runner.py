"""Orchestrator train/vi: config -> stage manifest -> tokenizer -> model -> fit(resume) -> eval cố định
-> save .nemo + backup + results.json. Bám luồng đã verified của finetune_vivos, thêm checkpoint/resume + eval đa-set.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
import time
from pathlib import Path

import torch
from omegaconf import open_dict

from asr_lab.common.metrics import extract_text, normalize_vi, wer
from asr_lab.train.vi.callbacks import build_callbacks
from asr_lab.train.vi.config import snapshot
from asr_lab.train.vi.model import load_model
from asr_lab.train.vi.stages import build_stage_manifests
from asr_lab.train.vi.tokenizer import assert_no_oov, build_tokenizer


def eval_wer(model, manifest: str, limit: int, batch: int) -> tuple[float, float]:
    """Transcribe -> (WER, RTF), chuẩn hoá normalize_vi như harness. limit<=0 = hết."""
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        return -1.0, -1.0
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize_vi(r["text"]) for r in rows]
    audio_sec = sum(r["duration"] for r in rows)
    model.eval()
    t0 = time.perf_counter()
    out = model.transcribe(paths, batch_size=batch)
    dt = time.perf_counter() - t0
    hyps = [normalize_vi(extract_text(x)) for x in out]
    return wer(refs, hyps), dt / max(audio_sec, 1e-9)


def _configure(model, stage: dict, cfg, total_steps: int, cuda: bool) -> None:
    common = dict(sample_rate=16000, batch_size=int(cfg.train.batch_size),
                  num_workers=int(cfg.train.num_workers), pin_memory=True,
                  max_duration=float(cfg.train.max_duration), min_duration=float(cfg.train.min_duration),
                  is_tarred=False)
    model.setup_training_data({**common, "manifest_filepath": stage["train"], "shuffle": True})
    model.setup_validation_data({**common, "manifest_filepath": stage["val"], "shuffle": False})
    warmup = max(50, min(500, total_steps // 10))
    with open_dict(model.cfg):
        model.cfg.optim = {
            "name": str(cfg.optim.name), "lr": float(cfg.optim.lr),
            "weight_decay": float(cfg.optim.weight_decay),
            "sched": {"name": str(cfg.optim.sched.name), "warmup_steps": warmup,
                      "max_steps": total_steps, "min_lr": float(cfg.optim.sched.min_lr)},
        }
    model.setup_optimization(model.cfg.optim)
    # SpecAugment: best-effort rebuild từ cfg (không chặn smoke nếu lỗi)
    sa = cfg.train.get("spec_augment")
    if sa:
        try:
            with open_dict(model.cfg):
                model.cfg.spec_augment.freq_masks = int(sa.freq_masks)
                model.cfg.spec_augment.time_masks = int(sa.time_masks)
            model.spec_augmentation = model.from_config_dict(model.cfg.spec_augment)
        except Exception as e:  # noqa: BLE001
            print(f"[configure] spec_augment giữ mặc định (rebuild lỗi: {type(e).__name__})", flush=True)
    print(f"[configure] warmup={warmup} max_steps={total_steps} bs={common['batch_size']} "
          f"accum={cfg.train.accum_grad}", flush=True)


def run(cfg, resume_ckpt: str | None = None) -> dict:
    import lightning.pytorch as pl
    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(4)
    batch = int(cfg.train.batch_size)
    limit = int(cfg.get("eval_limit", 0))
    run_dir = Path(cfg.run.artifacts_dir) / "runs" / cfg.run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot(cfg, run_dir)

    stage = build_stage_manifests(cfg, run_dir)

    tok_dir = cfg.model.tokenizer.get("dir")
    if cfg.model.get("change_vocabulary") and not tok_dir:
        tok_dir = build_tokenizer(stage["train"], run_dir / "tokenizer_vi",
                                  int(cfg.model.tokenizer.vocab_size), str(cfg.model.tokenizer.type))
        assert_no_oov(stage["train"], tok_dir)   # cổng trước GPU

    model = load_model(cfg, tok_dir, cuda)

    before = {lab: eval_wer(model, p, limit, batch)[0] for lab, p in stage["eval_fixed"].items()}
    print(f"[eval TRƯỚC] {({k: round(v*100, 2) for k, v in before.items()})}", flush=True)

    n_train = sum(1 for l in open(stage["train"], encoding="utf-8") if l.strip())
    # total_steps cho cosine phải đếm theo OPTIMIZER-step (Lightning global_step tăng 1 mỗi accum_grad
    # micro-batch), KHÔNG phải micro-batch — nếu không LR decay chậm gấp accum lần (giữ LR cao).
    eff_batch = batch * int(cfg.train.accum_grad)
    steps_per_epoch = max(1, math.ceil(n_train / eff_batch))
    total_steps = int(cfg.train.epochs) * steps_per_epoch
    _configure(model, stage, cfg, total_steps, cuda)

    logger = pl.loggers.CSVLogger(save_dir=str(run_dir), name="logs", flush_logs_every_n_steps=25)
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.epochs), max_time={"minutes": int(cfg.train.max_minutes)},
        accelerator="gpu" if cuda else "cpu", devices=1,
        precision=(str(cfg.train.precision) if cuda else 32),
        accumulate_grad_batches=int(cfg.train.accum_grad),
        enable_checkpointing=True, callbacks=build_callbacks(cfg, run_dir / "checkpoints"),
        logger=logger, enable_progress_bar=True, log_every_n_steps=25,
        val_check_interval=float(cfg.train.get("val_check_interval", 1.0)),
    )
    model.set_trainer(trainer)
    t0 = time.perf_counter()
    trainer.fit(model, ckpt_path=resume_ckpt)   # resume giữa run nếu có
    train_sec = time.perf_counter() - t0

    after = {lab: eval_wer(model, p, limit, batch)[0] for lab, p in stage["eval_fixed"].items()}
    print(f"[eval SAU] {({k: round(v*100, 2) for k, v in after.items()})}", flush=True)

    nemo_path = run_dir / f"{cfg.run.id}.nemo"
    if not cfg.get("no_save"):
        model.save_to(str(nemo_path))
        bk = cfg.run.get("backup_dir")
        if bk:
            try:
                Path(bk).mkdir(parents=True, exist_ok=True)
                shutil.copy2(nemo_path, Path(bk) / nemo_path.name)
                print(f"[save] backup -> {bk}/{nemo_path.name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[save] backup lỗi: {type(e).__name__}: {e}", flush=True)

    results = {
        "run_id": cfg.run.id, "stage": cfg.run.get("stage"), "kind": "train_vi",
        "init_from": str(cfg.model.init_from), "vocab_size": model.tokenizer.vocab_size,
        "eff_batch": batch * int(cfg.train.accum_grad), "precision": str(cfg.train.precision),
        "epochs": int(cfg.train.epochs), "n_train": n_train, "total_steps": total_steps,
        "completed_epochs": trainer.current_epoch, "global_step": trainer.global_step,
        "train_sec": round(train_sec, 1), "cuda": cuda, "gpu": ("GB10" if cuda else "cpu"),
        "wer_before": {k: round(v, 4) for k, v in before.items()},
        "wer_after": {k: round(v, 4) for k, v in after.items()},
        "per_set": stage["per_set"],
        "nemo_file": (nemo_path.name if not cfg.get("no_save") else None),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "ok", "run_id": cfg.run.id, "wer_after": results["wer_after"]}, ensure_ascii=False))
    print("RESULTS:", json.dumps(results, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    return results
