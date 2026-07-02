from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import pandas as pd

from asr_lab.common.metrics import extract_text, normalize_vi, utterance_error_rows
from asr_lab.data.vivos import VivosManifests, read_manifest
from asr_lab.model.parakeet import ParakeetConfig


def baseline_eval(model_path: Path, manifests: VivosManifests, config: ParakeetConfig, device: str) -> dict:
    import nemo.collections.asr as nemo_asr
    import torch
    from asr_lab.train.finetune_vivos import eval_wer

    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device).eval()
    score, rtf = eval_wer(model, str(manifests.test), config.test_n, config.eval_batch)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"wer": score, "rtf": rtf}


def train_decoder_only(
    model_path: Path,
    manifests: VivosManifests,
    config: ParakeetConfig,
    device: str,
    run_dir: Path,
    output_model: Path,
) -> dict:
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from omegaconf import open_dict

    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        accumulate_grad_batches=config.accumulate_grad_batches,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), monitor="val_wer", mode="min", save_top_k=1),
            EarlyStopping(monitor="val_wer", mode="min", patience=4),
        ],
        logger=pl.loggers.CSVLogger(save_dir=str(run_dir), name="logs"),
        log_every_n_steps=25,
    )
    model.set_trainer(trainer)

    loader = {
        "sample_rate": 16000,
        "num_workers": 2,
        "pin_memory": True,
        "max_duration": 20.0,
        "min_duration": 0.1,
        "is_tarred": False,
    }
    model.setup_training_data(
        {**loader, "manifest_filepath": str(manifests.train), "batch_size": config.train_batch, "shuffle": True}
    )
    model.setup_validation_data(
        {**loader, "manifest_filepath": str(manifests.val), "batch_size": config.train_batch, "shuffle": False}
    )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.decoder.parameters():
        parameter.requires_grad = True
    model.encoder.eval()

    with open_dict(model.cfg):
        model.cfg.optim = {
            "name": "adamw",
            "lr": config.lr,
            "weight_decay": 1e-3,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 50,
                "max_steps": config.max_epochs * max(1, manifests.train_rows // config.train_batch),
                "min_lr": 1e-6,
            },
        }
    model.setup_optimization(model.cfg.optim)

    start = time.perf_counter()
    trainer.fit(model)
    train_sec = time.perf_counter() - start
    model.save_to(str(output_model))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "train_sec": train_sec,
        "completed_epochs": trainer.current_epoch,
        "global_step": trainer.global_step,
        "output_model": str(output_model),
    }


def evaluate_finetuned(
    model_path: Path,
    manifests: VivosManifests,
    config: ParakeetConfig,
    baseline: dict,
    train_summary: dict,
    device: str,
    results_json: Path,
) -> dict:
    import nemo.collections.asr as nemo_asr
    import torch
    from asr_lab.train.finetune_vivos import eval_wer

    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device).eval()
    score, rtf = eval_wer(model, str(manifests.test), config.test_n, config.eval_batch)
    results = {
        "run_id": config.run_id,
        "pretrained": config.model_repo,
        "model_file": config.model_file,
        "kind": "parakeet_ctc_decoder_only",
        "wer_before": round(baseline["wer"], 4),
        "wer_after": round(score, 4),
        "rtf_before": round(baseline["rtf"], 4),
        "rtf_after": round(rtf, 4),
        "epochs": config.max_epochs,
        "completed_epochs": train_summary.get("completed_epochs"),
        "global_step": train_summary.get("global_step"),
        "batch": config.train_batch,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "effective_batch": config.train_batch * config.accumulate_grad_batches,
        "lr": config.lr,
        "nemo_file": Path(model_path).name,
        "train_sec": round(train_summary["train_sec"], 1),
        "cuda": torch.cuda.is_available(),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


def write_error_analysis(
    model_path: Path,
    manifests: VivosManifests,
    config: ParakeetConfig,
    device: str,
    out_csv: Path,
) -> pd.DataFrame:
    import nemo.collections.asr as nemo_asr

    rows = read_manifest(manifests.test)
    if config.test_n > 0:
        rows = rows[: config.test_n]
    paths = [row["audio_filepath"] for row in rows]
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device).eval()
    outputs = model.transcribe(paths, batch_size=config.eval_batch)
    hyps = [normalize_vi(extract_text(output)) for output in outputs]
    errors = pd.DataFrame(utterance_error_rows(rows, hyps)).sort_values("utt_wer", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out_csv, index=False)
    return errors


def list_artifacts(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(run_dir).rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "size_mb": round(path.stat().st_size / 1_000_000, 3),
                }
            )
    return pd.DataFrame(rows)
