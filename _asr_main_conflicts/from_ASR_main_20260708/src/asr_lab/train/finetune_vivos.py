"""Fine-tune nemotron-speech-streaming-en-0.6b sang tiếng Việt bằng VIVOS — chạy trên Kaggle GPU.

Vì model nền là English-only (không có token tiếng Việt), fine-tune ĐÚNG cách = đổi bộ từ vựng:
  1. Tải VIVOS train/val/test (HF mirror parquet) -> manifest NeMo.
  2. Train BPE tokenizer tiếng Việt từ transcript train (SentencePiece, char_coverage=1.0 để phủ hết
     ký tự có dấu) -> thư mục tokenizer.
  3. Nạp model pretrained, đo WER test TRƯỚC (đầu English, kỳ vọng ~100%).
  4. change_vocabulary() -> dựng lại decoder + joint theo vocab tiếng Việt (encoder giữ nguyên).
  5. Train vài epoch, đo WER test SAU.
  6. Lưu .nemo + results.json (wer_before/after) + status.json vào artifacts/runs/<run_id>/.

Artifact root lấy từ env ASR_ARTIFACTS_DIR (Kaggle: /kaggle/working/artifacts) -> run-dir để adapter pull.

Chạy local CPU smoke nhỏ:
  uv run python -m asr_lab.train.finetune_vivos --run-id dev --train-n 40 --val-n 10 --limit-test 10 \
      --epochs 1 --batch 4 --vocab-size 128
Trên Kaggle GPU (qua adapter): --epochs 40 --batch 16 (mặc định lấy hết data).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

import nemo.collections.asr as nemo_asr  # noqa: E402
import lightning.pytorch as pl  # noqa: E402  (NeMo 2.x dùng lightning.pytorch, KHÔNG phải pytorch_lightning)
from omegaconf import open_dict  # noqa: E402

from asr_lab.data.vivos import dump_split  # noqa: E402  (data-prep dùng chung)
from asr_lab.common.metrics import normalize_vi, extract_text, wer  # noqa: E402

PRETRAINED = "nvidia/nemotron-speech-streaming-en-0.6b"  # default; đổi bằng --pretrained


def artifacts_root() -> Path:
    return Path(os.environ.get("ASR_ARTIFACTS_DIR", "artifacts"))


def prepare_data(data_dir: Path, train_n: int, val_n: int, test_n: int) -> dict:
    """Tải VIVOS, cắt val từ ĐUÔI train (không đụng test). Trả về dict manifest paths.

    QUAN TRỌNG: chuẩn hoá text train/val KHỚP tokenizer (lowercase + bỏ dấu câu). Tokenizer build
    từ text đã normalize_vi; nếu manifest train để RAW (có HOA + dấu câu) thì chữ hoa/dấu câu thành
    <unk> -> model học phát <unk> ở chữ hoa đầu câu/danh từ riêng (đã gặp: 'Tuy'->'⁇uy').
    """
    train_full = dump_split("train", data_dir / "raw/train",
                            data_dir / "manifests/train_full.jsonl", n=train_n)
    test_m = dump_split("test", data_dir / "raw/test",
                        data_dir / "manifests/test.jsonl", n=test_n)

    def norm_rows(lines):
        out = []
        for ln in lines:
            r = json.loads(ln)
            r["text"] = normalize_vi(r["text"])  # khớp vocab tokenizer -> không sinh <unk>
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
        return out

    rows = norm_rows(train_full.open(encoding="utf-8"))
    val_n = min(val_n, max(1, len(rows) // 10))
    train_m = data_dir / "manifests/train.jsonl"
    val_m = data_dir / "manifests/val.jsonl"
    train_m.write_text("".join(rows[:-val_n]), encoding="utf-8")
    val_m.write_text("".join(rows[-val_n:]), encoding="utf-8")
    print(f"train={len(rows)-val_n} val={val_n} (text đã normalize_vi khớp tokenizer) test=đọc riêng")
    return {"train": str(train_m), "val": str(val_m), "test": str(test_m)}


def build_vi_tokenizer(train_manifest: str, out_dir: Path, vocab_size: int) -> Path:
    """Train SentencePiece BPE từ transcript train. char_coverage=1.0 để phủ hết dấu tiếng Việt."""
    import sentencepiece as spm
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / "corpus.txt"
    with txt.open("w", encoding="utf-8") as w:
        for line in open(train_manifest, encoding="utf-8"):
            w.write(normalize_vi(json.loads(line)["text"]) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(txt), model_prefix=str(out_dir / "tokenizer"),
        vocab_size=vocab_size, model_type="bpe", character_coverage=1.0,
        bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,  # ASR BPE: không bos/eos/pad, có unk
    )
    # NeMo SentencePieceTokenizer cần tokenizer.model trong thư mục; viết kèm vocab.txt cho tiện soi.
    vocab_txt = out_dir / "vocab.txt"
    with open(str(out_dir / "tokenizer.vocab"), encoding="utf-8") as f, vocab_txt.open("w", encoding="utf-8") as w:
        for ln in f:
            w.write(ln.split("\t")[0] + "\n")
    print(f"tokenizer VN vocab_size={vocab_size} -> {out_dir}")
    return out_dir


def assert_no_oov(train_manifest: str, tok_dir: Path, max_rate: float = 1e-4) -> None:
    """CỔNG trước train: nhãn train chạy qua tokenizer KHÔNG được sinh <unk>. OOV>0 = lệch chuẩn hoá
    vocab vs nhãn (bug đã gặp: chữ hoa/dấu câu -> <unk> -> WER thổi phồng ngầm). Dừng ngay, đừng tốn GPU.

    Soi trực tiếp file SentencePiece (.model) — unk_id=0 như lúc train tokenizer."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(tok_dir / "tokenizer.model"))
    unk = sp.unk_id()
    n_unk = n_tok = 0
    samples = []
    for line in open(train_manifest, encoding="utf-8"):
        text = json.loads(line)["text"]
        ids = sp.encode(text, out_type=int)
        u = sum(1 for i in ids if i == unk)
        if u and len(samples) < 5:
            samples.append(text)
        n_unk += u
        n_tok += len(ids)
    rate = n_unk / max(n_tok, 1)
    print(f"CỔNG OOV: rate={rate:.4%} ({n_unk}/{n_tok} token là <unk>)", flush=True)
    if rate > max_rate:
        for s in samples:
            print("  nhãn sinh <unk>:", s[:80])
        raise SystemExit(f"OOV={rate:.4%} > {max_rate:.4%} — lệch chuẩn hoá vocab vs nhãn train. DỪNG.")


def eval_wer(model, test_manifest: str, limit: int, batch: int) -> tuple[float, float]:
    """Transcribe test -> (WER, RTF). Dùng cùng chuẩn hoá normalize_vi như harness eval."""
    rows = [json.loads(l) for l in open(test_manifest, encoding="utf-8")]
    if limit > 0:
        rows = rows[:limit]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize_vi(r["text"]) for r in rows]
    audio_sec = sum(r["duration"] for r in rows)
    model.eval()
    t0 = time.perf_counter()
    out = model.transcribe(paths, batch_size=batch)
    dt = time.perf_counter() - t0
    hyps = [normalize_vi(extract_text(x)) for x in out]
    return wer(refs, hyps), dt / max(audio_sec, 1e-9)


def configure_finetune(model, data: dict, args, total_steps: int) -> None:
    """Gắn train/val data + optimizer. Cosine cần max_steps THẬT (đếm từ data) — nếu không
    biết horizon, LR decay sụp sớm -> train đứng (đã gặp ở verify)."""
    common = dict(sample_rate=16000, batch_size=args.batch, num_workers=2,
                  pin_memory=True, max_duration=20.0, min_duration=0.1, is_tarred=False)
    model.setup_training_data({**common, "manifest_filepath": data["train"], "shuffle": True})
    model.setup_validation_data({**common, "manifest_filepath": data["val"], "shuffle": False})
    warmup = max(50, min(500, total_steps // 10))
    with open_dict(model.cfg):
        model.cfg.optim = {
            "name": "adamw", "lr": args.lr, "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": warmup,
                      "max_steps": total_steps, "min_lr": 1e-5},
        }
    model.setup_optimization(model.cfg.optim)
    if args.freeze_encoder:
        model.encoder.freeze()  # tiết kiệm bộ nhớ + nhanh: chỉ học decoder+joint mới (vocab VN)
        print(f"đã freeze encoder — chỉ train decoder+joint | warmup={warmup} max_steps={total_steps}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="vivos-ft")
    ap.add_argument("--pretrained", default=PRETRAINED, help="model nền NeMo để fine-tune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 sẽ override epochs")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--train-n", type=int, default=0, help="0 = lấy hết train")
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--limit-test", type=int, default=0, help="0 = eval hết test")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=480,
                    help="chặn thời gian train (Lightning max_time) -> luôn kịp eval+save trong khung Kaggle")
    ap.add_argument("--no-save", action="store_true", help="bỏ qua save .nemo (cho run verify nhanh)")
    ap.add_argument("--precision", default="32",
                    help="32 (ổn định, tránh RNNT collapse-to-blank do fp16) | 16-mixed | bf16-mixed")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(4)  # an toàn CPU khi smoke local
    run_dir = artifacts_root() / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Data để NGOÀI run-dir: 11k wav không được lẫn vào artifact pull. Kaggle: ASR_DATA_DIR=/tmp/...
    data_dir = Path(os.environ.get("ASR_DATA_DIR", str(run_dir / "data")))
    print(f"== fine-tune {args.pretrained} | cuda={cuda} | run_dir={run_dir} | data_dir={data_dir} ==", flush=True)

    data = prepare_data(data_dir, args.train_n, args.val_n, args.limit_test)
    tok_dir = build_vi_tokenizer(data["train"], run_dir / "tokenizer_vi", args.vocab_size)
    assert_no_oov(data["train"], tok_dir)  # cổng: chặn bug lệch chuẩn hoá TRƯỚC khi tốn GPU

    model = nemo_asr.models.ASRModel.from_pretrained(
        args.pretrained, map_location="cuda" if cuda else "cpu")
    wer_before, rtf_before = eval_wer(model, data["test"], args.limit_test, args.batch)
    print(f"WER TRƯỚC (đầu English): {wer_before*100:.2f}%", flush=True)

    import math
    n_train = sum(1 for _ in open(data["train"], encoding="utf-8"))
    steps_per_epoch = max(1, math.ceil(n_train / args.batch))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    print(f"n_train={n_train} steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    model.change_vocabulary(new_tokenizer_dir=str(tok_dir), new_tokenizer_type="bpe")
    configure_finetune(model, data, args, total_steps)

    # CSVLogger ghi metrics.csv (train_loss/val_loss/val_wer) vào run-dir -> pull về vẽ curve.
    logger = pl.loggers.CSVLogger(save_dir=str(run_dir), name="logs", flush_logs_every_n_steps=25)
    trainer = pl.Trainer(
        max_epochs=args.epochs if args.max_steps <= 0 else -1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        max_time={"minutes": args.max_minutes},  # chốt: luôn dừng kịp để eval+save trong khung Kaggle
        accelerator="gpu" if cuda else "cpu", devices=1,
        precision=(args.precision if cuda else 32),
        enable_checkpointing=False, logger=logger,
        enable_progress_bar=True, log_every_n_steps=25,
        val_check_interval=1.0,
    )
    model.set_trainer(trainer)
    t0 = time.perf_counter()
    trainer.fit(model)
    train_sec = time.perf_counter() - t0
    print(f"train xong {train_sec/60:.1f} phút", flush=True)

    wer_after, rtf_after = eval_wer(model, data["test"], args.limit_test, args.batch)
    print(f"WER SAU (đầu tiếng Việt): {wer_after*100:.2f}%", flush=True)

    nemo_path = run_dir / "nemotron_vivos_ft.nemo"
    if not args.no_save:
        model.save_to(str(nemo_path))

    results = {
        "pretrained": args.pretrained, "run_id": args.run_id,
        "wer_before": round(wer_before, 4), "wer_after": round(wer_after, 4),
        "rtf_before": round(rtf_before, 4), "rtf_after": round(rtf_after, 4),
        "epochs": args.epochs, "max_steps": args.max_steps, "batch": args.batch,
        "lr": args.lr, "vocab_size": args.vocab_size, "freeze_encoder": args.freeze_encoder,
        "max_minutes": args.max_minutes, "completed_epochs": trainer.current_epoch,
        "global_step": trainer.global_step, "train_sec": round(train_sec, 1), "cuda": cuda,
        "nemo_file": (nemo_path.name if not args.no_save else None),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "ok", "run_id": args.run_id, "wer_before": results["wer_before"],
         "wer_after": results["wer_after"]}, ensure_ascii=False))
    print("RESULTS:", json.dumps(results, ensure_ascii=False), flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
