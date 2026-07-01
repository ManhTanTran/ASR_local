"""Fine-tune TIẾP từ ckpt VIVOS (.nemo) + GỘP Common Voice — Phương án A: GIỮ NGUYÊN tokenizer.

Khác `finetune_vivos` (from_pretrained + change_vocabulary = dựng tokenizer MỚI): ở đây
`restore_from` ckpt đã fine-tune VIVOS -> giữ luôn tokenizer + decoder đã chín, chỉ train tiếp
trên data gộp (VIVOS train + CV train). Vì giữ tokenizer VIVOS nên CV phải lọc theo charset tokenizer
(0 <unk>) — xem asr_lab.data.common_voice + experiments/04_add_commonvoice/spec.md.

Đo 2×2 trong MỘT run: {VIVOS test, CV test} × {trước resume, sau train} -> results.json.

Chạy local CPU smoke:
  uv run python -m asr_lab.train.continue_vi --resume-from artifacts/runs/vivos-fc115m-v2norm/nemotron_vivos_ft.nemo \
      --run-id dev-cv --vivos-train-n 30 --cv-train-n 30 --limit-test 8 --epochs 1 --batch 4
Trên Kaggle GPU: --resume-from <glob .nemo trong /kaggle/input> --epochs 25 --batch 16 --precision 32
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import torch

import nemo.collections.asr as nemo_asr  # noqa: E402
import lightning.pytorch as pl  # noqa: E402

from asr_lab.data.vivos import dump_split as vivos_dump
from asr_lab.data.common_voice import dump_split as cv_dump
from asr_lab.train.finetune_vivos import eval_wer, configure_finetune, artifacts_root


def resolve_nemo(hint: str) -> str:
    """Đường dẫn .nemo trực tiếp; nếu không tồn tại, rglob *.nemo trong /kaggle/input (kernel_sources)."""
    p = Path(hint)
    if p.is_file():            # CHỈ nhận khi là FILE thật; '/kaggle/input' là DIR -> phải rglob bên dưới
        return str(p)
    for base in ("/kaggle/input", hint):
        hits = sorted(Path(base).rglob("*.nemo")) if Path(base).exists() else []
        if hits:
            print(f"[resolve] .nemo -> {hits[0]}", flush=True)
            return str(hits[0])
    raise SystemExit(f"Không thấy .nemo từ hint='{hint}' (cả /kaggle/input).")


def tokenizer_charset(model) -> set[str]:
    """Tập ký tự mà tokenizer của model phủ — gom từ mọi piece vocab (bỏ meta '▁' + token special).

    Dùng để lọc CV train: char nào không thuộc set này -> SP sinh <unk> -> phải drop clip."""
    vocab = model.tokenizer.vocab  # list[str] các piece
    chars: set[str] = set()
    for piece in vocab:
        if piece.startswith("<") and piece.endswith(">"):
            continue  # token special (<unk>...) — không tính vào charset thật
        chars.update(piece.replace("▁", ""))  # bỏ ký tự meta đầu-từ của SentencePiece
    chars.discard("")
    return chars


def assert_charset_ok(manifest: str, charset: set[str], max_rate: float = 1e-4) -> None:
    """CỔNG: nhãn train chỉ chứa ký tự trong charset tokenizer (== 0 <unk> với SP char-coverage).

    Tương đương assert_no_oov nhưng kiểm ở mức KÝ TỰ (không cần file .model rời sau restore_from)."""
    n_bad = n_char = 0
    samples = []
    for line in open(manifest, encoding="utf-8"):
        text = json.loads(line)["text"]
        for c in text:
            if c == " ":
                continue
            n_char += 1
            if c not in charset:
                n_bad += 1
                if len(samples) < 5:
                    samples.append(text)
    rate = n_bad / max(n_char, 1)
    print(f"CỔNG charset: {rate:.4%} ký tự ngoài vocab ({n_bad}/{n_char})", flush=True)
    if rate > max_rate:
        for s in samples:
            print("  nhãn lỗi:", s[:80])
        raise SystemExit(f"charset OOV={rate:.4%} > {max_rate:.4%} — lệch vocab vs nhãn train. DỪNG.")


def prepare_combined(data_dir: Path, charset: set[str], vivos_train_n: int, cv_train_n: int,
                     test_n: int) -> dict:
    """Dựng manifest gộp: train = VIVOS train + CV train (lọc charset); val = đuôi mỗi nguồn;
    GIỮ RIÊNG 2 test (VIVOS test, CV test) để đo bảng 2×2. Text mọi nơi đã normalize_vi."""
    # --- train sources ---
    v_train = vivos_dump("train", data_dir / "raw/vivos/train",
                         data_dir / "m/vivos_train_full.jsonl", n=vivos_train_n)
    c_train = cv_dump("train", data_dir / "raw/cv/train",
                      data_dir / "m/cv_train_full.jsonl", charset=charset, n=cv_train_n)
    # VIVOS dump KHÔNG normalize (loader giữ raw) -> normalize tại đây cho khớp tokenizer.
    from asr_lab.common.metrics import normalize_vi

    def norm_lines(path):
        out = []
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln); r["text"] = normalize_vi(r["text"])
            out.append(json.dumps(r, ensure_ascii=False) + "\n")
        return out

    v_rows = norm_lines(v_train)      # VIVOS cần normalize
    c_rows = [ln for ln in open(c_train, encoding="utf-8")]  # CV đã normalize trong loader
    # cắt val: 5% mỗi nguồn từ đuôi (không đụng phần train)
    v_val_n = max(1, len(v_rows) // 20)
    c_val_n = max(1, len(c_rows) // 20)
    train_rows = v_rows[:-v_val_n] + c_rows[:-c_val_n]
    val_rows = v_rows[-v_val_n:] + c_rows[-c_val_n:]
    train_m = data_dir / "m/train.jsonl"; val_m = data_dir / "m/val.jsonl"
    train_m.write_text("".join(train_rows), encoding="utf-8")
    val_m.write_text("".join(val_rows), encoding="utf-8")
    # --- test sources (giữ riêng, KHÔNG lọc charset) ---
    v_test = vivos_dump("test", data_dir / "raw/vivos/test", data_dir / "m/vivos_test.jsonl", n=test_n)
    c_test = cv_dump("test", data_dir / "raw/cv/test", data_dir / "m/cv_test.jsonl",
                     charset=None, n=test_n)
    print(f"train={len(train_rows)} (VIVOS {len(v_rows)-v_val_n} + CV {len(c_rows)-c_val_n}) "
          f"val={len(val_rows)} | test: VIVOS+CV riêng", flush=True)
    return {"train": str(train_m), "val": str(val_m),
            "vivos_test": str(v_test), "cv_test": str(c_test)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-from", required=True, help=".nemo ckpt VIVOS (hoặc hint glob /kaggle/input)")
    ap.add_argument("--run-id", default="vivos-cv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vivos-train-n", type=int, default=0, help="0 = hết")
    ap.add_argument("--cv-train-n", type=int, default=0, help="0 = hết")
    ap.add_argument("--limit-test", type=int, default=0, help="0 = eval hết mỗi test")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=480)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--precision", default="32")
    args = ap.parse_args()
    # configure_finetune đọc args.freeze_encoder; alias để khớp.

    cuda = torch.cuda.is_available()
    if not cuda:
        torch.set_num_threads(4)
    run_dir = artifacts_root() / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.environ.get("ASR_DATA_DIR", str(run_dir / "data")))
    nemo_path = resolve_nemo(args.resume_from)
    print(f"== continue {nemo_path} | cuda={cuda} | run_dir={run_dir} ==", flush=True)

    # restore: GIỮ tokenizer + decoder đã fine-tune VIVOS (không change_vocabulary)
    model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cuda" if cuda else "cpu")
    charset = tokenizer_charset(model)
    print(f"charset tokenizer: {len(charset)} ký tự", flush=True)

    data = prepare_combined(data_dir, charset, args.vivos_train_n, args.cv_train_n, args.limit_test)
    assert_charset_ok(data["train"], charset)  # cổng trước GPU

    # --- eval TRƯỚC (ckpt hiện tại) trên cả 2 test ---
    wv_b, rv_b = eval_wer(model, data["vivos_test"], args.limit_test, args.batch)
    wc_b, rc_b = eval_wer(model, data["cv_test"], args.limit_test, args.batch)
    print(f"TRƯỚC | VIVOS {wv_b*100:.2f}% | CV {wc_b*100:.2f}%", flush=True)

    n_train = sum(1 for _ in open(data["train"], encoding="utf-8"))
    steps_per_epoch = max(1, math.ceil(n_train / args.batch))
    total_steps = args.epochs * steps_per_epoch
    print(f"n_train={n_train} steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    configure_finetune(model, data, args, total_steps)
    logger = pl.loggers.CSVLogger(save_dir=str(run_dir), name="logs", flush_logs_every_n_steps=25)
    trainer = pl.Trainer(
        max_epochs=args.epochs, max_time={"minutes": args.max_minutes},
        accelerator="gpu" if cuda else "cpu", devices=1,
        precision=(args.precision if cuda else 32),
        enable_checkpointing=False, logger=logger,
        enable_progress_bar=True, log_every_n_steps=25, val_check_interval=1.0,
    )
    model.set_trainer(trainer)
    t0 = time.perf_counter()
    trainer.fit(model)
    train_sec = time.perf_counter() - t0
    print(f"train xong {train_sec/60:.1f} phút", flush=True)

    # --- eval SAU trên cả 2 test ---
    wv_a, rv_a = eval_wer(model, data["vivos_test"], args.limit_test, args.batch)
    wc_a, rc_a = eval_wer(model, data["cv_test"], args.limit_test, args.batch)
    print(f"SAU | VIVOS {wv_a*100:.2f}% | CV {wc_a*100:.2f}%", flush=True)

    out_nemo = run_dir / "vivos_cv_ft.nemo"
    if not args.no_save:
        model.save_to(str(out_nemo))

    results = {
        "run_id": args.run_id, "resume_from": Path(nemo_path).name, "kind": "continue_vivos+cv",
        "wer_vivos_before": round(wv_b, 4), "wer_vivos_after": round(wv_a, 4),
        "wer_cv_before": round(wc_b, 4), "wer_cv_after": round(wc_a, 4),
        "rtf_vivos_after": round(rv_a, 4), "rtf_cv_after": round(rc_a, 4),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "n_train": n_train, "completed_epochs": trainer.current_epoch,
        "global_step": trainer.global_step, "train_sec": round(train_sec, 1), "cuda": cuda,
        # khoá tương thích scoreboard cũ: wer_before/after = bộ VIVOS (test xương sống)
        "wer_before": round(wv_b, 4), "wer_after": round(wv_a, 4),
        "rtf_before": round(rv_b, 4), "rtf_after": round(rv_a, 4),
        "pretrained": f"resume:{Path(nemo_path).name}", "vocab_size": model.tokenizer.vocab_size,
        "nemo_file": (out_nemo.name if not args.no_save else None),
    }
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (run_dir / "status.json").write_text(json.dumps(
        {"state": "ok", "run_id": args.run_id, "wer_vivos_after": results["wer_vivos_after"],
         "wer_cv_after": results["wer_cv_after"]}, ensure_ascii=False))
    print("RESULTS:", json.dumps(results, ensure_ascii=False), flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()
