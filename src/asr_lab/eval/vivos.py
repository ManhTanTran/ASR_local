"""Đo WER của 1 model NeMo ASR trên manifest VIVOS (tiếng Việt) — CPU.

Khác bench_asr (tiếng Anh): chuẩn hoá GIỮ dấu tiếng Việt. Dùng `\\w` (unicode) nên các chữ
có dấu (à, ệ, ữ...) được giữ, chỉ bỏ dấu câu + hạ thường. Nếu strip về ASCII như bench tiếng
Anh thì mọi dấu biến mất -> WER sai bét.

Nhận model theo TÊN pretrained (--model nvidia/...) hoặc file local (.nemo) để so trước/sau
fine-tune bằng CÙNG một harness.

Chạy:
  uv run python -m asr_lab.eval.vivos --manifest data/manifests/vivos_test.jsonl \\
      --model nvidia/nemotron-speech-streaming-en-0.6b [--limit 0] [--batch 8] [--out results.json]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(4)  # tránh treo máy: cap thread

import nemo.collections.asr as nemo_asr  # noqa: E402

from asr_lab.common.metrics import normalize_vi, wer, extract_text  # noqa: E402


def load_model(name_or_path: str):
    """Nạp từ tên pretrained (from_pretrained) hoặc file .nemo local (restore_from)."""
    if name_or_path.endswith(".nemo") or Path(name_or_path).exists():
        return nemo_asr.models.ASRModel.restore_from(name_or_path, map_location="cpu")
    return nemo_asr.models.ASRModel.from_pretrained(model_name=name_or_path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="tên pretrained nvidia/... hoặc đường dẫn .nemo")
    ap.add_argument("--limit", type=int, default=0, help="0 = chạy hết manifest")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="ghi kết quả ra json (tuỳ chọn)")
    ap.add_argument("--show", type=int, default=5, help="số cặp ref/hyp in mẫu để mắt thường soi")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize_vi(r["text"]) for r in rows]
    audio_sec = sum(r["duration"] for r in rows)
    print(f"VIVOS: {len(rows)} utt, {audio_sec/60:.1f} phút audio | model={args.model}\n")

    model = load_model(args.model)
    model.eval()
    t0 = time.perf_counter()
    out = model.transcribe(paths, batch_size=args.batch)
    infer_sec = time.perf_counter() - t0
    hyps = [normalize_vi(extract_text(x)) for x in out]
    score = wer(refs, hyps)
    rtf = infer_sec / max(audio_sec, 1e-9)

    print(f"--- {args.show} cặp mẫu (ref | hyp) ---")
    for r, h in list(zip(refs, hyps))[: args.show]:
        print(f"  REF: {r}")
        print(f"  HYP: {h}\n")
    print(f"==> WER {score*100:.2f}% | infer {infer_sec:.1f}s | RTF {rtf:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": args.model, "manifest": args.manifest, "n_utt": len(rows),
            "audio_sec": round(audio_sec, 1), "wer": round(score, 4),
            "infer_sec": round(infer_sec, 1), "rtf": round(rtf, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> ghi {args.out}")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
