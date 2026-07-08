"""Suy luận ĐỘC LẬP model ASR tiếng Việt (.nemo) — không cần repo train.

Nhận 1 trong 3 dạng input:
  - 1 file wav                       -> in text
  - 1 thư mục (glob *.wav/*.flac)    -> in "path <tab> text" từng file
  - 1 manifest .jsonl (audio_filepath[, text]) -> in text; nếu có 'text' thì kèm WER

Chạy (khuyến nghị uv, xem README):
  uv run --with "nemo-toolkit[asr]==2.7.3" --with soundfile python infer.py \
      --nemo s3-fc115m-full.nemo --audio mau.wav
  ... --audio thu_muc_wav/ --batch 16
  ... --audio test.jsonl                 # có cột text -> in kèm WER

Model chạy CPU được (chậm) hoặc GPU tự động nếu có CUDA.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from _common import extract_text, normalize_vi, wer


def _collect(audio: str) -> tuple[list[str], list[str] | None]:
    """Trả (paths, refs|None). refs != None chỉ khi input là manifest có cột text."""
    if os.path.isdir(audio):
        paths = sorted(glob.glob(os.path.join(audio, "*.wav")) +
                       glob.glob(os.path.join(audio, "*.flac")))
        if not paths:
            sys.exit(f"[infer] thư mục {audio} không có *.wav/*.flac")
        return paths, None
    if audio.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(audio, encoding="utf-8") if l.strip()]
        paths = [r["audio_filepath"] for r in rows]
        refs = [r["text"] for r in rows] if all("text" in r for r in rows) else None
        return paths, refs
    if not os.path.isfile(audio):
        sys.exit(f"[infer] không thấy input: {audio}")
    return [audio], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="đường dẫn model .nemo")
    ap.add_argument("--audio", required=True, help="wav | thư mục | manifest .jsonl")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="in text thô, KHÔNG normalize_vi")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.utils import logging as _nlog
    _nlog.set_verbosity(_nlog.ERROR)   # giữ stdout sạch: chỉ in text, không log NeMo

    cuda = torch.cuda.is_available()
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cuda" if cuda else "cpu")
    if cuda:
        model = model.cuda()
    model.eval()
    print(f"[infer] loaded {os.path.basename(args.nemo)} | device={'cuda' if cuda else 'cpu'} "
          f"| vocab={model.tokenizer.vocab_size}", file=sys.stderr, flush=True)

    paths, refs = _collect(args.audio)
    t0 = time.perf_counter()
    out = model.transcribe(paths, batch_size=args.batch)
    dt = time.perf_counter() - t0
    hyps = [extract_text(x) for x in out]

    for p, h in zip(paths, hyps):
        text = h if args.raw else normalize_vi(h)
        # 1 file -> chỉ in text; nhiều file -> "path <tab> text" để nối script khác
        print(text if len(paths) == 1 else f"{p}\t{text}")

    print(f"[infer] {len(paths)} clip trong {dt:.1f}s", file=sys.stderr, flush=True)
    if refs is not None:
        w = wer([normalize_vi(r) for r in refs], [normalize_vi(h) for h in hyps])
        print(f"[infer] WER (vs cột text) = {w*100:.2f}%", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
