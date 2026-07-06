"""Đo WER ĐỘC LẬP trên 1 hay nhiều manifest test (.jsonl) — tái tạo bảng RESULT.md.

Manifest mỗi dòng: {"audio_filepath": "...", "duration": .., "text": "nhãn"}.
Chuẩn hoá normalize_vi cho CẢ nhãn và output -> khớp số trong README.

Chạy:
  uv run --with "nemo-toolkit[asr]==2.7.3" --with soundfile python eval_wer.py \
      --nemo s3-fc115m-full.nemo \
      --manifests vivos=vivos.test.jsonl cv=common_voice_vi.test.jsonl
  # hoặc trỏ 1 thư mục chứa *.test.jsonl:
  ... --dir /srv/team-share/datasets/asr_vi/_manifests --limit 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from _common import extract_text, normalize_vi, wer


def _eval_one(model, manifest: str, limit: int, batch: int) -> tuple[float, float, int]:
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        return -1.0, -1.0, 0
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize_vi(r["text"]) for r in rows]
    audio_sec = sum(r.get("duration", 0.0) for r in rows) or 1e-9
    t0 = time.perf_counter()
    out = model.transcribe(paths, batch_size=batch)
    dt = time.perf_counter() - t0
    hyps = [normalize_vi(extract_text(x)) for x in out]
    return wer(refs, hyps), dt / audio_sec, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--manifests", nargs="*", default=[], help="cặp label=path")
    ap.add_argument("--dir", default=None, help="thư mục chứa *.test.jsonl (tự lấy label)")
    ap.add_argument("--limit", type=int, default=0, help=">0: mỗi set chỉ N clip (smoke)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    tests: dict[str, str] = {}
    for kv in args.manifests:
        label, _, path = kv.partition("=")
        tests[label] = path
    if args.dir:
        for p in sorted(glob.glob(os.path.join(args.dir, "*.test.jsonl"))):
            tests[os.path.basename(p).replace(".test.jsonl", "")] = p
    if not tests:
        raise SystemExit("cần --manifests label=path ... hoặc --dir")

    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.utils import logging as _nlog
    _nlog.set_verbosity(_nlog.ERROR)   # bảng WER sạch, không lẫn log NeMo

    cuda = torch.cuda.is_available()
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cuda" if cuda else "cpu")
    if cuda:
        model = model.cuda()
    model.eval()
    print(f"# model={os.path.basename(args.nemo)} device={'cuda' if cuda else 'cpu'} "
          f"vocab={model.tokenizer.vocab_size}\n")

    print(f"{'test':24} {'#clip':>7} {'WER%':>8} {'RTF':>8}")
    results = {}
    for label, path in tests.items():
        if not os.path.isfile(path):
            print(f"{label:24} (thiếu file)"); continue
        w, rtf, n = _eval_one(model, path, args.limit, args.batch)
        results[label] = round(w * 100, 2)
        print(f"{label:24} {n:7d} {w*100:8.2f} {rtf:8.4f}", flush=True)

    print("\n# JSON:", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
