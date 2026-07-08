"""Step 0 — dump hypothesis từng câu của S3 trên các manifest test (GPU infer nhẹ, read-only).

Chạy trên DGX trong venv repo:
  cd /srv/team-share/projects/nvidia_asr_nemo
  PYTHONPATH=deploy/asr_vi .venv/bin/python -u experiments/08_vocab_expansion/step0_transcribe.py \
      --nemo /srv/team-share/models/asr_vi/s3-fc115m-full.nemo \
      --manifests vss=/srv/team-share/datasets/asr_vi/_manifests/vietsuperspeech.test.jsonl \
                  fleurs=/srv/team-share/datasets/asr_vi/_manifests/fleurs_vi.test.jsonl \
      --outdir /srv/team-share/datasets/asr_vi/_runs/probe_fjwz
"""
from __future__ import annotations

import argparse
import json
import os
import time

from _common import extract_text, normalize_vi  # PYTHONPATH=deploy/asr_vi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--manifests", nargs="+", help="cặp label=path")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help=">0: chỉ N clip đầu (smoke)")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.utils import logging as _nlog
    _nlog.set_verbosity(_nlog.ERROR)

    cuda = torch.cuda.is_available()
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cuda" if cuda else "cpu")
    if cuda:
        model = model.cuda()
    model.eval()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"# model={os.path.basename(args.nemo)} device={'cuda' if cuda else 'cpu'} "
          f"vocab={model.tokenizer.vocab_size}", flush=True)

    for kv in args.manifests:
        label, _, path = kv.partition("=")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if args.limit > 0:
            rows = rows[: args.limit]
        t0 = time.perf_counter()
        out = model.transcribe([r["audio_filepath"] for r in rows], batch_size=args.batch)
        dt = time.perf_counter() - t0
        outp = os.path.join(args.outdir, f"s3_{label}_hyp.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for r, o in zip(rows, out):
                f.write(json.dumps({
                    "audio_filepath": r["audio_filepath"],
                    "duration": r.get("duration", 0.0),
                    "ref": r["text"],
                    "ref_norm": normalize_vi(r["text"]),
                    "hyp_norm": normalize_vi(extract_text(o)),
                }, ensure_ascii=False) + "\n")
        print(f"DONE {label}: {len(rows)} câu, {dt/60:.1f} phút -> {outp}", flush=True)

    print("STEP0_TRANSCRIBE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
