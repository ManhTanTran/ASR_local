"""Eval-only: đo WER một checkpoint .nemo trên suite eval_fixed (KHÔNG train, read-only).

Dùng để lấp bức tranh cross-domain cho model đã train xong — vd đo s1-fc115m-full trên đủ
6 test (vivos/cv/fleurs/vlsp/lsvsc/fosd) trong khi run gốc chỉ snapshot cv+fleurs.

Tái dùng ĐÚNG hàm eval_wer của runner để con số nhất quán với lúc train.

  python -m asr_lab.eval_suite --nemo /srv/team-share/models/asr_vi/s1-fc115m-full.nemo \
      --config configs/_base.yaml --out s1_suite.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from asr_lab.train.vi.runner import eval_wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="đường dẫn .nemo cần đo")
    ap.add_argument("--config", default="configs/_base.yaml", help="lấy data.root + eval_fixed từ đây")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số clip mỗi test (0 = hết)")
    ap.add_argument("--out", default=None, help="ghi kết quả JSON ra file")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    man_dir = Path(cfg.data.root) / cfg.data.manifests_dir
    cuda = torch.cuda.is_available()

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location=("cuda" if cuda else "cpu"))
    if cuda:
        model = model.cuda()   # đảm bảo cả model trên GPU trước khi transcribe

    res = {}
    for label, fname in dict(cfg.data.eval_fixed).items():
        p = man_dir / fname
        if not p.exists():
            res[label] = {"skip": "file thiếu"}
            print(f"[skip] {label}: {p} thiếu (bộ chưa build)", flush=True)
            continue
        w, rtf = eval_wer(model, str(p), args.limit, args.batch)
        res[label] = {"wer": round(w, 4), "rtf": round(rtf, 4)}
        print(f"[{label}] WER={w*100:.2f}%  RTF={rtf:.4f}", flush=True)

    out = {"nemo": args.nemo, "vocab_size": int(model.tokenizer.vocab_size), "results": res}
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("EVAL_SUITE:", json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
