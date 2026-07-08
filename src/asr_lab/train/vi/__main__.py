"""CLI hệ train/vi.

  python -m asr_lab.train.vi --config configs/s1_clean.yaml
  python -m asr_lab.train.vi --config configs/s2_natural.yaml --resume artifacts/runs/<id>/checkpoints/last.ckpt
  python -m asr_lab.train.vi --config configs/s1_clean.yaml --set train.epochs=1 train.batch_size=2 eval_limit=4
"""
from __future__ import annotations

import argparse

from asr_lab.train.vi.config import load_config
from asr_lab.train.vi.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="file YAML nấc (merge với _base.yaml cùng thư mục)")
    ap.add_argument("--resume", default=None, help="đường dẫn last.ckpt để resume GIỮA run")
    ap.add_argument("--set", nargs="*", default=[], help="override dotlist, vd optim.lr=5e-5 train.epochs=1")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run(cfg, resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
