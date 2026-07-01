"""In report-card cho 1 run từ artifacts/runs/<id>/results.json (không train lại).

Dùng: uv run python -m asr_lab.analytics.report --run-id vivos-fc115m-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


def load(run_id: str) -> dict:
    rj = RUNS_DIR / run_id / "results.json"
    if not rj.exists():
        raise SystemExit(f"Không thấy {rj}")
    return json.loads(rj.read_text())


def report(run_id: str) -> str:
    d = load(run_id)
    wb, wa = d.get("wer_before"), d.get("wer_after")
    delta = (wa - wb) if (wb is not None and wa is not None) else None
    lines = [
        f"== Report-card: {run_id} ==",
        f"  model nền   : {d.get('pretrained','—')}",
        f"  WER trước   : {wb*100:.2f}%" if wb is not None else "  WER trước   : —",
        f"  WER sau     : {wa*100:.2f}%" if wa is not None else "  WER sau     : —",
        f"  ΔWER        : {delta*100:+.2f}%" if delta is not None else "  ΔWER        : —",
        f"  RTF sau     : {d.get('rtf_after','—')}",
        f"  epoch       : {d.get('completed_epochs','?')}/{d.get('epochs','?')}  "
        f"step={d.get('global_step','?')}  batch={d.get('batch','?')}  vocab={d.get('vocab_size','?')}",
        f"  lr          : {d.get('lr','—')}  precision={d.get('precision','—')}  "
        f"freeze_enc={d.get('freeze_encoder','—')}",
        f"  train       : {d.get('train_sec','—')}s  cuda={d.get('cuda','—')}",
        f"  artifact    : artifacts/runs/{run_id}/  (nemo={d.get('nemo_file')})",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    print(report(ap.parse_args().run_id))


if __name__ == "__main__":
    main()
