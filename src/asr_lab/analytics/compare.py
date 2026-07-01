"""So 2 run trên cùng chuẩn (đọc results.json) -> ΔWER/ΔRTF + verdict 3 cổng. Không train lại.

Dùng: uv run python -m asr_lab.analytics.compare --base vivos-fc115m-v1 --cand vivos-fc115m-v2norm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_lab.analytics.verdict import verdict

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


def _load(run_id: str) -> dict:
    rj = RUNS_DIR / run_id / "results.json"
    if not rj.exists():
        raise SystemExit(f"Không thấy {rj} — run xong chưa / pull về chưa?")
    return json.loads(rj.read_text())


def compare(base_id: str, cand_id: str) -> str:
    b, c = _load(base_id), _load(cand_id)
    v = verdict(b["wer_after"], c["wer_after"], b.get("rtf_after"), c.get("rtf_after"))
    lines = [
        f"== So sánh: base={base_id}  vs  cand={cand_id} ==",
        f"  WER base : {b['wer_after']*100:.2f}%   WER cand : {c['wer_after']*100:.2f}%",
        f"  ΔWER     : {v['d_wer_pct']:+.2f}%  (âm = cand tốt hơn)",
        f"  RTF      : base {b.get('rtf_after','—')}  ->  cand {c.get('rtf_after','—')}",
        "  cổng     : "
        f"dấu={'✔' if v['gate_sign'] else '✘'}  "
        f"tin-cậy={'✔' if v['gate_conf'] else '✘'}  "
        f"không-hồi-quy-RTF={'✔' if v['gate_rtf'] else '✘'}",
        f"  VERDICT  : {v['verdict']}",
        f"  (ghi chú CI: {v['note_ci']})",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    a = ap.parse_args()
    print(compare(a.base, a.cand))


if __name__ == "__main__":
    main()
