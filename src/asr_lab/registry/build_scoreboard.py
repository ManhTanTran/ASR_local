"""Sinh lại experiments/_SCOREBOARD.md từ artifacts/runs/*/results.json.

Quy ước (phỏng numerai lab_v2): KHÔNG gõ tay số vào scoreboard — luôn build từ artifact để mọi con số
truy ngược được về run-dir. Run smoke (không có wer_after) bị bỏ qua.

Dùng:
    uv run python -m asr_lab.registry.build_scoreboard            # ghi đè _SCOREBOARD.md
    uv run python -m asr_lab.registry.build_scoreboard --print    # chỉ in, không ghi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# src/asr_lab/registry/build_scoreboard.py -> repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
SCOREBOARD = REPO_ROOT / "experiments" / "_SCOREBOARD.md"

# Mốc tham chiếu ngoài (sense từ web 2026-06) — để xếp run của mình vào thang. Chi tiết +
# nguồn: insight/external/01_vivos_sota_survey.md. KHÔNG phải run của mình, đánh dấu rõ.
EXTERNAL_VIVOS = [
    ("ChunkFormer-CTC-large-vie", "110M", "4,18", "SOTA cỡ nhỏ (3.000h)"),
    ("PhoWhisper-large", "1,55B", "4,67", "Whisper 844h VI"),
    ("wav2vec2-large-vi-vlsp2020", "~300M", "8,61", "baseline mạnh"),
    ("wav2vec2-base-vietnamese-250h", "~95M", "10,83", "baseline phổ biến"),
]


def load_runs() -> list[dict]:
    rows = []
    for rj in sorted(RUNS_DIR.glob("*/results.json")):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        if d.get("wer_after") is None:  # bỏ smoke / run lỗi
            continue
        d["_run_id"] = rj.parent.name
        rows.append(d)
    return rows


def fmt_pct(x) -> str:
    return f"{x*100:.2f}".replace(".", ",") if isinstance(x, (int, float)) else "—"


def short_model(name: str) -> str:
    return name.split("/")[-1] if name else "—"


def render(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r.get("wer_after", 9))
    out = ["# 📊 _SCOREBOARD — mọi run fine-tune (sinh tự động)",
           "",
           "> Sinh bằng `uv run python -m asr_lab.registry.build_scoreboard` từ "
           "`artifacts/runs/*/results.json`. **Đừng sửa tay** — chạy lại script.",
           "",
           "## Run của mình (sắp theo WER sau, thấp = tốt)",
           "",
           "| run_id | model nền | WER trước | **WER sau** | RTF sau | epoch | batch | vocab | step |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| `{r['_run_id']}` | {short_model(r.get('pretrained',''))} | "
            f"{fmt_pct(r.get('wer_before'))}% | **{fmt_pct(r.get('wer_after'))}%** | "
            f"{r.get('rtf_after','—')} | {r.get('completed_epochs', r.get('epochs','—'))} | "
            f"{r.get('batch','—')} | {r.get('vocab_size','—')} | {r.get('global_step','—')} |")
    out += ["",
            "## Mốc ngoài trên VIVOS (sensing — KHÔNG phải run của mình)",
            "",
            "| model | tham số | WER | ghi chú |",
            "| --- | --- | --- | --- |"]
    for name, p, wer, note in EXTERNAL_VIVOS:
        out.append(f"| {name} | {p} | {wer}% | {note} |")
    out += ["",
            "## Thang baseline VIVOS (đọc verdict theo bậc)",
            "",
            "1. **Sàn** English zero-shot ~100% · 2. **Tối thiểu** fine-tune đầu 20,37% · "
            "3. **KPI cộng đồng** wav2vec2-base ~10,8% · 4. **Sao bắc đẩu** ChunkFormer 4,18%.",
            ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="only_print", action="store_true")
    args = ap.parse_args()
    rows = load_runs()
    md = render(rows)
    print(md)
    if not args.only_print:
        SCOREBOARD.parent.mkdir(parents=True, exist_ok=True)
        SCOREBOARD.write_text(md, encoding="utf-8")
        print(f"\n-> ghi {SCOREBOARD} ({len(rows)} run)")


if __name__ == "__main__":
    main()
