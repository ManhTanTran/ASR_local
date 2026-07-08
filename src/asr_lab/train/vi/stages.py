"""Gộp manifest theo cfg.data.train_sets (name + trọng số upsample) -> train.jsonl + val.jsonl cho nấc.
Eval cố định: map nhãn -> file manifest test (KHÔNG train).

Manifest nguồn do build_corpus sinh: <root>/<manifests_dir>/<name>.<split>.jsonl.
Tập không có val riêng -> cắt 5% từ đuôi train làm val.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def build_stage_manifests(cfg, run_dir: Path) -> dict:
    """Trả dict {train, val, eval_fixed:{label:path}} — đường dẫn manifest đã ghi ra run-dir."""
    man_dir = Path(cfg.data.root) / cfg.data.manifests_dir
    out_dir = run_dir / "stage_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lines: list[str] = []
    val_lines: list[str] = []
    per_set = {}
    for entry in cfg.data.train_sets:
        name = entry["name"]
        weight = int(round(float(entry.get("weight", 1))))
        cap = int(entry.get("cap", 0))        # >0: SUBSAMPLE về tối đa cap clip (bộ quá lớn không nuốt bộ nhỏ)
        tr = _read(man_dir / f"{name}.train.jsonl")
        va = _read(man_dir / f"{name}.val.jsonl")
        if not tr:
            # bỏ qua-với-cảnh-báo (bộ chưa build, vd bud500 đang pull) thay vì chết -> config linh hoạt.
            # Nếu KHÔNG bộ nào có data thì raise ở dưới.
            print(f"[stages] CẢNH BÁO: thiếu train manifest '{name}' ({man_dir}/{name}.train.jsonl) -> BỎ QUA", flush=True)
            continue
        if cap and len(tr) > cap:             # lấy đều tay (evenly-spaced) để đại diện toàn bộ, tất định
            step = len(tr) / cap
            tr = [tr[int(i * step)] for i in range(cap)]
        if not va:  # không có val riêng -> cắt 5% đuôi train
            k = max(1, len(tr) // 20)
            va, tr = tr[-k:], tr[:-k]
        train_lines += tr * weight            # upsample theo trọng số (replay tập sạch nhỏ)
        val_lines += va
        per_set[name] = {"train_clips": len(tr), "weight": weight, "cap": cap or None, "val_clips": len(va)}

    if not train_lines:
        raise SystemExit("stages: KHÔNG bộ train_sets nào có manifest — chạy build_corpus trước.")

    train_m = out_dir / "train.jsonl"
    val_m = out_dir / "val.jsonl"
    train_m.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_m.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    eval_fixed = {}
    for label, fname in dict(cfg.data.eval_fixed).items():
        p = man_dir / fname
        if p.exists():
            eval_fixed[label] = str(p)
        else:
            print(f"[stages] CẢNH BÁO: eval_fixed '{label}' thiếu file {p} -> bỏ qua", flush=True)

    print(f"[stages] train={len(train_lines)} (sau upsample) val={len(val_lines)} | "
          f"per_set={per_set} | eval_fixed={list(eval_fixed)}", flush=True)
    return {"train": str(train_m), "val": str(val_m), "eval_fixed": eval_fixed, "per_set": per_set}
