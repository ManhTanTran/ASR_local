#!/usr/bin/env python3
"""Kéo dataset ASR tiếng Việt về DGX (team-share) theo manifest datasets.yaml.

Thiết kế cho chạy-qua-đêm không người trông:
  - Resumable: snapshot_download tự nối tiếp file dở; đã xong (.done) thì bỏ qua.
  - Chịu lỗi từng dataset: một cái fail -> log rồi CHẠY TIẾP cái sau, không chết cả batch.
  - Ghi _pull_status.json + log để sáng hôm sau chốt kết quả bằng mắt.

Chạy (trên DGX, env login-shell đã set HF_HOME=/srv/team-share/cache/hf):
  uv run --no-project --with huggingface_hub --with pyyaml \
      python pull.py --stages 1,2,3 --dest /srv/team-share/datasets/asr_vi

Chỉ 1 dataset:      --only bud500
Kéo cả gated:       --include-gated --token <HF_TOKEN>   (cần đã chấp nhận điều khoản)
Chạy thử (in kế hoạch, không tải): --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["datasets"]


def dir_size_gb(p: Path) -> float:
    """du -sb: dung lượng thực trên đĩa (GB), rẻ hơn walk Python."""
    try:
        out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True, timeout=120)
        return int(out.stdout.split()[0]) / 1e9
    except Exception:
        return -1.0


def select(rows: list[dict], stages: set[int], only: str | None, include_gated: bool) -> list[dict]:
    sel = []
    for r in rows:
        if only:
            if r["name"] == only:
                sel.append(r)
            continue
        if r["stage"] not in stages:
            continue
        if r.get("gated") and not include_gated:
            continue  # gated -> bỏ qua ở chế độ auto (cần token + điều khoản)
        sel.append(r)
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(Path(__file__).parent / "datasets.yaml"))
    ap.add_argument("--dest", default="/srv/team-share/datasets/asr_vi")
    ap.add_argument("--stages", default="1,2,3", help="vd '1,2' — nấc curriculum cần kéo")
    ap.add_argument("--only", default=None, help="chỉ kéo 1 dataset theo name")
    ap.add_argument("--include-gated", action="store_true")
    ap.add_argument("--token", default=None, help="HF token cho dataset gated")
    ap.add_argument("--workers", type=int, default=8, help="số file tải song song mỗi dataset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    rows = load_manifest(Path(args.manifest))
    todo = select(rows, stages, args.only, args.include_gated)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest / "_pull_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    plan_gb = sum(r.get("gb", 0) for r in todo)
    print(f"[plan] {len(todo)} dataset, ~{plan_gb:.0f} GB, dest={dest}", flush=True)
    for r in todo:
        print(f"  - {r['name']:20s} stage{r['stage']} ~{r.get('gb',0):>4}GB "
              f"{r['license']:13s} {r['repo_id']}", flush=True)
    if args.dry_run:
        return 0

    for r in todo:
        name, repo = r["name"], r["repo_id"]
        target = dest / name
        done_flag = target / ".done"
        if done_flag.exists():
            print(f"[skip] {name}: đã .done", flush=True)
            status[name] = {**status.get(name, {}), "state": "done", "skipped": True}
            continue

        print(f"\n[pull] {name} <- {repo} (~{r.get('gb',0)}GB) ...", flush=True)
        t0 = time.time()
        try:
            kwargs = dict(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(target),
                max_workers=args.workers,
            )
            # repo đa-ngôn-ngữ (vd google/fleurs) -> chỉ lấy đúng subset VI, tránh kéo cả trăm GB.
            if r.get("allow_patterns"):
                kwargs["allow_patterns"] = r["allow_patterns"]
            if args.token:
                kwargs["token"] = args.token
            snapshot_download(**kwargs)
            dt = time.time() - t0
            gb = dir_size_gb(target)
            done_flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            status[name] = {
                "state": "done", "repo_id": repo, "gb": round(gb, 2),
                "seconds": round(dt), "license": r["license"], "stage": r["stage"],
            }
            print(f"[ok]  {name}: {gb:.1f}GB trong {dt/60:.1f} phút", flush=True)
        except Exception as e:  # noqa: BLE001 — cố tình nuốt để batch chạy tiếp
            dt = time.time() - t0
            status[name] = {"state": "failed", "repo_id": repo, "error": str(e)[:500],
                            "seconds": round(dt)}
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))

    ok = [k for k, v in status.items() if v.get("state") == "done"]
    bad = [k for k, v in status.items() if v.get("state") == "failed"]
    total_gb = sum(v.get("gb", 0) for v in status.values() if isinstance(v.get("gb"), (int, float)))
    print(f"\n===== TỔNG KẾT =====\nOK: {len(ok)} ({', '.join(ok)})\n"
          f"FAIL: {len(bad)} ({', '.join(bad)})\ndung lượng ~{total_gb:.0f}GB\n"
          f"status: {status_path}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
