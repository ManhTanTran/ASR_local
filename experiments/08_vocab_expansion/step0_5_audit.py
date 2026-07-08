"""Step 0.5 — audit nhóm english-heavy trong TRAIN + xem tay câu test (CPU, read-only).

Heuristic 2 mức, tách bạch:
  - EN_FUNC (từ chức năng tiếng Anh: the/of/is/you...) — dấu hiệu CÂU TIẾNG ANH THẬT → ứng viên lọc,
  - từ chứa f/j/w/z đơn lẻ trong câu Việt — loanword HỢP LỆ (đích của expansion) → PHẢI GIỮ.
Lọc theo tỉ lệ EN_FUNC (không theo f/j/w/z) để không vứt oan data loanword quý.

Chạy:
  python3 step0_5_audit.py --manifest <path.jsonl> --ratio 0.3 --samples 6
  python3 step0_5_audit.py --peek-hyp <s3_vss_hyp.jsonl> --words of,whisky,fpt
"""
from __future__ import annotations

import argparse
import json
import random
import re

FJWZ = re.compile(r"[fjwz]")
EN_FUNC = set(
    "the of and to in is you that it for on with as at this have from or by we be are was so "
    "what how can will my your me do not go yes an im its dont thats they he she his her them "
    "then than there here when where which who why would could should about into out up down".split()
)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def en_func_ratio(words: list[str]) -> float:
    return sum(1 for w in words if w in EN_FUNC) / max(len(words), 1)


def flag_en(words: list[str]) -> bool:
    """Câu tiếng Anh thật: ≥2 từ chức năng Anh KHÁC NHAU và tỉ lệ ≥0.3.
    Điều kiện ≥2 distinct để không oan câu Việt ngắn dính 1 chữ trùng ('làng thì to')."""
    hits = [w for w in words if w in EN_FUNC]
    return len(set(hits)) >= 2 and len(hits) / max(len(words), 1) >= 0.3


def audit_manifest(path: str, n_samples: int) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    flagged, kept_fjwz = [], 0
    for r in rows:
        ws = norm(r["text"]).split()
        if not ws:
            continue
        if flag_en(ws):
            flagged.append(r)
        elif FJWZ.search(r["text"].lower()):
            kept_fjwz += 1
    print(f"{path.split('/')[-1]}: n={len(rows)}")
    print(f"  lọc (flag_en, >=2 distinct + ratio 0.3): {len(flagged)} ({len(flagged)/len(rows)*100:.1f}%)")
    print(f"  GIỮ có f/j/w/z (loanword hợp lệ): {kept_fjwz} ({kept_fjwz/len(rows)*100:.1f}%)")
    random.seed(7)
    print(f"  --- {min(n_samples, len(flagged))} mẫu BỊ LỌC (xem tay có oan không):")
    for r in random.sample(flagged, min(n_samples, len(flagged))):
        print(f"    [{r.get('duration', 0):.1f}s] {r['text'][:95]}")


def _lev(r: list[str], h: list[str]) -> int:
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[n][m]


def build_clean(train_in: str, test_in: str, hyp: str | None) -> None:
    """Ghi manifest MỚI (không đè bản gốc): train.clean + train.flagged + test.inscope."""
    # train: lọc câu tiếng Anh nhãn nhiễu
    rows = [json.loads(l) for l in open(train_in, encoding="utf-8") if l.strip()]
    keep = [r for r in rows if not flag_en(norm(r["text"]).split())]
    drop = [r for r in rows if flag_en(norm(r["text"]).split())]
    clean_p = train_in.replace(".train.jsonl", ".train.clean.jsonl")
    flag_p = train_in.replace(".train.jsonl", ".train.flagged.jsonl")
    for p, rs in [(clean_p, keep), (flag_p, drop)]:
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train: giữ {len(keep)}, lọc {len(drop)} -> {clean_p}")

    # test: thước in-scope (cùng rule train + rule combined của step 0)
    rows = [json.loads(l) for l in open(test_in, encoding="utf-8") if l.strip()]
    ins = []
    for r in rows:
        ws = norm(r["text"]).split()
        comb = sum(1 for w in ws if w in EN_FUNC or FJWZ.search(w)) / max(len(ws), 1)
        if not flag_en(ws) and comb < 0.4:
            ins.append(r)
    ins_p = test_in.replace(".test.jsonl", ".test.inscope.jsonl")
    with open(ins_p, "w", encoding="utf-8") as f:
        for r in ins:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test: in-scope {len(ins)}/{len(rows)} -> {ins_p}")

    # WER chính thức trên thước in-scope, tính lại từ hyp đã dump (không cần GPU)
    if hyp:
        hyps = {json.loads(l)["audio_filepath"]: json.loads(l) for l in open(hyp, encoding="utf-8")}
        err = words = miss = 0
        for r in ins:
            h = hyps.get(r["audio_filepath"])
            if not h:
                miss += 1
                continue
            err += _lev(h["ref_norm"].split(), h["hyp_norm"].split())
            words += len(h["ref_norm"].split())
        print(f"WER in-scope chính thức: {err/max(words,1)*100:.2f}% ({len(ins)-miss} câu, thiếu hyp {miss})")


def peek_hyp(path: str, words: list[str], per_word: int) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for kw in words:
        print(f"--- câu in-scope chứa '{kw}':")
        shown = 0
        for r in rows:
            ws = r["ref_norm"].split()
            if kw not in ws:
                continue
            comb = sum(1 for w in ws if w in EN_FUNC or FJWZ.search(w)) / max(len(ws), 1)
            if comb >= 0.4:      # bỏ nhóm english-heavy như định nghĩa in-scope ở step 0
                continue
            print(f"  REF: {r['ref_norm'][:100]}")
            print(f"  HYP: {r['hyp_norm'][:100]}\n")
            shown += 1
            if shown >= per_word:
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--peek-hyp", default=None)
    ap.add_argument("--words", default="of,whisky,fpt")
    ap.add_argument("--per-word", type=int, default=3)
    ap.add_argument("--build-train", default=None, help="manifest train cần lọc")
    ap.add_argument("--build-test", default=None, help="manifest test cần dựng thước in-scope")
    ap.add_argument("--hyp", default=None, help="hyp jsonl để tính WER in-scope chính thức")
    args = ap.parse_args()

    for m in args.manifest:
        audit_manifest(m, args.samples)
        print()
    if args.peek_hyp:
        peek_hyp(args.peek_hyp, args.words.split(","), args.per_word)
    if args.build_train and args.build_test:
        build_clean(args.build_train, args.build_test, args.hyp)


if __name__ == "__main__":
    main()
