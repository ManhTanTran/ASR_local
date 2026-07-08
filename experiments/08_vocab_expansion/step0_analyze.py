"""Step 0 — phân tích lỗi f/j/w/z từ file hyp jsonl (CPU, chạy lại thoải mái, không cần GPU).

Trả lời 3 câu hỏi của step 0:
  1. WER subset câu chứa f/j/w/z cao hơn phần còn lại bao nhiêu (trần do vocab lớn cỡ nào)?
  2. Lỗi rơi đúng vào TỪ chứa f/j/w/z chiếm bao nhiêu % tổng lỗi?
  3. Model phiên âm các từ đó thành gì — biến thể ổn định đủ để map từ điển (giải pháp A) không?

Chạy:
  python3 step0_analyze.py --hyp s3_vss_hyp.jsonl --top 30
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

FJWZ = re.compile(r"[fjwz]")


def align(r: list[str], h: list[str]) -> list[tuple[str, str, str]]:
    """Levenshtein mức từ + backtrace -> list (op, ref_word, hyp_word), op ∈ eq/sub/del/ins."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            ops.append(("eq" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", r[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", h[j - 1]))
            j -= 1
    return ops[::-1]


def corpus_wer(pairs: list[tuple[str, str]]) -> tuple[float, int, int]:
    """WER gộp = tổng lỗi / tổng từ ref (đồng bộ _common.wer)."""
    err = words = 0
    for ref, hyp in pairs:
        ops = align(ref.split(), hyp.split())
        err += sum(1 for op, _, _ in ops if op != "eq")
        words += len(ref.split())
    return err / max(words, 1), err, words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="jsonl từ step0_transcribe.py")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--samples", type=int, default=0, help=">0: in N câu lỗi nặng nhất subset f/j/w/z")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.hyp, encoding="utf-8") if l.strip()]
    with_f = [(r["ref_norm"], r["hyp_norm"]) for r in rows if FJWZ.search(r["ref_norm"])]
    no_f = [(r["ref_norm"], r["hyp_norm"]) for r in rows if not FJWZ.search(r["ref_norm"])]

    print(f"# file={args.hyp}  n={len(rows)} (fjwz={len(with_f)}, khác={len(no_f)})\n")

    # --- Câu hỏi 1: WER theo subset ---
    w_all, e_all, n_all = corpus_wer([(r["ref_norm"], r["hyp_norm"]) for r in rows])
    w_f, e_f, n_f = corpus_wer(with_f)
    w_o, e_o, n_o = corpus_wer(no_f)
    print("## 1. WER theo subset")
    print(f"{'subset':16} {'câu':>6} {'từ ref':>8} {'lỗi':>7} {'WER%':>7}")
    print(f"{'toàn bộ':16} {len(rows):6d} {n_all:8d} {e_all:7d} {w_all*100:7.2f}")
    print(f"{'có f/j/w/z':16} {len(with_f):6d} {n_f:8d} {e_f:7d} {w_f*100:7.2f}")
    print(f"{'không f/j/w/z':16} {len(no_f):6d} {n_o:8d} {e_o:7d} {w_o*100:7.2f}\n")

    # --- Câu hỏi 2 + 3: lỗi tại từ f/j/w/z + biến thể hyp ---
    # Gộp sub + tối đa 2 ins liền sau thành "cụm hyp" (loanword hay bị chẻ: facebook -> phây búc).
    fjwz_err = 0          # lỗi (sub/del) tại từ ref chứa f/j/w/z
    fjwz_occ = 0          # tổng lần xuất hiện từ f/j/w/z
    fjwz_correct = 0      # số lần đúng nguyên văn (kỳ vọng 0 vì vocab không sinh nổi)
    variants: dict[str, Counter] = defaultdict(Counter)
    for ref, hyp in with_f:
        ops = align(ref.split(), hyp.split())
        k = 0
        while k < len(ops):
            op, rw, hw = ops[k]
            if rw and FJWZ.search(rw):
                fjwz_occ += 1
                if op == "eq":
                    fjwz_correct += 1
                else:
                    fjwz_err += 1
                    span = [hw] if hw else []
                    for nxt in ops[k + 1 : k + 3]:      # nuốt tối đa 2 ins liền sau
                        if nxt[0] == "ins":
                            span.append(nxt[2])
                        else:
                            break
                    variants[rw][" ".join(span) if span else "∅"] += 1
            k += 1

    print("## 2. Lỗi tại từ chứa f/j/w/z")
    print(f"tổng lần xuất hiện từ f/j/w/z : {fjwz_occ}")
    print(f"  đúng nguyên văn             : {fjwz_correct}")
    print(f"  lỗi (sub/del)               : {fjwz_err}")
    print(f"tổng lỗi toàn corpus          : {e_all}")
    print(f"→ lỗi thuộc từ f/j/w/z        : {fjwz_err/max(e_all,1)*100:.1f}% tổng lỗi\n")

    # --- Trần cứu được bằng map từ điển: biến thể top-1 per từ, LOẠI ∅ ---
    # ∅ = từ bị xoá hẳn trong hyp → map từ điển không có gì để map, không cứu được.
    recoverable = 0
    for c in variants.values():
        non_del = [(v, k) for v, k in c.most_common() if v != "∅"]
        if non_del:
            recoverable += non_del[0][1]
    deleted = sum(c.get("∅", 0) for c in variants.values())
    print("## 3. Biến thể hyp per từ (ước trần recall của map từ điển)")
    print(f"lỗi dạng xoá hẳn (∅, không map được) : {deleted}/{fjwz_err}")
    print(f"map biến thể top-1 (loại ∅) → cứu tối đa {recoverable}/{fjwz_err} lỗi f/j/w/z "
          f"({recoverable/max(fjwz_err,1)*100:.1f}%)")
    print("(vẫn là trần lạc quan: coi biến thể không đụng từ Việt thật + map không đè nhầm chỗ đúng)\n")

    print(f"{'từ ref':>18} {'n_lỗi':>6}  top biến thể hyp (số lần)")
    top_words = sorted(variants.items(), key=lambda kv: -sum(kv[1].values()))[: args.top]
    for w, c in top_words:
        tops = " | ".join(f"{v}×{k}" for v, k in c.most_common(3))
        print(f"{w:>18} {sum(c.values()):6d}  {tops}")

    # --- Mẫu câu lỗi nặng nhất trong subset f/j/w/z: nhìn tận mắt hiện tượng ---
    if args.samples > 0:
        scored = []
        for ref, hyp in with_f:
            ops = align(ref.split(), hyp.split())
            err = sum(1 for op, _, _ in ops if op != "eq")
            scored.append((err / max(len(ref.split()), 1), err, ref, hyp))
        scored.sort(reverse=True)
        print(f"\n## 4. {args.samples} câu lỗi nặng nhất (subset f/j/w/z)")
        for uwer, err, ref, hyp in scored[: args.samples]:
            print(f"\n[WER câu {uwer*100:.0f}%, {err} lỗi]")
            print(f"  REF: {ref}")
            print(f"  HYP: {hyp}")


if __name__ == "__main__":
    main()
