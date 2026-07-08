"""Tạo manifest _clean đặt tên đúng pattern stages (<name>.train/val.jsonl) + lọc val.
Không đụng file gốc. Chạy trên DGX."""
import json
import re
import shutil

EN_FUNC = set(
    "the of and to in is you that it for on with as at this have from or by we be are was so "
    "what how can will my your me do not go yes an im its dont thats they he she his her them "
    "then than there here when where which who why would could should about into out up down".split()
)
M = "/srv/team-share/datasets/asr_vi/_manifests"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def flag_en(ws):
    h = [w for w in ws if w in EN_FUNC]
    return len(set(h)) >= 2 and len(h) / max(len(ws), 1) >= 0.3


# train: đã có bản .train.clean.jsonl -> copy sang tên _clean.train.jsonl
shutil.copyfile(f"{M}/vietsuperspeech.train.clean.jsonl", f"{M}/vietsuperspeech_clean.train.jsonl")
n_train = sum(1 for _ in open(f"{M}/vietsuperspeech_clean.train.jsonl", encoding="utf-8"))

# val: lọc mới cùng rule
rows = [json.loads(l) for l in open(f"{M}/vietsuperspeech.val.jsonl", encoding="utf-8") if l.strip()]
keep = [r for r in rows if not flag_en(norm(r["text"]).split())]
with open(f"{M}/vietsuperspeech_clean.val.jsonl", "w", encoding="utf-8") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train_clean: {n_train} dòng -> vietsuperspeech_clean.train.jsonl")
print(f"val: {len(rows)} -> giữ {len(keep)} (lọc {len(rows) - len(keep)}) -> vietsuperspeech_clean.val.jsonl")
