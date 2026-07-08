"""Build SentencePiece BPE tiếng Việt từ manifest train gộp + cổng OOV (chặn <unk> trước GPU).

Text trong manifest đã normalize_vi (NFC+lower+bỏ dấu câu) + lọc whitelist bởi build_corpus,
nên char_coverage=1.0 an toàn. unk_id=0, không bos/eos/pad (chuẩn ASR-BPE).
"""
from __future__ import annotations

import json
from pathlib import Path

from asr_lab.common.metrics import normalize_vi


def build_tokenizer(train_manifest: str, out_dir: Path, vocab_size: int, tok_type: str = "bpe") -> Path:
    import sentencepiece as spm
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus.txt"
    with corpus.open("w", encoding="utf-8") as w:
        for ln in open(train_manifest, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                w.write(normalize_vi(json.loads(ln)["text"]) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(out_dir / "tokenizer"),
        vocab_size=vocab_size, model_type=tok_type, character_coverage=1.0,
        bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
    )
    # vocab.txt để soi nhanh
    with open(out_dir / "tokenizer.vocab", encoding="utf-8") as f, \
            (out_dir / "vocab.txt").open("w", encoding="utf-8") as w:
        for ln in f:
            w.write(ln.split("\t")[0] + "\n")
    print(f"[tokenizer] {tok_type} vocab={vocab_size} -> {out_dir}", flush=True)
    return out_dir


def assert_no_oov(train_manifest: str, tok_dir: Path, max_rate: float = 1e-4) -> None:
    """Cổng: nhãn train qua tokenizer KHÔNG được sinh <unk> quá ngưỡng. Dừng trước khi tốn GPU."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(Path(tok_dir) / "tokenizer.model"))
    unk = sp.unk_id()
    n_unk = n_tok = 0
    samples = []
    for ln in open(train_manifest, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        ids = sp.encode(json.loads(ln)["text"], out_type=int)
        u = sum(1 for i in ids if i == unk)
        if u and len(samples) < 5:
            samples.append(json.loads(ln)["text"][:80])
        n_unk += u
        n_tok += len(ids)
    rate = n_unk / max(n_tok, 1)
    print(f"[cổng OOV] rate={rate:.4%} ({n_unk}/{n_tok})", flush=True)
    if rate > max_rate:
        for s in samples:
            print("  nhãn sinh <unk>:", s)
        raise SystemExit(f"OOV={rate:.4%} > {max_rate:.4%} — lệch vocab vs nhãn. DỪNG.")
