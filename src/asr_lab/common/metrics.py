"""Shared text normalization and WER helpers for ASR scripts."""

from __future__ import annotations

import re
from collections.abc import Iterable


def normalize_en(text: str) -> str:
    """Normalize English ASR text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_vi(text: str) -> str:
    """Normalize Vietnamese ASR text while preserving accented characters."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Corpus WER using word-level Levenshtein distance."""
    total_err = 0
    total_words = 0
    for ref, hyp in zip(refs, hyps):
        r = ref.split()
        h = hyp.split()
        d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + cost,
                )
        total_err += d[len(r)][len(h)]
        total_words += len(r)
    return total_err / max(total_words, 1)


def extract_text(item) -> str:
    """Return text from NeMo transcribe output variants."""
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, (list, tuple)) and item:
        return extract_text(item[0])
    return str(item)


def utterance_error_rows(rows: list[dict], hyps: list[str]) -> list[dict]:
    out = []
    for idx, (row, hyp) in enumerate(zip(rows, hyps)):
        ref = normalize_vi(row["text"])
        hyp_norm = normalize_vi(hyp)
        out.append(
            {
                "idx": idx,
                "audio_filepath": row["audio_filepath"],
                "duration": row.get("duration"),
                "ref": ref,
                "hyp": hyp_norm,
                "utt_wer": wer([ref], [hyp_norm]),
            }
        )
    return out
