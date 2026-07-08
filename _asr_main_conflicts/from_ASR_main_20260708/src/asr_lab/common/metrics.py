"""Hàm đo dùng chung cho mọi script eval/bench/train: chuẩn hoá text, WER, bóc text.

Tách riêng để cả data/eval/train tham chiếu một nguồn — tránh import chéo giữa các script
(trước đây eval_vivos/finetune import từ bench_asr).
"""

import re
import unicodedata


def normalize_en(text: str) -> str:
    """Chuẩn hoá tiếng Anh: hạ thường + bỏ ký tự không phải [a-z0-9' ] (model PnC có dấu câu/hoa)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_vi(text: str) -> str:
    """Chuẩn hoá tiếng Việt: hạ thường + bỏ dấu câu nhưng GIỮ chữ có dấu (\\w unicode).

    Nếu strip về ASCII như tiếng Anh thì mọi dấu (à, ệ, ữ...) biến mất -> WER sai bét.

    NFC (thêm 2026-07): gộp nhiều nguồn HF trộn NFC/NFD -> cùng 1 từ ra 2 chuỗi codepoint khác nhau
    ("ệ" 1 codepoint vs "e"+2 dấu tổ hợp) -> vocab phình + OOV giả. NFC thống nhất TRƯỚC khi lower/strip.
    Với text đã NFC (VIVOS phần lớn) đây là noop -> số baseline cũ gần như không đổi.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def wer(refs, hyps) -> float:
    """WER gộp toàn corpus = tổng (sub+del+ins) / tổng số từ tham chiếu (Levenshtein mức từ)."""
    total_err, total_words = 0, 0
    for ref, hyp in zip(refs, hyps):
        r, h = ref.split(), hyp.split()
        d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        total_err += d[len(r)][len(h)]
        total_words += len(r)
    return total_err / max(total_words, 1)


def extract_text(item) -> str:
    """transcribe() trả Hypothesis (có .text) hoặc str tuỳ phiên bản/model."""
    return item.text if hasattr(item, "text") else str(item)
