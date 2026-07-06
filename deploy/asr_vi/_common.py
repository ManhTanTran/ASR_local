"""Hàm chuẩn hoá + đo WER — VENDOR từ asr_lab.common.metrics để package chạy ĐỘC LẬP
(không cần import cả repo train). Phải giữ ĐỒNG BỘ với bản gốc: cùng normalize thì số WER
mới khớp bảng trong README. Nếu sửa metrics gốc, copy lại 3 hàm này.
"""
import re
import unicodedata


def normalize_vi(text: str) -> str:
    """Hạ thường + bỏ dấu câu, GIỮ chữ có dấu (\\w unicode). NFC trước để gộp NFC/NFD.
    Dùng CHUNG cho cả nhãn tham chiếu và output model -> WER công bằng."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def wer(refs, hyps) -> float:
    """WER gộp corpus = tổng (sub+del+ins) / tổng từ tham chiếu (Levenshtein mức từ)."""
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
    """transcribe() trả Hypothesis (có .text) hoặc str tuỳ phiên bản NeMo."""
    return item.text if hasattr(item, "text") else str(item)
