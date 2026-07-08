"""Test normalize_vi — trọng tâm: bước NFC mới thêm (2026-07) thống nhất NFC/NFD.

Chạy: uv run python -m pytest tests/test_normalize_vi.py -q
Hoặc:  uv run python tests/test_normalize_vi.py   (self-run, không cần pytest)
"""
import unicodedata

from asr_lab.common.metrics import normalize_vi


def test_nfc_unifies_nfd_and_nfc():
    """Cùng 1 từ ở dạng NFD (dấu tổ hợp) và NFC (1 codepoint) -> normalize ra GIỐNG nhau."""
    nfc = "nghệ"                                  # dạng tổ hợp sẵn (composed)
    nfd = unicodedata.normalize("NFD", nfc)       # tách dấu ra (decomposed)
    assert nfc != nfd                              # đúng là 2 chuỗi codepoint khác nhau
    assert normalize_vi(nfc) == normalize_vi(nfd)  # sau normalize thì bằng nhau
    # output ở dạng NFC (1 codepoint / ký tự có dấu)
    assert normalize_vi(nfd) == unicodedata.normalize("NFC", "nghệ")


def test_lower_and_strip_punct_keep_digits():
    assert normalize_vi("Tuy nhiên, năm 2020!") == "tuy nhiên năm 2020"


def test_keeps_vietnamese_diacritics():
    assert normalize_vi("Ữ Ệ À đường") == "ữ ệ à đường"


def test_collapse_whitespace():
    assert normalize_vi("  a   b\tc\n") == "a b c"


if __name__ == "__main__":
    # self-run không cần pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"== {len(fns)} test PASS ==")
