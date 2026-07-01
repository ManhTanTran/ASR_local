# 09 — PhoAudiobook

Bộ **sách nói (audiobook)** rất lớn do nhóm liên quan VinAI curate, gốc dùng cho **TTS (text-to-speech)**
nhưng vì có cặp audio–text nên dùng được cho ASR.

> Kèm bài ACL 2025. HuggingFace: `thivux/phoaudiobook`

---

## Số liệu

- **Loại:** sách nói / đọc (audiobook), giọng chất lượng cao.
- **Giờ audio:** ~**940h** (bản chuẩn) + ~554h đoạn ngắn augment ⇒ tổng ~1.494h theo thẻ.
- **Số speaker (train):** ~710.
- **Sample rate / định dạng:** lưu **parquet**; sample rate **cần kiểm chứng**.
- **Transcript:** có (vốn là dữ liệu TTS nên cặp audio–text rất sạch).
- **License:** **license nghiên cứu tùy chỉnh** — *chỉ dùng cho nghiên cứu/giáo dục*, **cấm phân phối lại**
  bản gốc hoặc bản sửa đổi; phải cite bài ACL 2025 khi công bố.

## Cảnh báo license — quan trọng

License này **chặt**: chỉ nghiên cứu/giáo dục, **không thương mại**, **không phân phối lại**.
→ Không dùng cho sản phẩm callbot thương mại. Phải **đăng nhập HuggingFace + đồng ý điều khoản** mới tải được.

## Cách tải

```bash
# Cần huggingface-cli login + bấm đồng ý điều khoản trên trang dataset
uv run python -c "
from datasets import load_dataset
ds = load_dataset('thivux/phoaudiobook', split='train', streaming=True)
print(next(iter(ds)))
"
```

## Convert sang manifest NeMo

Pattern giống VIVOS. Vì là audiobook, transcript thường rất sạch → ground-truth tốt, nhưng đoạn có thể dài.

## Lưu ý

- Chất lượng cặp audio–text cao (vốn cho TTS) → tốt nếu cần **train** với nhãn sạch.
- Nhưng **license chỉ nghiên cứu** + rất nặng → không hợp làm bộ smoke-test mặc định; cân nhắc chỉ khi nghiên cứu.
- Là audiobook đọc → **xa domain callbot** (hội thoại/điện thoại).

## ✅ Tự kiểm nhanh

1. PhoAudiobook hợp cho mục tiêu nào, không hợp mục tiêu nào?
2. Rào cản license chính là gì?

<details>
<summary>Đáp án</summary>

1. Hợp **train với nhãn sạch khi nghiên cứu**; **không hợp** smoke-test nhanh (nặng) và **không hợp** sản phẩm thương mại.
2. License **chỉ nghiên cứu/giáo dục, cấm phân phối lại, không thương mại** — phải đồng ý điều khoản mới tải.

</details>
