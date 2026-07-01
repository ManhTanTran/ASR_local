# 05 — VietBud500

Bộ lớn (~500h) do nhóm **VietAI** tổng hợp từ nhiều nguồn audio công khai (podcast, sách nói, du lịch, ẩm thực...).

---

## Số liệu

- **Loại:** hỗn hợp — **đọc + podcast + sách nói**, có cả 3 giọng vùng miền (Bắc/Trung/Nam).
- **Giờ audio:** ~**500h**.
- **Số mẫu:** 634.158 train + 7.500 validation + 7.500 test.
- **Sample rate / định dạng:** **16kHz**. Dung lượng ~**98GB**.
- **Transcript:** có.
- **License:** **CC BY-NC-SA 4.0** → **chỉ phi thương mại** (thẻ có nhắc Apache 2.0 ở copyright notice nhưng license dataset là NC — coi như chỉ nghiên cứu).
- **Truy cập:** cần **đồng ý điều khoản** trên HuggingFace trước khi tải.

## License — cảnh báo

NC → **không dùng cho sản phẩm thương mại**. Hợp để **train/fine-tune thử nghiệm và đo WER nội bộ**.

## Cách tải

```bash
# Duyệt thử bằng streaming trước (KHÔNG tải 98GB ngay)
uv run python -c "
from datasets import load_dataset
ds = load_dataset('linhtran92/viet_bud500', split='test', streaming=True)
print(next(iter(ds)))
"
```

- Tải split `test` (7.500 mẫu) là đủ để đo WER; **không cần** tải toàn bộ 98GB cho việc test.
- Cần đăng nhập HuggingFace (`huggingface-cli login`) + bấm đồng ý điều khoản trên trang dataset.

## Convert sang manifest NeMo

Pattern giống VIVOS (cột `audio` + cột transcript — kiểm tên cột thật bằng `print(ds.features)`):

```python
import json, os, soundfile as sf
from datasets import load_dataset

OUT = "data/manifests/vietbud500/test.jsonl"
RAW = "data/raw/vietbud500/wav_test"
os.makedirs(os.path.dirname(OUT), exist_ok=True); os.makedirs(RAW, exist_ok=True)

ds = load_dataset("linhtran92/viet_bud500", split="test")   # 7.500 mẫu
TEXT_COL = "transcription"   # đổi cho khớp ds.features nếu khác
for i, row in enumerate(ds):
    arr, sr = row["audio"]["array"], row["audio"]["sampling_rate"]
    text = (row.get(TEXT_COL) or "").strip()
    if not text:
        continue
    path = os.path.join(RAW, f"vb_{i:06d}.wav")
    sf.write(path, arr, sr)
    with open(OUT, "a", encoding="utf-8") as w:
        w.write(json.dumps({
            "audio_filepath": path,
            "duration": round(len(arr) / sr, 3),
            "text": text,
        }, ensure_ascii=False) + "\n")
```

## Lưu ý

- **Lớn + nặng** → chỉ dùng split `test` cho việc đo WER trên CPU; train cần GPU.
- Đa dạng nguồn (podcast/sách) nên gần văn nói hơn các bộ tin tức, nhưng vẫn không phải điện thoại 8kHz.

## ✅ Tự kiểm nhanh

1. Có cần tải hết 98GB để đo WER không?
2. License VietBud500 cho phép thương mại không?

<details>
<summary>Đáp án</summary>

1. **Không** — chỉ cần split `test` (7.500 mẫu). Dùng `streaming=True` để duyệt thử trước.
2. **Không** — CC BY-NC-SA là phi thương mại.

</details>
