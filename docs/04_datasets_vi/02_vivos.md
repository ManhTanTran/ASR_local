# 02 — VIVOS

Bộ corpus đọc tiếng Việt do **AILAB, Đại học Quốc gia TP.HCM** phát hành. Nhỏ, sạch, kinh điển cho ASR Việt.

---

## Số liệu

- **Loại:** đọc văn bản (read speech), phòng yên tĩnh, mic tốt.
- **Giờ audio:** ~**15.7h** (train ~14:55, test ~0:45).
- **Số câu (utterance):** 11.660 train + 760 test.
- **Số speaker:** **65** (46 train, 19 test).
- **Sample rate / định dạng:** **16kHz**, **WAV**.
- **Transcript:** có (file `prompts.txt`).
- **License:** **CC BY-NC-SA 4.0** → **chỉ phi thương mại** (NC = non-commercial).

## License — cảnh báo

NC nghĩa là **không được dùng cho sản phẩm thương mại**. Dùng để **smoke-test / nghiên cứu / đo WER nội bộ**
thì hợp lệ. Nếu sau này callbot bán ra thị trường, **không** được dùng VIVOS để train model thương mại.

## Cách tải

```bash
# Cách nhanh nhất — HuggingFace, có cả split test ~45 phút
uv run python -c "
from datasets import load_dataset
ds = load_dataset('AILAB-VNUHCM/vivos', split='test')   # ~760 câu, đủ smoke-test
print(ds[0])
"
```

Bản gốc (link tự host của AILAB) đôi khi chậm/chết — ưu tiên mirror HuggingFace `AILAB-VNUHCM/vivos`.

## Convert sang manifest NeMo

Bản HuggingFace trả về cột `audio` (có `array` + `sampling_rate` + `path`) và `sentence`.

```python
import json, os, soundfile as sf
from datasets import load_dataset

OUT = "data/manifests/vivos/test.jsonl"
RAW = "data/raw/vivos/test_wav"        # nơi ghi wav ra để NeMo đọc theo path
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(RAW, exist_ok=True)

ds = load_dataset("AILAB-VNUHCM/vivos", split="test")
with open(OUT, "w", encoding="utf-8") as w:
    for i, row in enumerate(ds):
        arr = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]      # 16000
        path = os.path.join(RAW, f"vivos_test_{i:05d}.wav")
        sf.write(path, arr, sr)                  # ghi wav thật ra đĩa
        w.write(json.dumps({
            "audio_filepath": path,
            "duration": round(len(arr) / sr, 3),
            "text": row["sentence"].strip(),
        }, ensure_ascii=False) + "\n")
```

## Lưu ý

- Đây là bộ **đầu tiên nên dùng** để dựng pipeline đo WER — nhỏ, nhanh trên CPU.
- Vì là đọc trong phòng sạch nên WER sẽ **tốt giả tạo** so với callbot thật. Đừng vội mừng với con số đẹp.

## ✅ Tự kiểm nhanh

1. Split nào của VIVOS hợp để smoke-test nhanh nhất?
2. Vì sao WER trên VIVOS không phản ánh đúng callbot?

<details>
<summary>Đáp án</summary>

1. Split **test** — chỉ ~45 phút / 760 câu, chạy CPU nhanh.
2. VIVOS là **đọc trong phòng sạch, mic tốt** → quá dễ so với giọng điện thoại nhiễu, băng hẹp của callbot.

</details>
