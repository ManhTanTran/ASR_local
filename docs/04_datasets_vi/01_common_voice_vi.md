# 01 — Mozilla Common Voice (tiếng Việt)

Bộ thu âm cộng đồng của Mozilla. Người dùng đọc câu cho sẵn và vote đúng/sai cho nhau.

---

## Số liệu (bản scripted speech, tiếng Việt `vi`)

- **Loại:** đọc văn bản (read speech, câu ngắn).
- **Giờ audio:** bản v24 ghi ~**21.34h thu**, trong đó ~**6.7h đã validated** (cộng đồng vote đúng).
- **Số speaker:** ~**370**.
- **Sample rate / định dạng:** **mp3** (~16kHz–48kHz tùy mic người dùng; coi như 16kHz sau resample).
- **Transcript:** có (câu gốc người đọc).
- **License:** **CC0** (public domain) — **dùng thương mại thoải mái**, đây là điểm mạnh lớn nhất.
- *Số liệu trên theo trang Mozilla Data Collective tại 2026-06; mỗi version tăng dần, cần kiểm lại version mới nhất.*

## License — điểm cộng

CC0 nghĩa là **không ràng buộc** — có thể dùng cho sản phẩm callbot thương mại. Hiếm bộ tiếng Việt nào
có license rộng như vậy → đáng giữ làm bộ test "sạch pháp lý".

## Cách tải

Từ **2025-10**, Mozilla chuyển kho chính sang **Mozilla Data Collective** (cần tạo tài khoản miễn phí):

- Trang: `https://mozilladatacollective.com` → tìm "Common Voice Scripted Speech — Vietnamese".
- Tải file nén (`.tar.gz`) gồm `clips/*.mp3` + các file `.tsv` (`validated.tsv`, `train.tsv`, `test.tsv`, ...).

Mirror cũ trên HuggingFace (các version cũ, vẫn dùng được để thử nhanh):

```bash
# Bản cũ còn truy cập được, cần token HF
uv run python -c "
from datasets import load_dataset
ds = load_dataset('mozilla-foundation/common_voice_12_0', 'vi', split='test')
print(ds)
"
```

> Lưu ý: các repo `mozilla-foundation/common_voice_*` mới (16/17) đã bị làm rỗng — ưu tiên version 11–13
> trên HF, hoặc tải thẳng bản mới nhất từ Mozilla Data Collective.

## Convert sang manifest NeMo

Common Voice dùng cột `path` (tên file mp3) + `sentence` (transcript). Cần đo `duration` bằng `soundfile`/`librosa`.

```python
import json, soundfile as sf, csv, os

CLIPS_DIR = "data/raw/common_voice_vi/clips"
TSV = "data/raw/common_voice_vi/validated.tsv"   # dùng bản validated cho chắc
OUT = "data/manifests/common_voice_vi/validated.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(TSV, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as w:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        audio = os.path.join(CLIPS_DIR, row["path"])
        text = row["sentence"].strip()
        if not text:
            continue                      # bỏ mẫu thiếu transcript — tránh nhiễu WER
        info = sf.info(audio)             # mp3 cần backend đọc được (soundfile/audioread)
        w.write(json.dumps({
            "audio_filepath": audio,
            "duration": round(info.frames / info.samplerate, 3),
            "text": text,
        }, ensure_ascii=False) + "\n")
```

> NeMo thường cần **wav 16kHz mono**. Nên convert mp3 → wav trước bằng `ffmpeg`:
> `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`. Có thể batch bằng vòng lặp shell.

## Lưu ý

- Chỉ ~6.7h validated → bộ **test nhỏ**, hợp smoke-test, chưa đủ để fine-tune nghiêm túc một mình.
- Chất lượng mic không đồng đều (thu tại nhà) → khó hơn VIVOS một chút, gần đời thực hơn.

## ✅ Tự kiểm nhanh

1. Vì sao Common Voice đáng giữ dù chỉ ~6.7h validated?
2. Nên convert audio thế nào trước khi đưa vào NeMo?

<details>
<summary>Đáp án</summary>

1. Vì license **CC0** — dùng thương mại không ràng buộc, hiếm có ở dataset tiếng Việt.
2. Convert mp3 → **wav 16kHz mono** bằng `ffmpeg -ar 16000 -ac 1`, rồi mới sinh manifest.

</details>
