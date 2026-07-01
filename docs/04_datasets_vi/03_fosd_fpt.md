# 03 — FOSD (FPT Open Speech Dataset)

Bộ dữ liệu giọng nói tiếng Việt do **FPT Corporation** mở công khai năm 2018.

---

## Số liệu

- **Loại:** đọc văn bản (read speech), gom từ 3 sub-dataset.
- **Giờ audio:** ~**30h** (nguồn gốc Mendeley + GitHub FPT).
  - ⚠️ HF mirror `doof-ferb/fpt_fosd` ghi nhầm "100 hours" — **không tin con số đó**. Tin Mendeley ~30h.
- **Số mẫu:** **25.921** câu, có nhãn thời gian bắt đầu/kết thúc.
- **Sample rate / định dạng:** **mp3** + transcript `.txt` (UTF-8). Sample rate gốc cần kiểm chứng (đo file thật).
- **Transcript:** có.
- **License:** **CC BY 4.0** — **cho phép thương mại** (chỉ cần ghi công + giữ copyright notice).

## License — điểm cộng

CC BY 4.0 cho phép **dùng thương mại, sửa, phân phối lại**. Cùng với Common Voice (CC0), đây là một trong
số ít bộ tiếng Việt **an toàn pháp lý cho sản phẩm**.

## Cách tải

```bash
# Cách 1 — HuggingFace mirror (nhanh nhất)
uv run python -c "
from datasets import load_dataset
ds = load_dataset('doof-ferb/fpt_fosd', split='train')
print(ds, ds[0])
"
```

- **Cách 2 — Mendeley Data (bản gốc):** `https://data.mendeley.com/datasets/k9sxg2twv4/4`
  → tải zip gồm `mp3/` + file transcript `.txt`.

## Convert sang manifest NeMo

Bản gốc Mendeley: mỗi dòng transcript có dạng `tên_file.mp3 | transcript | start | end` (kiểm lại định dạng thật).
Pseudo-code cho bản gốc:

```python
import json, os, soundfile as sf

MP3_DIR = "data/raw/fosd/mp3"
TRANS = "data/raw/fosd/transcript.txt"   # mỗi dòng: file | text | ...
OUT = "data/manifests/fosd/all.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(TRANS, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as w:
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        fname, text = parts[0], parts[1]
        if not text:
            continue
        audio = os.path.join(MP3_DIR, fname)
        info = sf.info(audio)
        w.write(json.dumps({
            "audio_filepath": audio,
            "duration": round(info.frames / info.samplerate, 3),
            "text": text,
        }, ensure_ascii=False) + "\n")
```

Bản HuggingFace có sẵn cột `audio` + `transcription` → dùng pattern ghi wav như file VIVOS, đổi tên cột cho khớp.

> Nhớ convert mp3 → **wav 16kHz mono** cho NeMo: `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`.

## Lưu ý

- Cỡ ~30h vừa phải: đủ để đo WER có ý nghĩa, vẫn chạy được trên CPU trong thời gian chấp nhận được (nếu test trên subset).
- Vẫn là giọng đọc → dễ hơn callbot thật.

## ✅ Tự kiểm nhanh

1. FOSD bao nhiêu giờ, và vì sao đừng tin con số trên HF mirror?
2. Điểm mạnh pháp lý của FOSD so với VIVOS?

<details>
<summary>Đáp án</summary>

1. ~**30h** (theo Mendeley/FPT gốc). HF mirror ghi nhầm "100 hours" — sai, có thể copy nhầm từ card khác.
2. FOSD là **CC BY 4.0 (cho phép thương mại)**, còn VIVOS là **CC BY-NC-SA (chỉ phi thương mại)**.

</details>
