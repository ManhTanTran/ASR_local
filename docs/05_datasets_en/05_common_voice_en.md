# 05 — Common Voice English (license CC0, giọng cộng đồng)

**Vai trò trong lab:** không phải lựa chọn smoke-test đầu tiên (tải rườm rà, là mp3 phải convert), nhưng là
nguồn **license thoáng nhất (CC0)** và giọng đa dạng đời thực. Tốt khi cần dữ liệu sạch license để dùng lâu dài
hoặc đánh giá độ bền với accent.

---

## Mô tả

- **Common Voice** là dự án thu giọng cộng đồng của **Mozilla**: người dùng đọc câu cho sẵn và bình chọn
  (validate) bản thu của người khác. Đa ngôn ngữ; tiếng Anh là phần lớn nhất.
- Trang tải: `https://commonvoice.mozilla.org/datasets` (hoặc Mozilla Data Collective).
- Phát hành theo **phiên bản** (version) tăng dần (vd 17, 18, 19, ...). Tiếng Anh có hàng nghìn giờ.

---

## Số liệu

| Thuộc tính | Giá trị |
| --- | --- |
| Tổng tiếng Anh | Rất lớn (hàng nghìn giờ; mỗi version một con số — **cần kiểm chứng** version cụ thể) |
| Sample rate | 16kHz, mp3 (cần convert sang wav 16kHz cho NeMo) |
| Transcript | Có (file `.tsv`: cột `path` + `sentence`) |
| License | **CC0** (miễn phí, kể cả thương mại — license rộng nhất trong nhóm này) |
| Cách tải | Qua trang Mozilla (đồng ý điều khoản; có thể cần email/tài khoản) hoặc HuggingFace |

**Hai dạng giúp giảm kích thước tải:**
- **Delta segment:** chỉ chứa **dữ liệu mới** so với version trước (cùng cấu trúc tsv) → tải nhẹ hơn nếu
  chỉ cần phần mới. Phù hợp lấy một mẩu nhỏ để test.
- **Spontaneous Speech (English):** một bộ nói **tự nhiên** (không đọc kịch bản) — bản khảo sát thấy con số
  cỡ **~18h thu / ~7h đã validate** (cần kiểm chứng theo bản phát hành). Nhỏ và gần lời nói đời thực.

> Số giờ tiếng Anh thay đổi theo từng version; **đừng tin con số chung chung** — mở trang version cụ thể đọc
> đúng "validated hours" cho tiếng Anh.

---

## Lệnh tải

Common Voice **không có 1 link `wget` công khai ổn định** cho bản đầy đủ (phải bấm đồng ý điều khoản trên web).
Hai cách thực tế:

**Cách A — tải thủ công từ trang Mozilla:**
1. Vào `https://commonvoice.mozilla.org/datasets`.
2. Chọn **English**, chọn version (hoặc **Delta Segment** cho nhẹ), tick đồng ý điều khoản.
3. Tải file `.tar.gz` về `data/raw/common_voice_en/` rồi giải nén.

**Cách B — qua HuggingFace `datasets` (tiện cho subset nhỏ, cần đăng nhập + chấp nhận điều khoản trên HF):**

```python
# uv add datasets soundfile
# Cần: huggingface-cli login  (và bấm "Agree" trên trang dataset)
from datasets import load_dataset

# streaming=True để KHÔNG tải hết — lấy vài mẫu test luồng
ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
                  split="test", streaming=True)
for i, ex in enumerate(ds):
    print(ex["sentence"], ex["audio"]["sampling_rate"])
    if i >= 3:
        break
```

> Tên config version (`common_voice_17_0`, ...) đổi theo bản mới nhất trên HF — **cần kiểm chứng** version
> còn được host. Một số bản cũ đã chuyển sang `legacy-datasets`.

---

## Cấu trúc thư mục (bản tải thủ công)

```
cv-corpus-<ver>-en/
  clips/                # các file .mp3
  validated.tsv         # các bản thu đã được cộng đồng vote đúng (cột path, sentence, ...)
  test.tsv / dev.tsv / train.tsv
```

Để đo WER, **ưu tiên `test.tsv`** (đã chia sẵn) hoặc lọc từ `validated.tsv`.

---

## Convert sang manifest NeMo (mp3 -> wav 16kHz)

```python
# tools/build_cv_manifest.py — đọc test.tsv, convert mp3 -> wav 16kHz, build manifest
import csv, json, subprocess
from pathlib import Path
import librosa

root = Path("data/raw/common_voice_en/cv-corpus-XX-en")  # đổi XX theo version
clips = root / "clips"
wav_dir = Path("data/raw/common_voice_en/wav_test"); wav_dir.mkdir(parents=True, exist_ok=True)
out = Path("data/manifests/common_voice_en"); out.mkdir(parents=True, exist_ok=True)

with open(root / "test.tsv", encoding="utf-8") as f, \
     open(out / "test.jsonl", "w", encoding="utf-8") as fout:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        mp3 = clips / row["path"]
        wav = wav_dir / (Path(row["path"]).stem + ".wav")
        # convert mp3 -> wav mono 16kHz (cần ffmpeg hoặc sox)
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(mp3),
                            "-ar", "16000", "-ac", "1", str(wav)], check=True)
        duration = librosa.get_duration(path=str(wav))
        fout.write(json.dumps({
            "audio_filepath": str(wav.resolve()),
            "duration": round(duration, 3),
            "text": row["sentence"].strip().lower(),   # hạ thường; có thể cần bỏ dấu câu khi tính WER
        }, ensure_ascii=False) + "\n")
print("Đã ghi manifest Common Voice English")
```

> Cần `ffmpeg` (hoặc `sox` có plugin mp3) để decode mp3. Common Voice giữ **dấu câu và chữ hoa** trong
> `sentence` → khi tính WER nên chuẩn hoá (hạ thường + bỏ dấu câu) cho khớp output model.

---

## Lưu ý

- **Tải nặng hơn AN4/LibriSpeech** và phải đồng ý điều khoản → không phải lựa chọn smoke-test đầu tiên.
- Là **mp3** → bắt buộc convert sang wav 16kHz trước khi đưa vào `smoke_transcribe.py`.
- Transcript có dấu câu/chữ hoa, chất lượng không đều như LibriSpeech → WER thường cao hơn, cần chuẩn hoá
  text cẩn thận khi so sánh.
- Điểm mạnh: **CC0** (license thoáng nhất) + giọng đa dạng (nhiều accent) → tốt cho đánh giá độ bền.

---

## ✅ Tự kiểm nhanh

1. License của Common Voice là gì, vì sao đáng chú ý so với LibriSpeech?
2. Hai thứ phải xử lý trước khi đưa Common Voice vào pipeline NeMo là gì?

<details>
<summary>Đáp án</summary>

1. **CC0** — không ràng buộc gì (kể cả không bắt ghi công), thoáng hơn cả CC BY 4.0 của LibriSpeech. Hợp khi
   cần dữ liệu dùng lâu dài/thương mại không vướng license.
2. (1) Tải phải **đồng ý điều khoản trên trang Mozilla/HF** (không có link wget công khai ổn định);
   (2) audio là **mp3 → phải convert sang wav mono 16kHz** trước khi dùng.

</details>
