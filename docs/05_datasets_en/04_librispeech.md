# 04 — LibriSpeech (bộ test chuẩn của ASR tiếng Anh)

**Vai trò trong lab:** bộ test **chuẩn vàng** mà mọi paper ASR tiếng Anh đều báo cáo WER. Dùng khi cần
con số WER **so sánh được** với bảng công bố của model (parakeet-tdt-0.6b-v2, ...). Subset `dev-clean` /
`test-clean` chỉ ~5.4h nên vẫn vừa cho lab CPU.

---

## Mô tả

- **LibriSpeech** là corpus ~1000 giờ tiếng Anh đọc (read speech), 16kHz, trích từ sách nói LibriVox.
- Tác giả: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur.
- Trang chính thức: **OpenSLR #12** — `https://www.openslr.org/12`.
- Chia 6 subset: `dev-clean`, `dev-other`, `test-clean`, `test-other`, `train-clean-100/360`, `train-other-500`.
- "clean" = giọng dễ nghe (theo điểm WER của model nền khi tạo bộ); "other" = khó hơn.

---

## Số liệu (các subset nhỏ nhất — dùng cho smoke-test / đo WER)

| Subset | Giờ (approx) | Dung lượng tải | Ghi chú |
| --- | --- | --- | --- |
| `dev-clean` | ~5.4h | ~337MB | tập phát triển, giọng dễ |
| `test-clean` | ~5.4h | ~346MB | **tập test chuẩn**, giọng dễ |
| `dev-other` | ~5.3h | ~314MB | giọng khó |
| `test-other` | ~5.1h | ~328MB | giọng khó |
| `train-clean-100` | ~100h | ~6.3GB | (lớn — chỉ khi fine-tune) |

| Thuộc tính chung | Giá trị |
| --- | --- |
| Sample rate | 16kHz mono |
| Định dạng | `.flac` |
| Transcript | Có (file `.trans.txt`, in hoa, không dấu câu) |
| License | **CC BY 4.0** (thương mại OK, chỉ cần ghi công) |

> Để đo WER **so sánh được với bảng công bố**, dùng đúng **`test-clean`** (và/hoặc `test-other`).

---

## Lệnh tải

```bash
mkdir -p data/raw/librispeech
cd data/raw/librispeech

# Bộ test chuẩn (nhỏ, ~346MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz   # giải nén ra LibriSpeech/test-clean/...

# (tuỳ chọn) dev-clean để theo dõi
# wget https://www.openslr.org/resources/12/dev-clean.tar.gz
# tar -xzf dev-clean.tar.gz
```

> Có mirror EU/CN trên trang OpenSLR nếu mirror Mỹ chậm.

Tải qua HuggingFace `datasets` (tiện hơn, tự cache):

```python
# Cần: uv add datasets soundfile
from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")  # test-clean
print(ds[0].keys())   # 'audio', 'text', 'speaker_id', 'chapter_id', 'id'
print(len(ds), "mẫu")
```

---

## Cấu trúc thư mục (giải nén từ tar)

```
LibriSpeech/test-clean/
  <speaker>/<chapter>/
    <speaker>-<chapter>-<utt>.flac
    <speaker>-<chapter>.trans.txt   # "<speaker>-<chapter>-<utt> TEXT VIET HOA"
```

---

## Convert sang manifest NeMo

Cùng logic với Mini LibriSpeech (trang 03) — chỉ đổi đường dẫn gốc:

```python
# tools/build_librispeech_manifest.py
import json
from pathlib import Path
import librosa

root = Path("data/raw/librispeech/LibriSpeech/test-clean")
out = Path("data/manifests/librispeech")
out.mkdir(parents=True, exist_ok=True)

with open(out / "test_clean.jsonl", "w", encoding="utf-8") as fout:
    for trans in root.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, text = line.split(" ", 1)
            flac = trans.parent / f"{utt_id}.flac"
            duration = librosa.get_duration(path=str(flac))
            fout.write(json.dumps({
                "audio_filepath": str(flac.resolve()),
                "duration": round(duration, 3),
                "text": text.strip().lower(),
            }, ensure_ascii=False) + "\n")
print("Đã ghi manifest test-clean")
```

Nếu tải qua HuggingFace `datasets`, ghi audio ra wav rồi build manifest:

```python
import json, soundfile as sf
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
wav_dir = Path("data/raw/librispeech/wav_test_clean"); wav_dir.mkdir(parents=True, exist_ok=True)
out = Path("data/manifests/librispeech"); out.mkdir(parents=True, exist_ok=True)

with open(out / "test_clean_hf.jsonl", "w", encoding="utf-8") as f:
    for ex in ds:
        a = ex["audio"]                       # {'array', 'sampling_rate', 'path'}
        wav_path = wav_dir / f"{ex['id']}.wav"
        sf.write(wav_path, a["array"], a["sampling_rate"])   # 16kHz sẵn
        dur = len(a["array"]) / a["sampling_rate"]
        f.write(json.dumps({
            "audio_filepath": str(wav_path.resolve()),
            "duration": round(dur, 3),
            "text": ex["text"].strip().lower(),
        }, ensure_ascii=False) + "\n")
```

---

## Đo WER bằng NeMo (gợi ý)

```bash
# Dùng script đánh giá của NeMo trên manifest đã build
uv run python <NeMo>/examples/asr/speech_to_text_eval.py \
  pretrained_name=nvidia/parakeet-tdt-0.6b-v2 \
  dataset_manifest=data/manifests/librispeech/test_clean.jsonl \
  batch_size=1
```

(Trên CPU nên để `batch_size=1` và chấp nhận chạy chậm; smoke-test có thể cắt manifest còn vài chục dòng.)

---

## Lưu ý

- `test-clean` ~5.4h chạy hết trên CPU sẽ **lâu** — để smoke-test, cắt manifest còn ~20-50 dòng đầu là đủ
  xác nhận luồng đo WER chạy thông.
- Transcript in hoa, không dấu câu → hạ thường cả 2 phía khi tính WER cho công bằng.
- `test-clean`/`dev-clean` là "clean"; nếu muốn thấy model yếu chỗ nào, thêm `test-other`.

---

## ✅ Tự kiểm nhanh

1. Muốn so WER của model với bảng số trong model card thì đo trên subset nào?
2. Subset `test-clean` khoảng bao nhiêu giờ, và vì sao vẫn hợp lab CPU dù LibriSpeech gốc ~1000h?

<details>
<summary>Đáp án</summary>

1. **`test-clean`** (và/hoặc `test-other`) — đây là tập test chuẩn mọi paper ASR tiếng Anh báo cáo.
2. `test-clean` chỉ ~**5.4h** (~346MB) — là một subset nhỏ tách riêng khỏi 1000h gốc; với smoke-test còn có
   thể cắt manifest còn vài chục dòng để chạy nhanh trên CPU.

</details>
