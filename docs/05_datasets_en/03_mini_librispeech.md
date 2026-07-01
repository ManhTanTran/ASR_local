# 03 — Mini LibriSpeech (subset hồi quy của LibriSpeech)

**Vai trò trong lab:** giọng đọc **tự nhiên** (không phải đánh vần như AN4) nhưng vẫn nhỏ. Hợp để smoke-test
luồng WER với dữ liệu sát thực tế hơn AN4, license sạch (**CC BY 4.0**).

---

## Mô tả

- **Mini LibriSpeech** là một subset của LibriSpeech, tạo ra để **regression testing** (kiểm thử hồi quy)
  — tức bộ nhỏ để chạy thử nhanh, đúng tinh thần smoke-test.
- Trang chính thức: **OpenSLR #31** — `https://www.openslr.org/31/`.
- Nội dung là **read speech** (đọc sách nói), giọng tiếng Anh rõ, 16kHz.

---

## Số liệu

| Thuộc tính | Giá trị |
| --- | --- |
| `dev-clean-2.tar.gz` | ~126MB |
| `train-clean-5.tar.gz` | ~332MB |
| `md5sum.txt` | 108 bytes (checksum) |
| Sample rate | 16kHz (cần kiểm chứng — OpenSLR #31 không in rõ, nhưng LibriSpeech gốc là 16kHz) |
| Định dạng | `.flac` (như LibriSpeech gốc) |
| Transcript | Có (file `.trans.txt` mỗi chapter) |
| License | **CC BY 4.0** (thương mại OK, chỉ cần ghi công) |

> Số "giờ" không in rõ trên trang; ước lượng `dev-clean-2` cỡ vài giờ. **Cần kiểm chứng** bằng cách cộng
> `duration` trong manifest sau khi convert.

---

## Lệnh tải

```bash
mkdir -p data/raw/mini_librispeech
cd data/raw/mini_librispeech

# Bộ nhỏ nhất để smoke-test: dev-clean-2 (~126MB)
wget https://www.openslr.org/resources/31/dev-clean-2.tar.gz
tar -xzf dev-clean-2.tar.gz   # giải nén ra LibriSpeech/dev-clean-2/...

# (tuỳ chọn) bộ train nhỏ
# wget https://www.openslr.org/resources/31/train-clean-5.tar.gz
# tar -xzf train-clean-5.tar.gz

# Kiểm tra checksum
wget https://www.openslr.org/resources/31/md5sum.txt
md5sum -c md5sum.txt
```

> Nếu OpenSLR (Mỹ) chậm, có mirror EU/CN trên cùng trang `openslr.org/31`.

---

## Cấu trúc thư mục LibriSpeech (áp dụng cho cả Mini và bản đầy đủ)

```
LibriSpeech/dev-clean-2/
  <speaker>/<chapter>/
    <speaker>-<chapter>-<utt>.flac      # audio
    <speaker>-<chapter>.trans.txt       # transcript: mỗi dòng "<speaker>-<chapter>-<utt> TEXT VIET HOA"
```

Transcript trong LibriSpeech viết **chữ in hoa, không dấu câu** (đã chuẩn hoá sẵn cho ASR).

---

## Convert sang manifest NeMo

```python
# tools/build_librispeech_manifest.py — quét .flac + .trans.txt -> manifest NeMo
import json
from pathlib import Path
import librosa

root = Path("data/raw/mini_librispeech/LibriSpeech/dev-clean-2")
out = Path("data/manifests/mini_librispeech")
out.mkdir(parents=True, exist_ok=True)

with open(out / "dev_clean_2.jsonl", "w", encoding="utf-8") as fout:
    # Mỗi file .trans.txt chứa transcript của 1 chapter
    for trans in root.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, text = line.split(" ", 1)          # tách id và phần text
            flac = trans.parent / f"{utt_id}.flac"     # audio cùng thư mục
            duration = librosa.get_duration(path=str(flac))
            fout.write(json.dumps({
                "audio_filepath": str(flac.resolve()),
                "duration": round(duration, 3),
                "text": text.strip().lower(),          # hạ thường cho khớp output model
            }, ensure_ascii=False) + "\n")
print("Đã ghi manifest dev-clean-2")
```

> NeMo đọc `.flac` trực tiếp, không bắt buộc convert sang `.wav`. Nếu pipeline nào chỉ nhận wav thì convert
> bằng `sox in.flac -r 16000 -c 1 out.wav`.

---

## Chạy thử

```bash
# Phiên âm 1 file flac lẻ
uv run python -m asr_lab.eval.smoke \
  data/raw/mini_librispeech/LibriSpeech/dev-clean-2/<spk>/<chap>/<utt>.flac \
  --model nvidia/parakeet-tdt-0.6b-v2
```

(Nếu `smoke_transcribe.py` chỉ nhận wav, convert file flac sang wav 16kHz trước.)

---

## Lưu ý

- Là subset hồi quy nên **không phải bộ test chuẩn** để so số với paper — muốn so số dùng `test-clean` bản
  đầy đủ (trang 04).
- Transcript đã chuẩn hoá in hoa, không dấu câu → khi tính WER nên hạ thường cả hypothesis lẫn reference
  cho công bằng.

---

## ✅ Tự kiểm nhanh

1. Mini LibriSpeech khác AN4 ở điểm nào khiến nó "sát thực tế" hơn cho việc đo WER?
2. License của Mini LibriSpeech là gì, có cho dùng thương mại không?

<details>
<summary>Đáp án</summary>

1. Nội dung là **đọc sách nói tự nhiên** (read speech), không phải **đánh vần chữ/số** như AN4 → WER phản
   ánh năng lực model trên lời nói thường tốt hơn.
2. **CC BY 4.0** — cho phép thương mại, chỉ cần ghi công nguồn.

</details>
