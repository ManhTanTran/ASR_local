# 01 — Sample wav lẻ của NeMo (smoke-test nhanh nhất)

**Vai trò trong lab:** đây là cách **nhanh nhất** để biết luồng tải model + phiên âm chạy được trên CPU.
Chỉ cần 1 file `.wav` vài giây, 1 lệnh tải, chạy thẳng `src/asr_lab/eval/smoke.py`. **Không cần manifest, không cần WER.**

---

## Mô tả

NeMo và các tutorial của NVIDIA thường dùng một file mẫu cố định lấy từ LibriSpeech:

- **Tên file:** `2086-149220-0033.wav`
- **Nội dung:** một câu đọc ngắn (giọng tiếng Anh, đọc rõ).
- **Nguồn tải lẻ (S3 công khai của NVIDIA demo):**
  `https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav`
- File này xuất hiện trong hầu hết model card NeMo trên HuggingFace như ví dụ `transcribe([...])`.

> File đặt tên theo quy ước LibriSpeech `speaker-chapter-utterance`: speaker 2086, chapter 149220,
> utterance 0033. Vì lấy từ LibriSpeech nên gốc là **16kHz mono** — hợp luôn với yêu cầu của `smoke_transcribe.py`.

---

## Số liệu

| Thuộc tính | Giá trị |
| --- | --- |
| Số file | 1 (có thể tải thêm vài file LibriSpeech lẻ nếu muốn) |
| Thời lượng | vài giây/file |
| Sample rate | 16kHz mono (cần kiểm chứng bằng `soxi`/`ffprobe` sau khi tải) |
| Định dạng | wav |
| Transcript | Có — in trong tutorial/model card; với file lẻ, gõ tay transcript đúng để tính WER nếu cần |
| License | Mẫu demo công khai của NVIDIA; gốc LibriSpeech là CC BY 4.0 (cần kiểm chứng riêng cho file demo) |

---

## Lệnh tải

```bash
# Tạo thư mục lưu sample
mkdir -p data/raw/nemo_samples
cd data/raw/nemo_samples

# Tải file mẫu chuẩn
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav

# Kiểm tra đúng mono 16kHz chưa (nên cài sox: sudo apt-get install sox)
soxi 2086-149220-0033.wav
```

Nếu file không phải mono 16kHz (hiếm, nhưng để chắc), resample lại:

```bash
sox 2086-149220-0033.wav -r 16000 -c 1 2086-149220-0033_16k.wav
```

---

## Chạy thử (smoke-test)

Theo `src/asr_lab/eval/smoke.py` (yêu cầu wav **mono 16kHz**):

```bash
# Mặc định model nhỏ (conformer_ctc_small) — tải nhanh nhất để test luồng
uv run python -m asr_lab.eval.smoke data/raw/nemo_samples/2086-149220-0033.wav

# Hoặc chỉ định model tiếng Anh của lab:
uv run python -m asr_lab.eval.smoke data/raw/nemo_samples/2086-149220-0033.wav \
  --model nvidia/parakeet-tdt_ctc-110m

uv run python -m asr_lab.eval.smoke data/raw/nemo_samples/2086-149220-0033.wav \
  --model nvidia/parakeet-tdt-0.6b-v2
```

Kỳ vọng: in ra `KẾT QUẢ` là câu phiên âm. Nếu ra text hợp lý → luồng tải model + inference trên CPU đã thông.

---

## Convert sang manifest NeMo (nếu muốn đo WER trên vài file lẻ)

Với file lẻ không có transcript chuẩn dạng file, ta tự gõ transcript đúng rồi tạo manifest 1 dòng:

```python
# tools/make_manifest_samples.py — tạo manifest cho vài file wav lẻ
import json
import librosa
from pathlib import Path

# Map: đường dẫn wav -> transcript đúng (gõ tay, viết thường, bỏ dấu câu cho khớp chuẩn NeMo)
samples = {
    "data/raw/nemo_samples/2086-149220-0033.wav": "<điền transcript đúng của file này>",
}

out = Path("data/manifests/nemo_samples")
out.mkdir(parents=True, exist_ok=True)
with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
    for path, text in samples.items():
        duration = librosa.get_duration(path=path)  # đọc thời lượng thật từ file
        f.write(json.dumps({
            "audio_filepath": str(Path(path).resolve()),
            "duration": round(duration, 3),
            "text": text,
        }, ensure_ascii=False) + "\n")
print("Đã ghi manifest:", out / "manifest.jsonl")
```

> Lưu ý: muốn tính WER thì transcript phải **đúng từng từ**. File lẻ không có transcript đóng gói sẵn,
> nên với việc đo WER nghiêm túc hãy dùng AN4 hoặc LibriSpeech (có transcript chuẩn theo file) ở các trang sau.

---

## Lưu ý

- File mẫu này chỉ để **chắc luồng chạy**, không để báo cáo WER (1 file không đại diện cho gì cả).
- Nếu mạng chậm và `wget` đứt giữa chừng, thêm `-c` để tải tiếp: `wget -c <url>`.
- Muốn vài file thay vì 1: tải thêm file lẻ từ một subset LibriSpeech nhỏ (xem trang 04) rồi trích vài file.

---

## ✅ Tự kiểm nhanh

1. Vì sao file mẫu này hợp để smoke-test nhưng KHÔNG hợp để báo cáo WER?

<details>
<summary>Đáp án</summary>

Vì chỉ có **1 file vài giây**, đủ để xác nhận luồng tải model + phiên âm chạy thông, nhưng 1 file không đại
diện cho năng lực model → con số WER (nếu có) vô nghĩa thống kê. Muốn đo WER thật phải dùng bộ có nhiều file
và transcript chuẩn (AN4, LibriSpeech).

</details>
