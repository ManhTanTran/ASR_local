# 07 — VietSuperSpeech (gần domain callbot nhất)

Bộ **hội thoại tự nhiên** (~103h) cắt từ podcast/phỏng vấn, công bố kèm bài arXiv hướng tới
**ASR cho chatbot, customer support, call center** — đúng domain callbot của Mr. Kỳ.

> arXiv: 2603.01894 · HuggingFace: `thanhnew2001/VietSuperSpeech`

---

## Số liệu

- **Loại:** **hội thoại/tự nhiên** (conversational) — cắt từ podcast & phỏng vấn (HaveASip, Vietcetera, ...).
  Đây là điểm khác biệt: gần văn nói đời thực hơn hẳn các bộ đọc.
- **Giờ audio:** ~**103.18h** (29.041 train + 3.226 dev).
- **Sample rate / định dạng:** **16kHz**, **WAV**. **KHÔNG phải 8kHz điện thoại** — vẫn là audio chất lượng cao.
- **Độ dài đoạn:** trung bình ~12 giây/đoạn.
- **Transcript:** có, **nhưng là auto-label** — sinh bằng model `Zipformer-30M-RNNT-6000h`, **không phải người gõ tay**.
- **License:** **MIT** — **cho phép thương mại**.

## License + cảnh báo chất lượng nhãn

- MIT → dùng thoải mái kể cả thương mại. Điểm cộng lớn.
- **Nhưng transcript là máy sinh** → có sai số nền. Khi đo WER trên bộ này, con số phản ánh
  "model của mình so với model Zipformer kia" chứ không phải so với người. Dùng để **cảm nhận domain hội thoại**,
  không nên coi là ground-truth tuyệt đối. Muốn đo WER nghiêm túc trên hội thoại → cần tự kiểm/sửa transcript một phần.

## Cách tải

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('thanhnew2001/VietSuperSpeech', split='train', streaming=True)
print(next(iter(ds)))
"
```

## Convert sang manifest NeMo

Pattern giống VIVOS (ghi wav + sinh `.jsonl`). Kiểm `ds.features` để biết tên cột.

## Giả lập điện thoại 8kHz (quan trọng cho callbot)

Vì chưa có bộ điện thoại 8kHz công khai, có thể **giả lập băng thông hẹp** từ bộ 16kHz này để test gần domain hơn:

```bash
# Downsample 16kHz -> 8kHz rồi upsample lại 16kHz: mô phỏng mất băng thông kiểu điện thoại
ffmpeg -i in_16k.wav -ar 8000 tmp_8k.wav
ffmpeg -i tmp_8k.wav -ar 16000 out_phone_sim.wav
```

> Đây chỉ là **xấp xỉ** (mất băng thông cao). Điện thoại thật còn có codec nén (G.711/G.729), jitter, nhiễu nền —
> giả lập không thay được audio điện thoại thật, nhưng giúp ước lượng model sụt WER bao nhiêu khi băng thông hẹp.

## Lưu ý

- Đây là **bộ gần callbot nhất** tìm được trong khảo sát → ưu tiên cho bài đo "model chịu được hội thoại không".
- Vì auto-label nên dùng kèm 1 bộ ground-truth người gõ (VIVOS/FOSD) để có điểm tham chiếu đáng tin.

## ✅ Tự kiểm nhanh

1. Vì sao VietSuperSpeech gần callbot hơn các bộ đọc, nhưng vẫn phải dè chừng khi đo WER?
2. Làm sao mô phỏng audio điện thoại từ bộ 16kHz này?

<details>
<summary>Đáp án</summary>

1. Vì là **hội thoại tự nhiên** (podcast/phỏng vấn) → gần văn nói đời thực. Dè chừng vì **transcript là máy sinh
   (auto-label)**, có sai số nền, không phải ground-truth người gõ.
2. **Downsample 16kHz → 8kHz rồi upsample lại 16kHz** bằng `ffmpeg` để giả lập mất băng thông điện thoại.

</details>
