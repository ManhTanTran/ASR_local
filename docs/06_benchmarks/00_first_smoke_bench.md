# Đo thông luồng lần đầu — RTF + WER trên CPU (tiếng Anh)

Lần đo đầu tiên để **thông luồng** và so sánh **bước đầu** tốc độ suy luận + WER giữa 3 model
tiếng Anh đã có sẵn trong cache. Không phải benchmark chuẩn (số utt nhỏ, chạy CPU).

Ngày đo: 2026-06-19. Script: `src/asr_lab/eval/bench.py`.

---

## Glossary

- **RTF (Real-Time Factor):** thời gian xử lý chia thời lượng audio. RTF < 1 nghĩa là nhanh hơn thời gian thực; RTF 0.1 = xử lý 1 giây audio mất 0,1 giây.
- **WER (Word Error Rate):** tỉ lệ lỗi từ sau khi chuẩn hoá (hạ thường + bỏ dấu câu), tính gộp toàn tập.
- **PnC:** dấu câu + viết hoa; Parakeet/Nemotron có PnC nên phải chuẩn hoá trước khi so WER với LibriSpeech (ref không dấu câu).
- **CTC / TDT / RNNT:** ba kiểu decoder (xem `../02_asr_components/`).

---

## 1. Cấu hình đo

- **Phần cứng:** CPU, cap **4 thread** (`OMP_NUM_THREADS=4` + `torch.set_num_threads(4)`) — chống treo máy do quá tải.
- **Tập test:** 30 utterance đầu của mini-LibriSpeech `dev-clean-2` (read speech, giọng đọc sạch), tổng ~3,3 phút audio, trung bình 6,7s/utt.
- **batch_size:** 8, mỗi model nạp riêng rồi giải phóng trước khi sang model sau.
- **Chuẩn hoá WER:** hạ thường, bỏ ký tự không phải chữ/số, gộp khoảng trắng (áp cho cả ref và hyp).

---

## 2. Kết quả

| Model | Params | Decoder | infer (s) | RTF | WER |
| --- | --- | --- | --- | --- | --- |
| `stt_en_conformer_ctc_small` | 13M | CTC (Conformer cũ) | 19,7 | **0,116** | 5,42% |
| `stt_en_fastconformer_transducer_large` | ~115M | RNNT (FastConformer, giống VPB) | 36,8 | 0,217 | **3,54%** |
| `parakeet-tdt-0.6b-v2` | 618M | TDT (FastConformer) | 111,3 | 0,655 | 3,77% |
| `nemotron-speech-streaming-en-0.6b` | 618M | RNNT streaming (FastConformer) | 115,5 | 0,680 | 5,66% |

> Lần đo lại 4 model (2026-06-19) sau khi thêm FastConformer-large 115M làm baseline khớp kiến trúc VPB.

---

## 3. Đọc kết quả (khách quan)

- **FastConformer-large 115M là điểm ngọt (trên tập này):** WER thấp nhất (3,54%) — còn nhỉnh hơn cả hai model 0,6B — mà nhanh hơn ~3 lần (RTF 0,217 so với 0,66–0,68). Đây đúng là cỡ model VPB từng dùng.
- **Tốc độ theo cỡ:** 13M (RTF 0,116) < 115M (0,217) < 0,6B (0,66–0,68). Tỉ lệ thuận với số tham số. Cả bốn đều RTF < 1 → trên CPU vẫn nhanh hơn thời gian thực với batch 8.
- **WER:** FastConformer 115M (3,54%) ≈ Parakeet-TDT (3,77%) < Conformer nhỏ (5,42%) ≈ Nemotron (5,66%).
- **Bigger KHÔNG tự khắc tốt hơn ở đây:** trên LibriSpeech `dev-clean` (giọng đọc sạch, dễ), tăng từ 115M lên 618M không cải thiện WER — thậm chí 115M còn thắng sít sao. Phần dung lượng dư của model 0,6B chỉ phát huy trên dữ liệu KHÓ (ồn, hội thoại, điện thoại). Đừng kết luận "115M > 0,6B" chung chung từ một tập dễ.
- **Khoảng cách hẹp:** đúng như dự đoán — tập quá dễ nên chênh lệch bị nén. Muốn giãn phải đo trên tập khó hơn.

## 4. Lưu ý / hạn chế (không thổi phồng)

- **Số utt nhỏ (30):** WER có sai số thống kê lớn, chỉ mang tính thông luồng. Muốn số đáng tin cần chạy cả tập (1089 utt) hoặc tập chuẩn.
- **Nemotron chạy chế độ offline:** ở đây gọi `transcribe()` nguyên câu, KHÔNG dùng streaming — nên RTF này KHÔNG phản ánh thế mạnh streaming (nhiều luồng đồng thời) của nó. So tốc độ offline là hơi thiệt cho Nemotron.
- **RTF phụ thuộc batch + thread:** đổi batch_size hay số thread sẽ ra số khác; đây là số tương đối trong cùng một lần đo.
- **Tiếng Anh, không phải tiếng Việt:** ba model này English-only; đây chỉ là đo thông luồng. Đo tiếng Việt phải dùng `nemotron-3.5-asr-streaming-0.6b` + tập VIVOS (xem `../04_datasets_vi/`).

## 5. Cách chạy lại

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
uv run python -m asr_lab.eval.bench data/manifests/librispeech_dev_clean_2.jsonl --limit 30 --batch 8
```

---

## ✅ Tự kiểm nhanh

1. Vì sao WER giữa 3 model chênh ít trong lần đo này?

<details><summary>Đáp án</summary>

Vì LibriSpeech dev-clean là giọng đọc sạch, dễ; ngay model nhỏ 13M đã đạt ~5% WER nên khoảng cách bị nén. Tập khó hơn (ồn/hội thoại/điện thoại) mới giãn ra.
</details>

2. Số RTF của Nemotron ở đây có phản ánh đúng thế mạnh của nó không?

<details><summary>Đáp án</summary>

Không. Đây là chạy offline (transcribe nguyên câu), không dùng streaming. Thế mạnh của Nemotron là cache-aware streaming (nhiều luồng đồng thời, độ trễ thấp) — không thể hiện qua phép đo offline này.
</details>
