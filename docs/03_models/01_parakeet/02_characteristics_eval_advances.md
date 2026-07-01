# Parakeet — đặc tính, đánh giá, điểm tiến bộ

Nội dung khách quan (đặc tính + đánh giá) ở Mục 1–3; phần so sánh tiến bộ và cách tự đánh giá ở Mục 4–5.

---

## Glossary

- **PnC** — Punctuation and Capitalization: tự thêm dấu câu, viết hoa.
- **RTFx** — số lần nhanh hơn thời gian thực (nghịch đảo RTF).
- **Open-ASR Leaderboard** — bảng xếp hạng ASR tiếng Anh công khai của HuggingFace.
- **pseudo-label** — nhãn do model sinh ra để huấn luyện tiếp.

---

## 1. Đặc tính

- **PnC tự động** — xuất văn bản có dấu câu và viết hoa, không cần hậu xử lý.
- **Timestamp** — dự đoán mốc thời gian mức ký tự / từ / đoạn.
- **Audio dài** — xử lý tới ~24 phút trong một lần.
- **Tiếng Anh** (bản v2); họ Parakeet còn bản đa ngôn ngữ (xem Mục 2).

## 2. Các biến thể

- **parakeet-ctc / rnnt / tdt** — cùng encoder, khác decoder; TDT nhanh nhất.
- **parakeet-tdt_ctc** — hybrid hai đầu giải mã (TDT + CTC).
- **Parakeet V3** — đa ngôn ngữ (25 ngôn ngữ châu Âu), nhận dạng + dịch.
- **parakeet-unified-en-0.6b** — một checkpoint cả offline lẫn streaming (độ trễ tối thiểu 160ms).

## 3. Đánh giá (số công bố)

- **Tốc độ** — RTFx ~3380 trên Open-ASR Leaderboard (batch 128); rất nhanh nhờ TDT bỏ khung.
- **Dữ liệu huấn luyện** — ~120.000 giờ tiếng Anh: **10.000 giờ gán nhãn thủ công + 110.000 giờ pseudo-label**.
- **Chất lượng** — thuộc nhóm dẫn đầu Open-ASR Leaderboard tại thời điểm phát hành (WER thấp; con số cụ thể thay đổi theo bản leaderboard, tra trực tiếp khi cần trích dẫn).

> Lưu ý đọc số: RTFx và WER là số benchmark offline của NVIDIA trên phần cứng GPU, không phải đo trên máy CPU của lab này.

---

## 4. Điểm tiến bộ so với Fast-Conformer (model VPB)

| Khía cạnh | Fast-Conformer RNNT (VPB) | Parakeet-TDT | Ý nghĩa |
| --- | --- | --- | --- |
| **Decoder** | RNNT duyệt từng khung | **TDT bỏ 0–4 khung/bước** | Nhanh hơn rõ rệt, WER tương đương |
| **Quy mô** | ~120M (d512/17 lớp) | ~618M (d1024/24 lớp) | Dung lượng biểu diễn lớn hơn → chính xác hơn |
| **PnC** | không có | có sẵn | Bớt một bước hậu xử lý dấu câu |
| **Timestamp** | cơ bản | mức ký tự/từ/đoạn | Hữu ích cho căn phụ đề, phân đoạn |
| **Audio dài** | giới hạn theo bộ nhớ | tới 24 phút/lần | Ít phải cắt khúc thủ công |
| **Recipe** | pretrain → ft pseudo → ft VPB | 10k thật + 110k pseudo | Cùng mô-típ pseudo-label, quy mô lớn hơn |

- **Điểm cốt lõi** — encoder vẫn cùng họ FastConformer; tiến bộ chính nằm ở **decoder TDT** (tốc độ) và **quy mô + dữ liệu** (chất lượng). Phần anh đã hiểu về Conformer vẫn áp dụng.

---

## 5. Cách tự đánh giá tại máy (đề xuất)

Vì lab chạy CPU, nên đánh giá nhẹ (đối chiếu định tính), không đo tốc độ GPU:

- **Phiên âm thử** — dùng `src/asr_lab/eval/smoke.py <wav> --model nvidia/parakeet-tdt-0.6b-v2` để xem chất lượng + PnC trên audio của mình.
- **Đo WER** nếu có nhãn — chuẩn bị manifest `.jsonl` (`audio_filepath`, `text`), dùng metric WER của NeMo (xem `../../02_asr_components/09_evaluation_wer.md`); luôn dùng tập độc lập, chuẩn hóa văn bản thống nhất.
- **Cảnh báo CPU** — model 0.6B chạy CPU chậm; giới hạn nhân (`OMP_NUM_THREADS`) để khỏi treo máy. Đo tốc độ thật cần GPU.

---

## ✅ Tự kiểm nhanh

1. Tiến bộ chính của Parakeet so với Fast-Conformer nằm ở phần nào?

<details><summary>Đáp án</summary>

Chủ yếu ở decoder TDT (bỏ khung → nhanh hơn) và quy mô + dữ liệu (lớn hơn → chính xác hơn, kèm PnC/timestamp). Encoder vẫn cùng họ FastConformer.
</details>

2. Vì sao không nên trích RTFx 3380 như tốc độ trên máy CPU của mình?

<details><summary>Đáp án</summary>

Vì RTFx 3380 là benchmark của NVIDIA trên GPU (batch 128); máy CPU của lab sẽ chậm hơn nhiều. Đó là số tham khảo về tiềm năng, không phải đo tại chỗ.
</details>
