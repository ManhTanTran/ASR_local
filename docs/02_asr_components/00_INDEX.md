# 00 — INDEX: thành phần ASR dùng chung

Kiến thức nền về các thành phần của một pipeline ASR, **không gắn với model cụ thể** — dùng chung giữa Fast-Conformer (model VPB cũ), Parakeet, Nemotron.
Clone từ cụm deep-dive bên `stt_nvidia_nemo`; dùng làm tham chiếu khi phân tích model mới ở `../03_models/`.

---

## Thứ tự đọc (audio → text)

| File | Thành phần |
| --- | --- |
| [01_pipeline_overview.md](01_pipeline_overview.md) | Toàn cảnh pipeline audio → text |
| [02_tokenizer.md](02_tokenizer.md) | Tokenizer SentencePiece BPE |
| [03_audio_to_mel.md](03_audio_to_mel.md) | Tiền xử lý waveform → log-mel |
| [04_specaugment.md](04_specaugment.md) | Tăng cường dữ liệu khi huấn luyện |
| [05_encoder_conformer.md](05_encoder_conformer.md) | **Encoder Conformer / Fast-Conformer** (trung tâm) |
| [06_decode_ctc.md](06_decode_ctc.md) | Giải mã CTC |
| [07_decode_rnnt.md](07_decode_rnnt.md) | Giải mã RNNT + Joint |
| [08_decode_aed.md](08_decode_aed.md) | Giải mã AED encoder–decoder |
| [09_evaluation_wer.md](09_evaluation_wer.md) | Đánh giá chất lượng (WER) |

---

## Liên hệ với model mới

- **TDT** (Parakeet) là mở rộng của **RNNT** (`07_decode_rnnt.md`) — thêm dự đoán duration.
- **Cache-aware streaming** (Nemotron) dựng trên encoder Conformer (`05_encoder_conformer.md`) + giải mã RNNT (`07`).
- Chi tiết model cụ thể: `../03_models/01_parakeet_nemotron.md`.

> Ghi chú: vài file còn tham chiếu tới tài liệu review VPB (`01_asr_domain_review.md`) nằm ở repo `stt_nvidia_nemo` (không clone sang đây).
