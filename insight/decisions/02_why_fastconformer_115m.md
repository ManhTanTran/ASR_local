# decision 02 — chọn fastconformer-transducer-large 115M làm model nền

**Bối cảnh:** cần model nền NeMo để fine-tune tiếng Việt, vừa cỡ deploy thực tế.

**Lựa chọn:** parakeet-TDT 0.6B · nemotron streaming 0.6B · **fastconformer-transducer-large 115M**.

**Lý do chốt B (115M):**
- **Đúng cỡ kiến trúc VPB** (~120M large) — sát điều kiện deploy callbot, không phình.
- **Offline RNNT** → fine-tune đổi-vocab hội tụ (xem decision 01).
- **Cùng hạng cân SOTA cỡ nhỏ**: ChunkFormer-CTC 110M đạt 4,18% trên VIVOS ⇒ cỡ này ĐỦ chạm SOTA
  nếu có đủ data (xem `proposals/01`). Không cần model tỉ-tham-số.

**Bằng chứng hội tụ:** `vivos-fc115m-v1` WER 100%→20,37% (`experiments/00_vivos_fc115m/RESULT.md`).

**Đánh đổi đã biết:** offline (không streaming) — cho callbot cần độ trễ thấp sau này có thể phải xét
lại sang biến thể cache-aware, nhưng tách thành việc riêng khi tới đó.
