# Nemotron Speech Streaming — đặc tính, đánh giá, điểm tiến bộ

Nội dung khách quan ở Mục 1–3; so sánh tiến bộ và cách tự đánh giá ở Mục 4–5.

---

## Glossary

- **Full-duplex** — nói và nghe đồng thời, ngắt lời được (trong VoiceChat).
- **Concurrent streams** — số luồng audio xử lý song song trên một GPU.
- **Pareto latency-accuracy** — đường đánh đổi độ trễ và độ chính xác.

---

## 1. Đặc tính

- **Streaming độ trễ thấp** — giải mã tăng dần theo chunk; độ trễ chọn được từ ~80ms đến ~1s.
- **Điều chỉnh độ trễ lúc suy luận** — đổi `att_context_size`, không cần train lại (xem `01_structure.md` Mục 4).
- **PnC** — hỗ trợ dấu câu + viết hoa.
- **Đa ngôn ngữ** — bản en là tiếng Anh; **Nemotron-3.5-ASR-Streaming-0.6B** mở rộng **40 ngôn ngữ-locale**.

## 2. Các bản liên quan

- **nemotron-speech-streaming-en-0.6b** — tiếng Anh (bản phân tích ở đây).
- **Nemotron-3.5-ASR-Streaming-0.6B** — 40 ngôn ngữ, độ trễ 80ms–1s.
- **Nemotron 3 VoiceChat** — hội thoại full-duplex, ngắt lời được; xây trên **LLM backbone Nemotron Nano v2** + speech decoder + TTS decoder (hợp nhất ASR + LLM + TTS).

## 3. Đánh giá (số công bố)

- **Hiệu quả streaming** — nhờ cache-aware (mỗi khung tính một lần), đạt ~3 lần hiệu quả so với buffered; báo cáo **~17 lần số luồng đồng thời** trên 1 GPU H100.
- **Chất lượng theo độ trễ** — WER giảm khi tăng `right` (độ trễ cao hơn); người dùng chọn điểm trên đường Pareto.

> Lưu ý: các con số này là benchmark GPU của NVIDIA, không phải đo trên CPU của lab.

---

## 4. Điểm tiến bộ so với Fast-Conformer (model VPB)

| Khía cạnh | Fast-Conformer RNNT (VPB) | Nemotron-Streaming | Ý nghĩa |
| --- | --- | --- | --- |
| **Streaming** | cache-aware có sẵn trong NeMo nhưng buffered phổ biến | cache-aware tối ưu, không overlap | Mỗi khung tính một lần → nhiều luồng hơn |
| **Điều chỉnh độ trễ** | cố định theo cấu hình train | đổi `att_context_size` lúc suy luận | Một model phục vụ nhiều mức độ trễ |
| **Quy mô** | ~120M | ~618M | Chính xác hơn |
| **Ngôn ngữ** | đơn ngữ (train tiếng Việt) | en, hoặc 40 ngôn ngữ (3.5) | Phủ rộng |
| **Decoder** | RNNT | RNNT (giữ nguyên) | Phần này KHÔNG đổi — tiến bộ nằm ở encoder cache-aware |

- **Điểm cốt lõi** — decoder vẫn là RNNT quen thuộc; tiến bộ chính nằm ở **encoder cache-aware** (hiệu quả streaming) và **điều chỉnh độ trễ runtime**. Đây đúng là hướng callbot VPB cần (real-time, độ trễ thấp).

---

## 5. Cách tự đánh giá tại máy (đề xuất)

- **Streaming demo** — NeMo có script `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`; chạy với các `att_context_size` khác nhau để cảm nhận đánh đổi độ trễ.
- **Phiên âm offline thử** — vẫn dùng `src/asr_lab/eval/smoke.py <wav> --model nvidia/nemotron-speech-streaming-en-0.6b` để kiểm tra chất lượng (tiếng Anh).
- **Cảnh báo CPU** — giới hạn `OMP_NUM_THREADS` khi chạy; đo độ trễ streaming thật nên dùng GPU. Bản en chỉ tiếng Anh — không hợp test tiếng Việt.

---

## ✅ Tự kiểm nhanh

1. Tiến bộ chính của Nemotron-Streaming so với Fast-Conformer nằm ở phần nào, decoder có đổi không?

<details><summary>Đáp án</summary>

Nằm ở encoder cache-aware (hiệu quả streaming, không overlap) và điều chỉnh độ trễ runtime qua att_context_size. Decoder vẫn là RNNT, không đổi.
</details>

2. Vì sao Nemotron hợp bài toán callbot real-time?

<details><summary>Đáp án</summary>

Vì giải mã streaming tăng dần độ trễ thấp (80ms–1s), cache-aware cho nhiều luồng đồng thời, và chọn được mức độ trễ phù hợp lúc chạy mà không train lại.
</details>
