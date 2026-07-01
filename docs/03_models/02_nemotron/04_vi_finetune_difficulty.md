# Vì sao nemotron-streaming-0.6b KHÓ fine-tune đơn giản sang tiếng Việt

Report giải thích đặc thù kiến trúc khiến `nemotron-speech-streaming-en-0.6b` **không hội tụ** khi
fine-tune kiểu đổi-vocab sang tiếng Việt (VIVOS) — trong khi `stt_en_fastconformer_transducer_large`
(115M, offline) cùng recipe lại học tốt (WER 100% → 20%). Số liệu kiến trúc trích trực tiếp từ
`model_config.yaml` trong file `.nemo`.

---

## Glossary

- **Cache-aware streaming:** model nghe theo TỪNG KHÚC (chunk) ngắn, chỉ thấy ít ngữ cảnh trái/phải
  quanh khúc đó — để chạy thời gian thực (streaming). Khác **offline** = thấy TRỌN câu mới giải mã.
- **att_context_style:** kiểu ngữ cảnh attention. `regular` = nhìn toàn câu (offline). `chunked_limited`
  = nhìn theo khúc giới hạn (streaming).
- **conv_context = causal:** tích chập chỉ nhìn quá khứ (không nhìn tương lai) — đặc trưng streaming.
- **Hybrid RNNT-CTC:** model có 2 đầu giải mã (RNNT chính + CTC phụ), train bằng tổng 2 loss.
- **change_vocabulary:** thay tokenizer + dựng lại decoder/joint cho ngôn ngữ mới (vocab cũ là tiếng Anh).
- **collapse-to-blank:** model RNNT học mẹo luôn phát ký tự rỗng → loss kẹt, WER 100%.

---

## 1. Làm rõ: model đã test KHÔNG phải bản đa ngữ

Có hai nhánh "nemotron ASR" dễ lẫn:

| Model | Ngôn ngữ | Kiểu | Ghi chú |
| --- | --- | --- | --- |
| `nemotron-speech-streaming-en-0.6b` (đã test) | **Chỉ tiếng Anh** | RNNT **streaming** | Bản mình dùng — đặc thù là *streaming*, không phải đa ngữ |
| `nemotron-3.5-asr-streaming-0.6b` (chưa test) | **Đa ngữ** (có vi-VN) | Multitask/Canary-style | Bản này mới là "multi-language" |

Vậy cái khó mình GẶP THẬT là do **streaming + hybrid**, không phải đa ngữ. Phần đa ngữ (bản 3.5)
bàn riêng ở §4 (chưa kiểm chứng trực tiếp).

## 2. Đặc thù kiến trúc nemotron-streaming (trích config thật)

| Thuộc tính | nemotron-streaming-en | fastconformer-large (đối chứng) |
| --- | --- | --- |
| Encoder | Conformer 24 lớp, d_model 1024 | Conformer 17 lớp, d_model 512 |
| **att_context_style** | **`chunked_limited`** (streaming) | **`regular`** (offline, nhìn trọn câu) |
| **conv_context** | **`causal`** (chỉ quá khứ) | `null` (nhìn 2 chiều) |
| Đầu giải mã | **Hybrid RNNT + CTC** (`ctc_loss_weight: 0.3`) | RNNT thuần |
| fuse_loss_wer | true | (mặc định) |

Ba điểm in đậm chính là nguồn gốc khó:

1. **Ngữ cảnh bị giới hạn theo khúc (`chunked_limited` + conv `causal`).**
   Encoder streaming mỗi frame chỉ thấy một cửa sổ ngắn, không thấy trọn câu. Đặc trưng âm vì thế
   "mỏng" hơn. Khi decoder + joint bị reset (đổi vocab) và phải học lại từ đầu cách ánh xạ âm→chữ
   tiếng Việt, ngữ cảnh mỏng khiến tín hiệu học yếu → model dễ chọn lối tắt **collapse-to-blank**.

2. **Hybrid RNNT-CTC + fuse_loss_wer.**
   `change_vocabulary` phải dựng lại ĐÚNG cả joint RNNT lẫn đầu CTC phụ và nối lại fused-loss. Recipe
   đổi-vocab tiêu chuẩn (mình dùng) tối ưu cho RNNT thuần; với hybrid streaming, phần CTC/ngữ-cảnh
   không được cấu hình khớp → gradient không đủ "sạch" để thoát blank.

3. **att_context_size là một DANH SÁCH** (train đa-ngữ-cảnh cho cache-aware). Fine-tune ngây thơ không
   set lại lịch ngữ cảnh này → chế độ train lệch với lúc model được huấn luyện gốc.

## 3. Bằng chứng thực nghiệm (khớp lý thuyết)

- nemotron-streaming: loss kẹt ~85-135 (mức ngẫu nhiên ≈ log(vocab)·độ-dài) suốt 1.600+ step, cả khi
  **full** lẫn **freeze** encoder, cả fp16; WER giữ 100% (toàn blank).
- fastconformer-large (offline, `regular`): cùng recipe → loss giảm đều 470 → ~0,7; WER 100% → 20,37%.
- → Biến số khác biệt quyết định là **ngữ cảnh encoder (streaming vs offline)**, không phải precision
  hay freeze. (fp16 từng bị nghi nhưng fastconformer fp16 vẫn học tốt → loại trừ.)

## 4. Bản đa ngữ nemotron-3.5 (chưa test — vì sao cũng không "đổi-vocab" được)

Bản `nemotron-3.5-asr-streaming-0.6b` đa ngữ theo hướng **multitask/Canary** (một model làm nhiều
ngôn ngữ + dịch, điều khiển bằng *token ngôn ngữ* trong chuỗi đích). Với loại này:

- Vocab đã **bao gồm tiếng Việt** sẵn → KHÔNG nên `change_vocabulary` (sẽ vứt mất năng lực đa ngữ).
- Cách đúng = fine-tune **GIỮ vocab gốc**, nạp dữ liệu Việt + token langid `vi`, học tiếp (domain-adapt).

> Lưu ý: phần §4 dựa trên hiểu biết chung về dòng multitask/Canary, **chưa kiểm chứng trực tiếp** trên
> bản 3.5 (chưa tải/chạy). Cần một vòng thử riêng để xác nhận.

## 5. Kết luận + khuyến nghị

- Muốn fine-tune nhanh-gọn sang tiếng Việt theo kiểu đổi-vocab → chọn model **offline, RNNT thuần**
  (như fastconformer-large). Đây là lựa chọn đã chứng minh hoạt động.
- Nếu BẮT BUỘC dùng nemotron-streaming (vì cần streaming thời gian thực): không đổi-vocab ngây thơ;
  cần (a) cấu hình lại lịch att_context cho fine-tune, (b) xử lý đúng nhánh CTC + fused-loss, (c) có
  thể train nhiều bước hơn / khởi tạo decoder ấm. Đây là việc chuyên sâu, để vòng sau.
- Nếu cần 0.6B đa ngữ: thử bản 3.5 theo hướng **giữ vocab + langid**, không đổi-vocab.

---

## ✅ Tự kiểm nhanh

1. Biến số kiến trúc nào quyết định việc fine-tune đổi-vocab thành/bại, và bằng chứng?

<details><summary>Đáp án</summary>

**Ngữ cảnh encoder**: streaming `chunked_limited` + conv `causal` (nemotron) vs offline `regular`
(fastconformer). Bằng chứng: cùng recipe/precision, fastconformer (offline) hội tụ WER 100%→20%,
nemotron (streaming) kẹt 100% (collapse-to-blank).
</details>

2. Với bản nemotron đa ngữ (3.5) nên fine-tune thế nào, vì sao KHÔNG đổi-vocab?

<details><summary>Đáp án</summary>

Giữ vocab gốc (đã có tiếng Việt) + dùng token ngôn ngữ `vi`, học tiếp trên dữ liệu Việt. Đổi-vocab
sẽ vứt mất năng lực đa ngữ đã học và phá cấu trúc multitask.
</details>
