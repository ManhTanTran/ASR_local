# 07.05 — Phân tích model + ước lượng batch_size, epoch

Chốt kiến trúc dùng, ước lượng bộ nhớ RNNT để chọn batch, và số epoch mỗi nấc curriculum.
Con số bộ nhớ/throughput là **ước lượng lý thuyết** — phải xác nhận bằng smoke 50-step trên GB10.

---

## Glossary

- **Subsampling 8×:** encoder FastConformer nén 8 lần chiều thời gian → 1 frame ≈ 80ms (từ 10ms).
- **Joint network (RNNT):** tensor `B×T×U×V` (batch × frame audio × token nhãn × vocab) — **ngốn bộ nhớ nhất** khi train.
- **Effective batch:** batch thật × `accumulate_grad_batches` (cộng dồn gradient nhiều bước rồi mới update).

---

## Model dùng — ĐÃ xác nhận (inspect ckpt `vivos-fc115m-v2norm`, 2026-07-01)

Số thật từ ckpt `.nemo` của lab (không còn là giả định):

- **Class:** `EncDecRNNTBPEModel` (RNNT thuần, offline). Tổng **114.095.617 param** (đúng ~115M, KHÔNG phải 0.6B).
- **Encoder:** `ConformerEncoder` (FastConformer) 108.8M — d_model **512**, **17 lớp**, subsampling **dw_striding 8×**, 8 heads, feat_in 80.
- **Decoder** RNNT 3.9M · **Joint** 1.4M · **tokenizer vocab = 1024** (ckpt hiện đã 1024, không phải 512 — validate luôn QĐ-3 ở [03](03_tokenizer_vocab.md)).
- sample_rate 16000.

Ghi chú:
- Nhánh chạy thật của lab là ckpt trên (override `--pretrained`, KHÔNG dùng default `nemotron-0.6b` trong code).
- **Hướng nâng cấp CTC head** = đổi class sang `EncDecHybridRNNTCTCBPEModel` (không chỉ thêm head vào RNNT thuần) → là thay đổi lớn hơn, để nấc sau.

---

## Ước lượng bộ nhớ (chọn batch)

Với clip ≤20s: audio 20s → ~250 encoder frame (T); nhãn ~60 token (U); V=1024.
Joint activation 1 batch B (float): `B×T×U×V×4 bytes`.

| B   | Joint activation | ×3 (grad+trung gian) | Kết luận trên 120GB unified            |
| --- | ---------------- | -------------------- | -------------------------------------- |
| 16  | ~0.98 GB         | ~3 GB                | thừa sức                               |
| 32  | ~1.9 GB          | ~6 GB                | thoải mái (khuyến nghị khởi điểm)      |
| 64  | ~3.9 GB          | ~12 GB               | được, nhưng compute-bound sẽ chậm/step |

→ GB10 **dư VRAM** (120GB unified); nghẽn là **băng thông + compute**, không phải bộ nhớ.
Chiến lược đúng: batch vừa (**32**) + **bucketing** (giảm padding) + **grad accum** để đạt effective batch ~256, KHÔNG cố nhồi batch cực lớn (không tăng tốc, chỉ tăng rủi ro).

**Giảm bộ nhớ nếu cần:** `fused_batch_size` của RNNT joint (chia nhỏ tính joint theo thời gian) — bật khi OOM.

---

## Batch + epoch đề xuất theo nấc

Nguyên tắc: data càng nhiều → càng ít epoch (đủ số **bước** mới quan trọng). Dùng bucketing mọi nấc.

| Nấc                         | Giờ ~ | batch | grad accum | eff. batch | epoch | ~steps     | LR (cosine)   |
| --------------------------- | ----- | ----- | ---------- | ---------- | ----- | ---------- | ------------- |
| **S1** đọc sạch             | 55    | 32    | 8          | 256        | 25–40 | ~50k–80k   | 1e-4          |
| **S2** +tự nhiên            | 640   | 32    | 8          | 256        | 6–10  | ~130k–220k | 5e-5 (resume) |
| **S3** +hội thoại/audiobook | 1.660 | 32    | 8          | 256        | 3–5   | ~180k–300k | 3e-5 (resume) |

- LR **giảm dần** qua các nấc (resume) để không phá biểu diễn đã học.
- `warmup_steps` = min(500, total//10) như code hiện có; `min_lr` 1e-6.
- **Best-checkpoint theo val WER** (không phải loss) + có thể bật EMA (~0.999) — xem [07](07_training_lifecycle.md).
- `max_duration=30`, `min_duration=0.3` (nới từ 20s hiện tại để nhận câu audiobook dài; canh OOM).

> Con số epoch/steps là điểm khởi đầu. Sau **smoke đo throughput thật** (bước dưới) sẽ quy ra giờ và chốt lại.

---

## Precision

- **`bf16-mixed`** trên Blackwell (ổn hơn fp16 — RNNT fp16 dễ collapse-to-blank, code note rõ; fp32 an toàn nhưng chậm).
- Smoke phải xác nhận RNNT loss + SpecAugment kernel chạy đúng ở bf16 trên sm_121.

## SpecAugment (bật khi fine-tune)

- Khởi điểm: `freq_masks=2` (width~27), `time_masks=5–10` (width theo tỉ lệ ~0.05). Tăng masking nếu overfit, giảm nếu underfit ở run ngắn.

---

## Việc code / đo

1. `inspect_arch` xác nhận kiến trúc + #param model nền thật.
2. Smoke 50-step S1 (batch 32, bf16, bucketing) trên GB10 → đo **giây/step + GB VRAM thực** → quy ra giờ/epoch.
3. Chốt lại batch (nhích 48/64 nếu compute còn rảnh) + epoch từng nấc theo số đo.
4. Thêm CTC head (hybrid) — thử nghiệm riêng sau khi RNNT-only chạy ổn.

## ✅ Tự kiểm nhanh

1. Cái gì ngốn bộ nhớ nhất khi train RNNT? 2. GB10 nghẽn ở đâu — VRAM hay compute? 3. Vì sao LR giảm dần qua các nấc?

<details><summary>Đáp án</summary>
1. Joint activation `B×T×U×V`. 2. Compute/băng thông (VRAM 120GB dư). 3. Resume từ nấc trước — LR cao sẽ phá biểu diễn đã học; giảm dần để tinh chỉnh.
</details>
