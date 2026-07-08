# spec — 07_s3_fc115m_full

Run nấc **S3** curriculum: hội thoại/tự nhiên (mục tiêu callbot FCI).
Continue từ `s2-fc115m-full.nemo`, thêm domain hội thoại (vietsuperspeech + viVoice + bud500),
audiobook liều nhỏ (đo generalize), replay S1/S2. Pre-register TRƯỚC khi chốt kết quả.
Bối cảnh + eval: [`docs/07_dgx_training/09_splits_eval_lineage.md`](../../docs/07_dgx_training/09_splits_eval_lineage.md).

---

## Giả thuyết

- **H1 (chính) — mở khoá domain hội thoại:**
  - S2 chưa thấy dữ liệu hội thoại → `vietsuperspeech_test` (podcast/phỏng vấn, gần callbot nhất) hiện đo cao.
  - Thêm vietsuperspeech + viVoice + bud500 → kỳ vọng `vietsuperspeech_test` và `bud500_test` **giảm mạnh** (vào in-domain).
  - Đây là **metric callbot** — mục tiêu tối thượng của nấc này.
- **H2 (giữ nền) — không quên đọc-sạch/formal:**
  - replay vivos/cv/fleurs ×2 + vlsp/lsvsc/fosd → 6 test cũ không xấu quá 2 điểm.
- **H3 (audiobook generalize) — liều nhỏ có hại không:**
  - phoaudiobook + infore2 chỉ cap nhỏ (~13% mix) → thêm đa dạng đọc-dài mà không kéo lệch hội thoại.

---

## Model & lineage

- **Parent:** `s2-fc115m-full.nemo` (`change_vocabulary=false`, giữ tokenizer S2/charset VIVOS).
- Lineage: `v2norm → s1-fc115m-full → s2-fc115m-full → s3-fc115m-full`.
- Hệ quả loanword `<unk>` vẫn còn (rebuild-vocab để run riêng sau).

---

## Dataset (từ dry-run stages, 2026-07-03)

**Train ~503k dòng (chưa bud500) → ~623k (khi bud500 build xong)**, cap + upsample:

| Nhóm | Nguồn | Clip (sau cap) | Trọng số |
| --- | --- | --- | --- |
| **Hội thoại (chính)** | vietsuperspeech | 59.656 | ×2 |
| | vivoice | cap 150.000 | ×1 |
| | bud500 | cap 120.000 | ×1 (khi build) |
| **Audiobook (liều nhỏ)** | phoaudiobook | cap 40.000 | ×1 |
| | infore2_audiobooks | cap 30.000 | ×1 |
| **Replay S1/S2** | vivos / cv / fleurs | 10.849 / 2.462 / 2.984 | ×2 |
| | fosd / vlsp / lsvsc / infore1 | 23.668 / 52.414 / 50.139 / 14.188 | ×1 |

- `cap` = subsample tối đa N clip (lấy đều tay, tất định) — để viVoice/bud500 không nuốt vietsuperspeech, audiobook không áp đảo.
- Tỉ lệ: hội thoại ~382k (chính) · audiobook ~68k (~13%, đo generalize) · replay ~173k.

**Test cố định — suite MỞ RỘNG 9 test** (6 cũ + 3 mới):
- Mới: `vietsuperspeech_test` (**callbot**), `bud500_test` (3 vùng miền), `vietmed_test` (y tế — probe generalize thuần, KHÔNG train).
- Baseline S2 cho 3 test mới: **đo ở eval-before của chính run S3** (S2 chưa từng đo 3 test này).

---

## Cấu hình

`configs/s3_conversational.yaml` + override `data.root`, `run.artifacts_dir`.
`epochs=4`, `lr=1e-4` cosine, `batch=32`, `accum=8` (eff 256), bf16-mixed, `max_minutes=1000`,
checkpoint mỗi 500 opt-step (top-3 theo `val_wer`). ~1966 step/epoch (chưa bud500) → ~2470 (có bud500).

---

## Tiêu chí nghiệm thu (cổng)

- [ ] **Cổng chính H1** — `vietsuperspeech_test` giảm ≥ 10 điểm so eval-before; `bud500_test` giảm rõ.
- [ ] **Cổng giữ nền H2** — 6 test cũ không xấu quá 2 điểm (đặc biệt lsvsc 15,88% và cv 21,41%).
- [ ] **Hội tụ** — `val_wer` giảm đều, chọn được best-ckpt.
- [ ] Sạch `<unk>` bất thường.

## Dự đoán (pre-registered)

| Test | Baseline (S2 / eval-before) | Dự đoán SAU 4ep |
| --- | --- | --- |
| vietsuperspeech_test | cao (đo ở before, ~40-50%?) | **20–28%** |
| bud500_test | cao (before) | **18–26%** |
| vietmed_test | cao (probe, không train) | cải thiện nhẹ (generalize) |
| vivos_test | 9,63% | 9–11% (giữ) |
| cv_test | 21,41% | 19–22% (giữ/nhẹ) |
| lsvsc_test | 15,88% | 14–17% (giữ) |
| vlsp_test | 31,20% | 28–32% (giữ) |

## Confounder / rủi ro

- Tokenizer thiếu `f/j/w/z` → trần loanword (viVoice YouTube nhiều loanword). Rebuild-vocab gỡ sau.
- viVoice/bud500 auto-label (pseudo) → nhiễu nhãn có thể hạn chế trần.
- Chạy có thể chạm `max_minutes` nếu share GPU → vẫn eval+save (ít epoch hơn).
- `vietmed` chỉ ở eval (không train) → là probe generalize thuần, không kỳ vọng giảm mạnh.

Kết quả thực đo: `RESULT.md` (sau khi run xong).
