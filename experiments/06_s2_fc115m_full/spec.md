# spec — 06_s2_fc115m_full

Run nấc **S2** curriculum: continue-train từ `s1-fc115m-full.nemo`, thêm 3 domain mới
(FOSD + VLSP + LSVSC) và replay các bộ S1 để chống quên.
Pre-register TRƯỚC khi chốt kết quả. Bối cảnh + chiến lược eval: [`docs/07_dgx_training/09_splits_eval_lineage.md`](../../docs/07_dgx_training/09_splits_eval_lineage.md).

---

## Giả thuyết

- **H1 (chính) — kéo domain S2 xuống rõ:**
  - S1 chưa hề thấy dữ liệu cùng loại vlsp/lsvsc/fosd → hiện đo cao (`vlsp_test` 45,13%, `lsvsc_test` 36,80%).
  - Thêm 3 bộ này vào train → kỳ vọng 3 test tương ứng giảm mạnh (vào in-domain).
- **H2 (giữ nền) — không catastrophic forgetting:**
  - replay vivos/cv/fleurs ×2 → các test đọc-sạch (vivos/cv/fleurs) giữ nguyên, không xấu quá 2 điểm.
- **H0b (kiểm chứng) — cv có thể chạm trần:**
  - `cv_test` bị giới hạn loanword do tokenizer S1 thiếu `f/j/w/z` → cải thiện có thể ít.

---

## Model & lineage

- **Parent:** `s1-fc115m-full.nemo` (`EncDecRNNTBPEModel` 114M).
- **Cách nối:** `restore_from` + `change_vocabulary=false` → **giữ tokenizer + decoder S1** (charset VIVOS).
  - Hệ quả loanword `<unk>` vẫn còn — chấp nhận ở nấc này; rebuild-vocab để run sau.
- Lineage: `v2norm → s1-fc115m-full → s2-fc115m-full`.

---

## Dataset chính xác (từ snapshot run, 2026-07-02)

**Train — 172.999 dòng** (replay S1 ×2 cho tập sạch nhỏ + 3 bộ S2 mới):

| Nguồn | Clip unique (train) | Trọng số | Val |
| --- | --- | --- | --- |
| vivos | 10.849 | ×2 | 571 |
| common_voice_vi | 2.462 | ×2 | 392 |
| fleurs_vi | 2.984 | ×2 | 356 |
| infore1 | 14.188 | ×1 | 746 |
| fosd | 23.668 | ×1 | 1.245 |
| vlsp2020_100h | 52.414 | ×1 | 2.758 |
| lsvsc | 50.139 | ×1 | 5.682 |

**Val — 11.750 clip** (gộp val 7 bộ; chọn checkpoint theo `val_wer`).
- ⚠️ Val này gộp domain khó (vlsp/lsvsc) → `val_wer` cao hơn S1 và **không so được** với `val_wer` của S1 (val đổi theo nấc).

**Test cố định (KHÔNG train) — báo cáo, so với baseline S1:**

| Tập test | Baseline S1 (eval-before) | Vai trò |
| --- | --- | --- |
| `vivos_test` | 10,82% | giám sát quên VIVOS |
| `cv_test` | 31,33% | đọc mic thường |
| `fleurs_test` | 25,05% | benchmark quốc tế |
| `vlsp_test` | 45,13% | **domain S2 — kỳ vọng giảm mạnh** |
| `lsvsc_test` | 36,80% | **domain S2 — kỳ vọng giảm mạnh** |
| `fosd_test` | 28,04% | **domain S2 — kỳ vọng giảm** |

> Test carve của vlsp/lsvsc/fosd đã gỡ khỏi train (held-out) → không rò rỉ.

---

## Cấu hình (config.yaml snapshot trong run-dir)

`configs/s2_natural.yaml` + override: `epochs=8`, `lr=1e-4` cosine (warmup 500 opt-step, min_lr 1e-6),
`batch_size=32`, `accum_grad=8` (**eff batch 256**), `precision=bf16-mixed`, `num_workers=8`,
`max_minutes=1200` (đệm khi share GPU), checkpoint mỗi 300 opt-step (save top-3 theo `val_wer`), EMA tắt.
`max_steps=5408` (8 epoch × 676 opt-step/epoch).

---

## Tiêu chí nghiệm thu (cổng)

- [ ] **Cổng chính H1** — cả `vlsp_test`, `lsvsc_test`, `fosd_test` giảm rõ so baseline S1 (mỗi tập |Δ| ≥ 6 điểm).
- [ ] **Cổng giữ nền H2** — `vivos_test`, `cv_test`, `fleurs_test` KHÔNG xấu quá 2 điểm.
- [ ] **Hội tụ** — `val_wer` giảm đều theo epoch, chọn được best-ckpt (không phân kỳ).
- [ ] Sạch `<unk>` bất thường ở hyp test.

## Dự đoán (pre-registered)

| Ô | Baseline S1 | Dự đoán SAU 8ep |
| --- | --- | --- |
| vlsp_test | 45,13% | **22–28%** |
| lsvsc_test | 36,80% | **24–30%** |
| fosd_test | 28,04% | **18–24%** |
| vivos_test | 10,82% | 10–12% (giữ) |
| cv_test | 31,33% | 29–32% (giữ/nhẹ) |
| fleurs_test | 25,05% | 24–26% (giữ) |

## Confounder / rủi ro

- Tokenizer S1 thiếu `f/j/w/z` → trần loanword cho cv/fleurs. Rebuild-vocab sẽ gỡ.
- Chạy concurrent với job khác trên GPU → chậm ~2x; nếu chạm `max_minutes` trước 8 epoch, model vẫn eval+save (ít epoch hơn).
- Bộ lớn (vlsp/lsvsc) có thể lấn tập sạch → theo dõi vivos_test để phát hiện quên.

Kết quả thực đo: `RESULT.md` (sau khi run xong).
