# spec — 05_s1_fc115m_full

Run huấn luyện THẬT đầu tiên trên DGX GB10: nấc S1 curriculum (đọc sạch), continue từ ckpt VIVOS.
Pre-register TRƯỚC khi chốt kết quả. Bối cảnh + chiến lược eval: [`docs/07_dgx_training/09_splits_eval_lineage.md`](../../docs/07_dgx_training/09_splits_eval_lineage.md).

---

## Giả thuyết

**H1 (chính):** continue-train từ `v2norm` trên toàn corpus S1 (~51h, thêm FLEURS + InfoRE1 ngoài VIVOS+CV)
kéo **Test WER trên `cv_test` xuống rõ** so với v2norm, nhờ nhiều data + đa nguồn.
Run thử 2 epoch đã cho CV 42,95%→37,58% (−5,4%); đủ 20 epoch kỳ vọng xuống sâu hơn.

**H0b (kiểm chứng):** FLEURS test có thể **không cải thiện nhiều** (v2norm vốn khá trên đọc-sạch; run thử 2 epoch ~phẳng 31,5%).

---

## Model & lineage

- **Parent:** `vivos-fc115m-v2norm.nemo` (`EncDecRNNTBPEModel` 114M, encoder ConformerEncoder 512/17 lớp, vocab 1024).
- **Cách nối:** `restore_from` + `change_vocabulary=false` → **giữ tokenizer + decoder v2norm** (charset VIVOS).
  - Hệ quả: clip có `f/j/w/z` (loanword CV/FLEURS/InfoRE) bị 1 phần thành `<unk>` — chấp nhận ở nấc này; rebuild-vocab để run sau.
- **KHÔNG** kế thừa `vivos-cv` (exp04, nhánh Kaggle song song).

---

## Dataset chính xác (từ snapshot run, 2026-07-02)

**Train — 63.073 dòng / 30.483 clip unique** (upsample ×3 cho tập sạch nhỏ, chống InfoRE1 nuốt):

| Nguồn | Clip unique | Trọng số |
| --- | --- | --- |
| vivos | 10.849 | ×3 |
| common_voice_vi | 2.462 | ×3 |
| fleurs_vi | 2.984 | ×3 |
| infore1 | 14.188 | ×1 |

**Val — 2.065 clip** (chọn checkpoint theo `val_wer`): vivos 571 + infore1 746 (cut-tail 5%) + cv 392 + fleurs 356 (val official).

**Test cố định (KHÔNG train, KHÔNG chọn ckpt) — báo cáo:**
- `cv_test` = `common_voice_vi.test.jsonl` — 1.225 clip (~1,3h).
- `fleurs_test` = `fleurs_vi.test.jsonl` — 844 clip (~2,92h).

> Không rò rỉ: cv/fleurs test là split held-out official, tách hẳn train.
> ⚠️ VIVOS test (1.000) KHÔNG nằm trong eval run này (không giám sát regression VIVOS ở nấc S1).

---

## Cấu hình (config.yaml snapshot trong run-dir)

`configs/s1_clean.yaml` + override: `epochs=20`, `lr=1e-4` cosine (warmup ~494 opt-step, min_lr 1e-6),
`batch_size=32`, `accum_grad=8` (**eff batch 256**), `precision=bf16-mixed`, `num_workers=8`,
`max_minutes=540`, checkpoint mỗi 250 opt-step (save top-3 theo `val_wer`), EMA tắt.
Tốc độ đo thật: ~0,68s/micro-batch → **~22,5 phút/epoch**, ~247 opt-step/epoch.

---

## Tiêu chí nghiệm thu (cổng)

- [ ] **Cổng chính H1** — `cv_test` SAU < v2norm baseline 42,95%, |Δ| ≥ 8% (kỳ vọng vùng ~33-35% hoặc thấp hơn).
- [ ] **Cổng phụ FLEURS** — `fleurs_test` SAU không xấu quá 2% so 31,47%.
- [ ] **Hội tụ** — `val_wer` giảm đều theo epoch, best-ckpt chọn được (không phân kỳ).
- [ ] Sạch `<unk>` bất thường ở hyp test (soi log wer).

## Dự đoán (pre-registered)

| Ô | v2norm (trước) | Dự đoán SAU 20ep |
| --- | --- | --- |
| cv_test | 42,95% | **32–36%** |
| fleurs_test | 31,47% | 30–32% |

## Confounder / rủi ro

- Tokenizer v2norm thiếu `f/j/w/z` → `<unk>` ngầm hạn chế trần CV (loanword). Rebuild-vocab sẽ gỡ.
- Val đổi theo nấc → chỉ dùng chọn ckpt, không so nấc (dùng test cố định).
- Chưa có test cho domain S2/S3 → nấc sau cần carve test slice trước khi train.

Kết quả thực đo: `RESULT.md` (sau khi run xong).
