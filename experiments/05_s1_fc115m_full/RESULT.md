# RESULT — 05_s1_fc115m_full

Kết quả thực đo của run huấn luyện THẬT đầu tiên trên DGX GB10 (nấc S1 curriculum).
Đối chiếu trực tiếp với [`spec.md`](spec.md) đã pre-register TRƯỚC khi chạy.
Số liệu lấy từ `_runs/runs/s1-fc115m-full/results.json` (2026-07-02, hoàn tất 20/20 epoch).

---

## 1. Con số chính (Test WER cố định)

Hai tập snapshot trong run gốc (before/after):

| Tập test | Trước (v2norm) | Sau S1 | Tuyệt đối | Tương đối |
| --- | --- | --- | --- | --- |
| `cv_test` (1.225 clip) | 42,95% | **31,32%** | −11,63 điểm | −27,1% |
| `fleurs_test` (844 clip) | 31,47% | **25,04%** | −6,43 điểm | −20,4% |

- **`wer_before` khớp baseline exp04** (cv_test 42,95%) → xác nhận đường ống eval đo đúng, không lệch.
- Cả hai tập đều giảm rõ, kể cả FLEURS (vốn v2norm đã khá) — trái với giả thuyết H0b dự đoán FLEURS gần như phẳng.

### 1b. Suite đầy đủ 6 test (eval-lại bằng `asr_lab.eval_suite`)

Run gốc chỉ snapshot cv+fleurs. Đo lại `s1-fc115m-full.nemo` trên đủ suite để có tranh cross-domain
(`_runs/runs/s1-fc115m-full/suite_eval.json`, 2026-07-02):

| Tập test | WER sau S1 | Đặc thù | Nhận xét |
| --- | --- | --- | --- |
| `vivos_test` | **10,82%** | đọc studio | Giữ vững, còn **tốt hơn** v2norm 11,93% → không quên VIVOS |
| `cv_test` | 31,33% | mic đời thường | Khớp run gốc |
| `fleurs_test` | 25,05% | studio chuẩn | Khớp run gốc |
| `fosd_test` | 28,04% | đọc nguồn FPT | Khá, dù chưa train FOSD |
| `lsvsc_test` | 36,80% | tự nhiên đa vùng miền | Domain S2 chưa train → còn cao |
| `vlsp_test` | 45,13% | đọc tin tức formal | Domain S2 chưa train → cao nhất |

**Insight rút ra:**
- **Không có catastrophic forgetting:** `vivos_test` không những giữ mà còn nhích tốt hơn baseline v2norm — replay ×3 trong S1 hiệu quả.
- **Đọc-sạch đã bão hoà tương đối:** vivos/cv/fleurs/fosd nằm vùng 10–31%; thêm data đọc-sạch nữa lợi ích giảm dần.
- **Khoảng trống rõ nằm ở domain S2:** `vlsp_test` 45% và `lsvsc_test` 37% là hai tập S1 chưa hề thấy dữ liệu cùng loại → đây chính là mục tiêu nấc S2 kéo xuống.
- **Trần loanword của cv_test** (31%) một phần do tokenizer v2norm thiếu `f/j/w/z`; cần run rebuild-vocab để gỡ.
- Đây là **baseline cross-domain sạch** để đo tiến bộ S2/S3: cùng 6 tập test cố định, chỉ đổi model.

---

## 2. Đối chiếu tiêu chí nghiệm thu (cổng pre-registered)

- [x] **Cổng chính H1** — `cv_test` sau < 42,95% với |Δ| ≥ 8%:
  - đạt 31,32%, Δ = 11,63 điểm → **PASS**, còn tốt hơn mép dưới dự đoán (32–36%).
- [x] **Cổng phụ FLEURS** — không xấu quá 2% so 31,47%:
  - thực tế 25,04%, **cải thiện 6,43 điểm** → **PASS vượt kỳ vọng** (dự đoán chỉ 30–32%).
- [x] **Hội tụ** — `val_wer` giảm đều, chọn được best-ckpt:
  - best `val_wer` = 0,1473 tại step 4000 (~epoch 16,2); không phân kỳ.
- [x] **Sạch `<unk>`** — không thấy `<unk>` bất thường trong hyp test.

Kết luận: **cả 4 cổng PASS**, hai cổng vượt dự đoán.

---

## 3. So dự đoán và thực tế

| Ô | v2norm | Dự đoán (pre-reg) | Thực đo | Nhận xét |
| --- | --- | --- | --- | --- |
| cv_test | 42,95% | 32–36% | **31,32%** | tốt hơn mép dưới ~0,7 điểm |
| fleurs_test | 31,47% | 30–32% | **25,04%** | vượt xa, giảm ~5–7 điểm ngoài dự đoán |

- **Vì sao tốt hơn dự đoán:** thêm FLEURS + InfoRE1 vào mix (ngoài VIVOS + CV của exp04) cấp thêm đa dạng nguồn đọc-sạch, kéo cả hai domain xuống cùng lúc.
- **Trần còn lại của cv_test** một phần do tokenizer v2norm thiếu `f/j/w/z` → loanword vẫn thành `<unk>` ngầm; run rebuild-vocab sau sẽ gỡ giới hạn này.

---

## 4. Cấu hình và tài nguyên thực đo

- **Model:** continue từ `vivos-fc115m-v2norm.nemo` (`change_vocabulary=false`, giữ tokenizer VIVOS).
- **Train:** 63.073 dòng (30.483 clip unique, upsample ×3 cho vivos/cv/fleurs), eff batch 256, bf16-mixed.
- **Thời lượng:** 20 epoch, 4.940 optimizer-step, **27.143 giây ≈ 7,54 giờ** trên 1× GB10.
- **Checkpoint giữ lại (top theo val_wer):**
  - step 4000 → `val_wer` 0,1473 (best)
  - step 4250 → 0,1494
  - step 3250 → 0,1517
- **Bàn giao:** `s1-fc115m-full.nemo` (457 MB) đã backup về `/srv/team-share/models/asr_vi/`.

---

## 5. Hạn chế và việc tiếp theo

- **Eval hiện chỉ 2 tập** (cv + fleurs) vì run snapshot suite eval cũ:
  - chưa có số trên `vivos_test` (giám sát quên VIVOS), `vlsp/lsvsc/fosd_test` (domain S2).
  - → Nên chạy **eval-lại `s1` trên suite 6 test** đã carve (GPU đang rảnh) để có tranh cross-domain đầy đủ.
- **Nấc tiếp `s2`:** continue từ `s1-fc115m-full.nemo`, thêm FOSD + VLSP + LSVSC (đã build + carve test xong).
- **Rebuild-vocab:** một nhánh riêng dùng tokenizer full-charset `vi_1024_s1` để gỡ trần loanword — cân nhắc sau S2.

> Lineage sau run này: `v2norm → s1-fc115m-full` (nhánh DGX).
>
> Chi tiết gia phả và chiến lược eval: [`docs/07_dgx_training/09_splits_eval_lineage.md`](../../docs/07_dgx_training/09_splits_eval_lineage.md).
