# RESULT — 07_s3_fc115m_full

Kết quả thực đo nấc **S3** (hội thoại/tự nhiên, mục tiêu callbot). Continue từ `s2-fc115m-full.nemo`.
Đối chiếu [`spec.md`](spec.md) pre-register. Số liệu từ `_runs/runs/s3-fc115m-full/results.json` (2026-07-04).

> ✅ Run đã hoàn tất **đủ 4 epoch** (`completed_epochs=4`, `global_step=7790`, `n_train=623.311`).
> Run gồm 2 chặng: chặng đầu ~2,5 epoch (chạm `max_time` do nửa đầu share GPU với job YOLO),
> sau đó **resume trọn epoch cuối** trên GPU đã rảnh (`train_sec=15.655` ≈ 4,35h cho chặng resume).
> Số dưới đây là **bản 4-epoch cuối cùng** — đã thay bản 2,5-epoch (giữ cột 2,5ep để so tiến bộ).

---

## 1. Test WER — before (S2) vs after (S3 đủ 4 epoch), suite 9 test

| Test | Đặc thù | Before S2 | 2,5ep | **After S3 (4ep)** | Tuyệt đối | Tương đối |
| --- | --- | --- | --- | --- | --- | --- |
| `vietsuperspeech_test` | **hội thoại (callbot)** | 40,00% | 23,38% | **22,87%** | −17,13 | **−42,8%** |
| `bud500_test` | tự nhiên 3 vùng miền | 16,38% | 7,23% | **6,73%** | −9,65 | **−58,9%** |
| `vietmed_test` | y tế (probe, KHÔNG train) | 31,01% | 26,72% | **26,38%** | −4,63 | −14,9% |
| `cv_test` | mic đời thường | 21,42% | 17,69% | **17,19%** | −4,23 | −19,7% |
| `fleurs_test` | studio chuẩn | 20,25% | 16,93% | **16,46%** | −3,79 | −18,7% |
| `vlsp_test` | tin tức formal | 31,20% | 25,35% | **24,81%** | −6,39 | −20,5% |
| `lsvsc_test` | tự nhiên đa miền | 15,87% | 13,36% | **13,12%** | −2,75 | −17,3% |
| `vivos_test` | đọc studio | 9,63% | 8,73% | **8,47%** | −1,16 | −12,0% |
| `fosd_test` | đọc FPT | 22,75% | 20,56% | **19,96%** | −2,79 | −12,3% |

**Epoch cuối (2,5 → 4) kéo TOÀN BỘ 9 test xuống thêm ~0,2-0,6 điểm** — không test nào xấu đi.
Xác nhận val chưa plateau ở 2,5 epoch là đúng: còn dư địa, train thêm là có lời đều tay.

---

## 2. Đối chiếu tiêu chí nghiệm thu (cổng pre-registered)

- **Cổng chính H1** — `vietsuperspeech_test` giảm ≥10 điểm, `bud500_test` giảm rõ:
  - vietsuperspeech **−17,13** ✅✅ · bud500 **−9,65** ✅✅ → **PASS mạnh**.
- **Cổng giữ nền H2** — 6 test cũ không xấu quá 2 điểm:
  - cả 6 **đều cải thiện** (fosd −2,79 · vlsp −6,39 · cv −4,23 · fleurs −3,79 · lsvsc −2,75 · vivos −1,16) → **PASS vượt xa** (không quên gì).
- **Cổng generalize H3 (audiobook liều nhỏ)** — vietmed (probe không train) **giảm −4,63** → data hội thoại generalize sang cả y tế mà audiobook liều nhỏ không kéo lệch.
- **Hội tụ** — val_wer 0,1916 (ep0) → 0,1689 (ep~2) → cải thiện tiếp ở epoch cuối (mọi test giảm thêm), chọn được best-ckpt theo `val_wer`.

→ **Tất cả cổng PASS, phần lớn vượt dự đoán, với đủ 4 epoch.**

---

## 3. So dự đoán

| Test | Baseline S2 | Dự đoán | Thực đo (4ep) | |
| --- | --- | --- | --- | --- |
| vietsuperspeech_test | 40,00% | 20–28% | **22,87%** | khớp tâm dự đoán |
| bud500_test | 16,38% | 18–26% | **6,73%** | **vượt xa** (dự đoán bảo thủ) |
| cv_test | 21,42% | 19–22% | 17,19% | tốt hơn |
| lsvsc_test | 15,87% | 14–17% | 13,12% | tốt hơn |

**Insight:**
- **Curriculum hoạt động trọn vẹn:** thêm domain hội thoại kéo callbot metric xuống mạnh, đồng thời replay giữ + cải thiện toàn bộ nền cũ.
- **Hiệu ứng lan tỏa cực rộng:** data hội thoại kéo cả cv/fleurs/vlsp/vietmed (không thuộc train S3 hoặc chỉ probe) xuống — model tổng quát hóa thật, không học vẹt.
- **bud500 6,73%** — bộ 3 vùng miền chất lượng cao, in-domain sau khi thêm vào train.
- **Mốc so PhoWhisper:** cv_test 17,19% ≈ PhoWhisper-base 74M (16,19%) — model 114M tự-gom-data bắt kịp vùng base (lưu ý normalizer khác nhau).
- **Giá trị của epoch cuối:** chỉ thêm ~4,35h train đã hạ đều 9 test — đầu tư đúng vì val chưa plateau.

---

## 4. Cấu hình & tài nguyên

- **Model:** continue từ `s2-fc115m-full.nemo` (`change_vocabulary=false`).
- **Train:** 623.311 dòng (hội thoại vietsuperspeech×2 + viVoice cap150k + bud500 cap120k · audiobook liều nhỏ · replay S1/S2), eff batch 256, bf16.
- **Thời lượng:** chặng đầu ~16,7h (chạm cap, ~2,5 epoch) + chặng resume ~4,35h = **~21h tổng** cho đủ 4 epoch.
- **Backup:** `s3-fc115m-full.nemo` (457MB) → `/srv/team-share/models/asr_vi/` (đã đè bản 2,5-epoch).

---

## 5. Bài học vận hành (resume)

- **Lightning lưu đồng hồ `max_time` VÀO checkpoint.** Resume mà giữ nguyên `max_minutes` cũ → nạp lại thấy elapsed cũ đã ≥ cap → dừng NGAY, không train step nào (`train_h=0`, WER y hệt). Lần resume đầu đã dính bẫy này.
- **Cách gỡ:** khi resume phải **nâng `max_minutes`** cao hơn elapsed đã tích trong checkpoint (đặt 3000 phút). Elapsed 16h43 << 50h → train tiếp bình thường.
- **Phòng ngừa gốc:** flagship run cần **độc chiếm GPU TRƯỚC khi launch** (kill job share), hoặc set `max_minutes` dư ngay từ đầu — nửa đầu share GPU đã cắt mất ~1,5 epoch của chặng một.

---

## 6. Lineage & việc tiếp

- Lineage: `v2norm → s1 → s2 → s3-fc115m-full` (đủ 4 epoch).
- **Rebuild-vocab:** trần loanword `f/j/w/z` (viVoice YouTube nhiều loanword) vẫn còn — nhánh riêng gỡ sau, kỳ vọng gỡ trần cho cv/fleurs.
- **Còn dư địa nhẹ:** giảm biên mỗi epoch đang thu hẹp (0,5 điểm/epoch cuối) → thêm epoch nữa lợi ít; ưu tiên rebuild-vocab hoặc thêm data hội thoại thật (gần callbot) sẽ có lời hơn.
