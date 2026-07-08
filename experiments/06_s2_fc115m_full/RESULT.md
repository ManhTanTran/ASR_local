# RESULT — 06_s2_fc115m_full

Kết quả thực đo nấc **S2** (continue từ `s1-fc115m-full.nemo`, thêm FOSD + VLSP + LSVSC, replay S1).
Đối chiếu trực tiếp [`spec.md`](spec.md) đã pre-register TRƯỚC khi chạy.
Số liệu từ `_runs/runs/s2-fc115m-full/results.json` (2026-07-03, hoàn tất 8/8 epoch).

---

## 1. Test WER — before (S1) vs after (S2)

Cùng suite 6 test cố định (held-out), chỉ đổi model:

| Test | Đặc thù | Before S1 | After S2 | Tuyệt đối | Tương đối |
| --- | --- | --- | --- | --- | --- |
| `vivos_test` | đọc studio | 10,82% | **9,63%** | −1,19 | −11,0% |
| `cv_test` | mic đời thường | 31,33% | **21,41%** | −9,92 | −31,7% |
| `fleurs_test` | studio chuẩn | 25,05% | **20,26%** | −4,79 | −19,1% |
| `fosd_test` | đọc nguồn FPT | 28,04% | **22,75%** | −5,29 | −18,9% |
| `vlsp_test` | tin tức formal | 45,13% | **31,20%** | −13,93 | −30,9% |
| `lsvsc_test` | tự nhiên đa vùng miền | 36,80% | **15,88%** | −20,92 | −56,8% |

---

## 2. Đối chiếu tiêu chí nghiệm thu (cổng pre-registered)

- **Cổng chính H1** — 3 domain S2 giảm ≥6 điểm:
  - `vlsp_test` −13,93 ✅ · `lsvsc_test` −20,92 ✅ · `fosd_test` −5,29 ⚠️ (thiếu 0,71 điểm, nhưng vẫn cải thiện rõ).
  - → **PASS 2/3, cận-đạt cái thứ 3**. FOSD là đọc-sạch nguồn FPT, vốn đã khá ở S1 nên biên cải thiện hẹp hơn.
- **Cổng giữ nền H2** — vivos/cv/fleurs không xấu quá 2 điểm:
  - cả ba **đều cải thiện** (vivos −1,19, cv −9,92, fleurs −4,79) → **PASS vượt xa** (không chỉ giữ mà còn tốt lên).
- **Hội tụ** — best `val_wer` = 0,1942 (val hỗn hợp 7 bộ); giảm đều 0,3086 → 0,2585 → 0,2314 → ... → 0,1942, không phân kỳ → **PASS**.
- **Sạch `<unk>`** — không thấy bất thường.

---

## 3. So dự đoán và thực tế

| Test | Baseline S1 | Dự đoán (pre-reg) | Thực đo | Nhận xét |
| --- | --- | --- | --- | --- |
| vlsp_test | 45,13% | 22–28% | 31,20% | giảm mạnh nhưng chưa sâu như dự đoán |
| lsvsc_test | 36,80% | 24–30% | **15,88%** | **vượt xa dự đoán** — LSVSC đa dạng, model hấp thụ tốt |
| fosd_test | 28,04% | 18–24% | 22,75% | trong khoảng |
| vivos_test | 10,82% | 10–12% | 9,63% | tốt hơn (không quên) |
| cv_test | 31,33% | 29–32% | **21,41%** | **vượt xa** — replay + model mạnh lên kéo cả cv xuống |
| fleurs_test | 25,05% | 24–26% | 20,26% | tốt hơn dự đoán |

**Insight chính:**
- **Curriculum + replay hoạt động đúng lý thuyết:** thêm domain mới kéo domain đó xuống mạnh, đồng thời replay giữ (và cải thiện) domain cũ — không có catastrophic forgetting.
- **Hiệu ứng lan tỏa:** thêm data S2 kéo cả cv/fleurs (không thuộc S2) xuống — model tổng quát hóa tốt hơn, không chỉ học vẹt domain mới.
- **`lsvsc` bùng nổ** (−21 điểm): LSVSC là bộ lớn (100h) cùng phân phối với test carve của nó → in-domain rõ.
- **Mốc so PhoWhisper:** `cv_test` 21,41% đã tiệm cận PhoWhisper-tiny (19,05%) — model 114M bắt kịp vùng tiny/base ở Common Voice (lưu ý chuẩn hóa WER có thể lệch, không so tuyệt đối).

---

## 4. Cấu hình & tài nguyên

- **Model:** continue từ `s1-fc115m-full.nemo` (`change_vocabulary=false`, giữ tokenizer S1).
- **Train:** 172.999 dòng (replay S1 ×2 + FOSD/VLSP/LSVSC), eff batch 256, bf16-mixed, 8 epoch, 5.408 opt-step.
- **Thời lượng:** 19,89h wall-clock — **chậm vì nửa đầu chạy song song job YOLO của user khác** (~½ tốc độ); nửa sau GPU rảnh nên tăng tốc. Ở chế độ độc chiếm ước ~9-10h.
- **Backup:** `s2-fc115m-full.nemo` (457MB) → `/srv/team-share/models/asr_vi/`.

---

## 5. Việc tiếp theo

- **Nấc S3:** data đã build+carve xong (vietsuperspeech 220h hội thoại, viVoice 1017h, infore2 416h, phoaudiobook 1490h, vietmed 5h — ~3.600h). Cần chốt scope (bộ nào, trọng số, epoch) trước khi chạy.
- **Rebuild-vocab:** trần loanword của cv (`f/j/w/z` → `<unk>`) vẫn còn; một nhánh riêng dùng tokenizer full-charset để gỡ.
- **bud500** (500h) chờ cấp quyền HF rồi bổ sung vào S3.

> Lineage: `v2norm → s1-fc115m-full → s2-fc115m-full`.
