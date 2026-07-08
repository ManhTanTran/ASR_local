# RESULT — 01_vivos_fc115m_norm

**Artifact:** `artifacts/runs/vivos-fc115m-v2norm/` · ✅ **COMPLETE** (50/50 epoch, ~4,3h train).

## Số

| | WER | RTF |
| --- | --- | --- |
| Trước (đầu English) | 100,13% | 0,007 |
| **Sau fine-tune** | **11,93%** | 0,068 |

So với run `00` (`vivos-fc115m-v1`, 20,37%): **ΔWER −8,44%** (lệnh `analytics.compare`).

**Round-trip xác nhận:** eval lại `.nemo` trên **CPU local** (1000 utt, `eval.vivos`, normalize_vi)
ra **WER 11,93%** — khớp tuyệt đối số GPU. Pipeline train→eval đáng tin, không lệch harness.
Cặp ref/hyp sạch `<unk>`/`⁇`, đủ dấu; lỗi còn lại chỉ là nhầm âm (cây/ca, lim/Liêm, tui/tôi).

## Verdict (3 cổng — `analytics.compare --base vivos-fc115m-v1 --cand vivos-fc115m-v2norm`)

- [x] Cổng OOV PASS (0% — verify local trước khi phóng; gate `assert_no_oov` trong code)
- [x] ΔWER vs `00` âm, |ΔWER| ≥ 1% → **−8,44%** ✔ (cổng tin-cậy proxy)
- [x] RTF không hồi quy (0,062 → 0,068, dưới ngưỡng 0,30) ✔
- [x] Hết `<unk>` ở chữ hoa: log không còn `⁇`/`<unk>` ✔

→ **VERDICT: THẮNG.**

## Insight

- Fix lệch chuẩn hoá tokenizer↔nhãn (`<unk>` chữ hoa) là đòn bẩy **rất rẻ mà hiệu quả lớn**: chỉ
  normalize nhãn train, WER tụt **gần một nửa** (20,37% → 11,93%) — H1 đúng.
- Đã **chạm sát bậc 3 "KPI cộng đồng"** (wav2vec2-base ~10,8%); chỉ còn ~1% nữa.
- Confounder đã ghi: vòng này đổi 2 biến (fix `<unk>` + epoch 40→50). Phần lớn cải thiện do fix
  (log sạch `<unk>`), nhưng chưa cô lập tuyệt đối phần đóng góp của 10 epoch thêm.

## Hướng kế

→ `02_best_ckpt_val` (chọn checkpoint theo val, rẻ) để khả năng phá mốc < 11%.
→ Sau đó `04_add_commonvoice` — đòn bẩy **data > model** đưa xuống vùng 6-9% (xem `insight/proposals/01`).
