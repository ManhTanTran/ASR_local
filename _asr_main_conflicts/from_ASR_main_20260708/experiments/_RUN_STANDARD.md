# 📐 _RUN_STANDARD — chuẩn 1-experiment-1-folder

> Phỏng theo `numerai/lab_v2` (`04_run_standard.md`), chế cho lab ASR. Mục tiêu: **tích luỹ chặt
> chẽ qua thời gian** — mỗi run là 1 toạ độ chắc chắn, truy ngược được, so sánh được.

## Quy ước cốt lõi

- **1 experiment = 1 folder** `experiments/NN_concept_name/`. Không rải code/kết quả 3 nơi.
- **Số máy nằm ở `artifacts/runs/<run_id>/`** (status.json + results.json + provenance.json + .nemo),
  do adapter Kaggle ghi. Folder experiment chỉ chứa **ý đồ + phán quyết**, KHÔNG copy số tay —
  luôn link về `run_id`.
- **Train một lần, phân tích vô hạn**: mọi so sánh đọc lại artifact (`asr_lab.analytics.*`), không train lại.

## Mỗi folder experiment gồm 3 file

| File | Khi nào viết | Nội dung |
| --- | --- | --- |
| `spec.md` | **TRƯỚC khi chạy** (pre-register) | Giả thuyết H1/H0 + dự đoán số + tiêu chí nghiệm thu cứng + confounder cần loại + scope |
| `config.md` | khi phóng run | Lệnh + args đã chạy (model nền, epoch, batch, vocab, lr, precision), run_id, account Kaggle |
| `RESULT.md` | **SAU khi pull về** | Bảng WER/RTF trước-sau + verdict 3 cổng (từ `analytics.compare`) + insight + hướng kế. Link `artifacts/runs/<run_id>/` |

## Luồng một vòng (bám workflow 0-5 bên `_1_dagfl_`)

```
insight/proposals (ý tưởng) -> spec.md (chốt giả thuyết, KHÔNG sửa sau)
   -> config.md + phóng Kaggle (asr_lab.deploy.kaggle push)
   -> artifacts/runs/<id>/ (pull về)
   -> asr_lab.analytics.compare/report + registry.build_scoreboard
   -> RESULT.md (verdict) -> insight/decisions (chốt) -> ý tưởng vòng sau
```

## Đặt tên

- `NN_concept_name` — `NN` số thứ tự (00,01,...); `concept_name` = đòn bẩy đang thử
  (vd `vivos_fc115m_norm`, `add_commonvoice`, `ctc_vs_rnnt`).
- `run_id` trong artifact nên trùng/*gần* tên folder để dò nhanh (vd folder `01_vivos_fc115m_norm`
  ↔ run_id `vivos-fc115m-v2norm`).

## Cổng kỷ luật (hard gate — tự động, đặt ở ĐẦU)

- **Cổng chuẩn hoá/OOV**: `assert_no_oov` trong `finetune_vivos` chặn lệch tokenizer↔nhãn TRƯỚC train
  (xem skill `.agent/skills/04_data_infra/data_prep/normalize-text-data.md`).
- **Cổng chi phí**: ước GPU-giờ trước khi push (quota Kaggle ~30h/tuần/account) — ghi ở `_RUN_QUEUE.md`.
- **Cổng no-leak**: speaker train/test tách rời; chọn best-checkpoint theo **val**, không đụng test.
