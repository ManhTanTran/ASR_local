# 🧭 STATE — đang ở đâu, làm gì tiếp

> Đọc ĐẦU MỖI session. Giữ ngắn (~15 dòng), cập nhật số dồn mới nhất, không append lịch sử dài.
> Phỏng `STATE.md` của `numerai/lab_v2`.

**Cập nhật:** 2026-06-23

## Đang ở đâu
- Lab ASR học lại NeMo qua thực nghiệm. Đã chứng minh luồng **zero-cost Kaggle GPU** (local điều phối).
- Model nền chốt: `fastconformer-transducer-large 115M` (offline RNNT). Streaming nemotron loại (collapse).
- Run đầu `vivos-fc115m-v1`: VIVOS WER **20,37%** (có bug `<unk>` chữ hoa, đã sửa).
- **Khung lab vừa dựng** (phỏng lab_v2): `experiments/` (1-exp-1-folder + spec/RESULT), `insight/`,
  `_PROTOCOL`/`_RUN_STANDARD`/`_RUN_QUEUE`, code `asr_lab.registry`/`asr_lab.analytics`.

## Vừa xong
- ✅ `vivos-fc115m-v2norm` (experiment `01`): fix chuẩn hoá + cổng OOV, 50 epoch → **WER 11,93%**
  (từ 20,37%, ΔWER −8,44%, VERDICT THẮNG). Gần sát bậc cộng đồng ~10,8%. RESULT đã điền.
- ✅ Round-trip eval local (CPU, 1000 utt) khớp đúng **11,93%** → pipeline đáng tin.

## Việc kế (theo `insight/proposals/01`)
1. `02_best_ckpt_val` — chọn checkpoint theo val (rẻ), khả năng phá mốc < 11%.
2. Sensing data VI (giờ/domain/license) → `04` gộp Common Voice/VLSP (đòn bẩy **data > model**, đích 6-9%).
3. (tuỳ) `03_specaugment` chống overfit ~15h data.

## Lệnh nhanh
```bash
uv run python -m asr_lab.registry.build_scoreboard          # cập nhật _SCOREBOARD
uv run python -m asr_lab.analytics.compare --base <a> --cand <b>
uv run python -m asr_lab.deploy.kaggle poll --account kyhoolee --kernel asr-ft-fc115m-v2norm
```

## Mốc đích (VIVOS): SOTA ~4,2-4,7% · cộng đồng ~10,8% · mình 20,37% (xem `insight/external/01`)
