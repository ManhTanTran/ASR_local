# RESULT — 00_vivos_fc115m

**Artifact:** `artifacts/runs/vivos-fc115m-v1/` · **report:** `uv run python -m asr_lab.analytics.report --run-id vivos-fc115m-v1`

## Số

| | WER | RTF |
| --- | --- | --- |
| Trước (đầu English) | 100,13% | 0,008 |
| **Sau fine-tune (eval Kaggle GPU)** | **20,37%** | 0,062 |
| Sau (eval CPU local, model kéo về) | 20,37% | 0,049 |

40 epoch / 27.800 step / ~3,8h train.

## Verdict

- **H1 ĐÚNG**: model offline 115M hội tụ được (khác hẳn nemotron streaming → xem `insight/decisions/01`).
- Round-trip GPU→CPU **khớp tuyệt đối** (cùng 20,37%) → luồng deploy đáng tin.
- Theo `_PROTOCOL` thang baseline: đạt **bậc 2 (công sức tối thiểu)**, CHƯA tới bậc 3 cộng đồng (~10,8%).

## Insight

- ✅ Luồng zero-cost (local điều phối ↔ Kaggle GPU train) chạy trọn vẹn.
- ⚠️ Lỗi đặc trưng "rớt ký tự đầu câu" → truy ra **không phải lỗi audio** mà **lệch chuẩn hoá
  tokenizer↔nhãn** (`<unk>` ở chữ hoa). Chi tiết: `docs/03_models/03_fastconformer/01_vi_finetune_error_analysis.md`.

## Hướng kế

→ experiment `01_vivos_fc115m_norm`: sửa chuẩn hoá + cổng OOV, kỳ vọng phá lỗi đầu câu, kéo WER < 11%.
