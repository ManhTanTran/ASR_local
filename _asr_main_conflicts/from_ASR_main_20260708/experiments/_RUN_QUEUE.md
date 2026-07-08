# 🧮 _RUN_QUEUE — hàng đợi run + chi phí GPU

> Cổng chi phí trước khi phóng GPU. Quota Kaggle: **~30h GPU/tuần/account**, 4 account
> (`kyhoolee`, `acc2`, `acc3`, `acc4`); session GPU tối đa ~12h. Phỏng `_RUN_QUEUE` của lab_v2.

## Chi phí tham chiếu (đo thật)

| Việc | Thời lượng | Ghi chú |
| --- | --- | --- |
| pip torch cu118 + nemo (mỗi kernel) | ~10-15 phút | bắt buộc, P100 sm_60 |
| fine-tune fc115m 40 epoch full VIVOS | ~3,8h | run `vivos-fc115m-v1` |
| eval test 1000 câu | ~4 phút GPU | trong cùng kernel |

→ Một full run ≈ **0,3h setup + ~5h train** (50 epoch) ⇒ gọn trong 1 session, ~5h/30h quota tuần.

## Đang chạy

| run_id | account | việc | trạng thái |
| --- | --- | --- | --- |
| `vivos-fc115m-v2norm` | kyhoolee | fc115m 50ep + fix `<unk>` + cổng OOV | RUNNING (watcher nền) |

## Hàng đợi (đề xuất, theo nấc thang `_PROTOCOL`)

| Ưu tiên | Experiment | Đòn bẩy (1 biến) | Kỳ vọng |
| --- | --- | --- | --- |
| ✅ | `04_add_commonvoice` | gộp Common Voice VI vào train | **DONE — THẮNG** (run `vivos-cv`): CV 42,95→36,85% (−6,1%), VIVOS 11,93→11,39%. Lợi khiêm tốn (CV ít data). Xem `04/RESULT.md` |
| 1 | `02_best_ckpt_val` | chọn best-checkpoint theo val thay vì epoch cuối | giảm nhẹ WER, rẻ |
| 2 | `03_specaugment` | bật SpecAugment | chống overfit ~15h data |
| 3 | `05_ctc_vs_rnnt` | so decoder CTC (như ChunkFormer) vs RNNT | hiểu trade-off |
