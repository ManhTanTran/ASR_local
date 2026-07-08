# proposal — hướng phát triển lab ASR (nấc thang WER)

> Cập nhật 2026-06-23. Kết luận sensing hôm nay (xem `external/01_vivos_sota_survey.md`):
> **nút thắt là DATA, không phải model/size.**

## Luận điểm trung tâm: data > model trên VIVOS

- ChunkFormer-CTC **110M** đạt **4,18%** — thấp hơn PhoWhisper-large **1,55B** (4,67%): nhỏ hơn ~14
  lần mà tốt hơn. ⇒ trên VIVOS, **cỡ model không phải nút thắt**.
- ChunkFormer (110M) cùng hạng cân model mình đang chạy (fastconformer 115M). Khác biệt chính:
  **data 3.000h đa nguồn** vs **~15h VIVOS thuần** của mình.
- ⇒ Với ~15h, tune đẹp đến đâu cũng khó chạm 4-5%. Lộ trình phải đi theo **data**.

## Nấc thang mục tiêu

| Nấc | WER đích | Đòn bẩy chính | Experiment |
| --- | --- | --- | --- |
| Hiện tại | 20,37% | (fine-tune đầu, có bug `<unk>`) | `00` |
| **Nấc 1** | **< 11%** (bậc cộng đồng) | fix chuẩn hoá + đủ epoch + best-ckpt val + SpecAugment | `01`(đang chạy), `02`, `03` |
| Nấc 2 | 6-9% | **gộp thêm data** (Common Voice VI, VLSP, FOSD, VietBud500) | `04+` |
| Nấc 3 | 4-5% (SOTA) | data nhiều nghìn giờ + có thể thêm LM/rescoring | dài hạn |

## Việc nên làm tiếp (rẻ → đắt)

1. **Chốt nấc 1** từ run `01`: nếu < 11% → ăn mừng nhẹ, sang nấc 2; nếu kẹt → `02`(best-ckpt) + `03`(SpecAug).
2. **Sensing data tiếng Việt** (giờ/domain/license) → bảng trong `external/`, chọn nguồn gộp cho `04`.
3. **So decoder CTC vs RNNT** (`05`) — ChunkFormer dùng CTC + streaming; hiểu trade-off cho callbot.
4. Dài hạn: dữ liệu giọng điện thoại 8kHz (đích thực tế của IruKa callbot, khác hẳn VIVOS phòng thu).
