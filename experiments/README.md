# experiments/ - local run ledger

Thư mục này ghi lại các run trong `ASR_local`: ý đồ, config, số thật và verdict. `ASR_local` là nơi thử nghiệm trước; chỉ khi số liệu đủ rõ và có lý do giữ lại thì mới promote sang `ASR` main.

## Index

| run | trạng thái | kết quả chính | file |
| --- | --- | --- | --- |
| `00_ctc_epoch22_trial` | local trial | test WER 47.84%, CER 22.47% | `00_ctc_epoch22_trial/RESULT.md` |
| `01_fastconformer_ctc_v2norm_kaggle` | complete | test WER 14.13%, CER 7.67%, best val WER 13.10% | `01_fastconformer_ctc_v2norm_kaggle/RESULT.md` |

## Quy ước đọc

- `_SCOREBOARD.md` là bảng số gọn để nhìn nhanh.
- `_RUN_QUEUE.md` là hàng đợi việc tiếp theo sau mỗi run.
- `_PROTOCOL.md` là quy tắc ghi run, gate promote và các caveat cần nêu rõ.
- Mỗi run nên có `spec.md`, `config.md`, `RESULT.md`; nếu có phân tích lỗi sâu thì link sang `insight/error_analysis/`.
