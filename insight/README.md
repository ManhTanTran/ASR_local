# insight/ - nhật ký nghiên cứu local

`insight/` giữ phần tri thức động của `ASR_local`: quyết định, error analysis và hướng chạy tiếp. Khác `experiments/`, thư mục này không phải ledger số liệu thô; mỗi số quan trọng phải link ngược về run tương ứng.

## Index

| thư mục | nội dung |
| --- | --- |
| `error_analysis/` | Nhận xét lỗi định tính/định lượng sau khi có prediction hoặc report |
| `proposals/` | Hướng chạy tiếp, giả thuyết cần kiểm chứng |
| `decisions/` | Quyết định ngắn: chốt/không chốt, bằng chứng, đánh đổi |

## Run đang neo insight

| run_id | insight chính | experiment |
| --- | --- | --- |
| `vivos-fc-ctc-v2norm` | substitution là lỗi chính; thử beam/LM, nghe lại mẫu WER cao, bổ sung profiler | `../experiments/01_fastconformer_ctc_v2norm_kaggle/RESULT.md` |
