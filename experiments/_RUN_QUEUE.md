# Local run queue

## Đã chốt

| run_id | việc | trạng thái |
| --- | --- | --- |
| `vivos-fc-ctc-v2norm` | Kaggle GitHub notebook, full fine-tune FastConformer CTC VIVOS | DONE, số chốt từ report DOCX |
| `ctc-epoch22-trial` | gắn notebook CTC đã chạy, đọc kết quả epoch ~22 | DONE, local trial |

## Kế tiếp

| ưu tiên | việc | lý do |
| --- | --- | --- |
| 1 | Kéo/lưu đầy đủ artifact của `vivos-fc-ctc-v2norm`: `.nemo`, checkpoint epoch 49, `results.json`, `run.log`, `error_analysis.csv` | Report đã có số, nhưng repo local nên giữ lại nguồn máy đọc được để tái lập |
| 2 | So sánh `.nemo` cuối với best checkpoint theo validation WER | Report ghi best val WER và val cuối cùng đều 13.10%, nhưng vẫn nên xác nhận checkpoint được export đúng |
| 3 | Thử beam search/LM hoặc rescoring cho CTC | Substitution chiếm 88.3% lỗi word; decoder/LM có thể giảm nhầm âm/từ gần giống |
| 4 | Nghe lại nhóm câu WER cao: idx 825, 547, 813 | Phân biệt lỗi audio/transcript với lỗi mô hình |
| 5 | Bổ sung profiler: params, trainable params, FLOPs, latency, peak VRAM | Report hiện chưa đo các số này |
| 6 | Ghi rõ split VIVOS: số câu, giờ audio, sampling rate, speaker cho train/val/test | Report hiện mới ghi dataset tổng quát, chưa có metadata split đầy đủ |
