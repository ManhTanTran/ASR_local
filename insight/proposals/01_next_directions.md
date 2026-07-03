# proposal - bước tiếp sau CTC 14.13%

Run neo: `../../experiments/01_fastconformer_ctc_v2norm_kaggle/RESULT.md`.

## Luận điểm

Run CTC đã qua giai đoạn "pipeline có chạy không": WER giảm **100.43% -> 14.13%** và CER còn **7.67%**. Việc tiếp theo nên tập trung vào giảm **substitution**, vì substitution chiếm **88.3%** tổng lỗi word.

## Thứ tự ưu tiên

| ưu tiên | việc | vì sao |
| ---: | --- | --- |
| 1 | Beam search / LM / rescoring cho CTC | Rẻ hơn train lại, đánh trực tiếp vào nhầm từ gần âm |
| 2 | Nghe lại nhóm WER cao | Xác định lỗi do audio, transcript hay model trước khi sửa theo cảm tính |
| 3 | Confusion report theo âm/từ | Biến top substitution thành danh sách âm/từ cần augment |
| 4 | Best checkpoint audit | Report ghi best val và val cuối đều 13.10%, nhưng cần xác nhận `.nemo` export đúng checkpoint |
| 5 | Profiler params/FLOPs/latency/VRAM | Report hiện thiếu số chi phí mô hình |
| 6 | Gộp thêm data hoặc SpecAugment có mục tiêu | Là đòn bẩy lớn hơn nhưng tốn hơn; làm sau khi decoder baseline rõ |

## Gate cho lần chạy tiếp

- Phải giữ WER test dưới **14.13%** trên cùng harness.
- Nên báo thêm CER, S/D/I và RTF, không chỉ WER.
- Nếu thử decoder/LM, cần so sánh phân rã lỗi trước-sau để xem substitution có giảm thật không.
- Nếu train lại, phải lưu `results.json`, `run.log`, `.nemo`/checkpoint và predictions/error analysis.
