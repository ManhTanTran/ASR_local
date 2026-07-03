# error analysis - `vivos-fc-ctc-v2norm`

Nguồn: `D:\Downloads\final_report_updated_error_analysis.docx`.

Experiment: `../../experiments/01_fastconformer_ctc_v2norm_kaggle/RESULT.md`.

## Tóm tắt

FastConformer CTC fine-tune trên VIVOS đã giảm WER từ **100.43%** xuống **14.13%**; corpus CER sau fine-tune là **7.67%**. Đây là mức cải thiện rõ, đủ để giữ làm baseline CTC local.

Điểm đáng chú ý nhất của error analysis: lỗi còn lại chủ yếu là **substitution**, không phải deletion/insertion hay output rỗng.

| loại lỗi | số lượng | tỉ lệ trong tổng lỗi word |
| --- | ---: | ---: |
| Substitution | 1,579 | 88.3% |
| Deletion | 94 | 5.3% |
| Insertion | 115 | 6.4% |
| Tổng | 1,788 | 100.0% |

## Mẫu lỗi chính

| nhóm | ví dụ | nhận định |
| --- | --- | --- |
| Nhầm từ/âm gần giống | `việt -> việc`, `chị -> chỉ`, `đều -> điều`, `xin -> sinh`, `dạy -> dậy` | Cần cải thiện phân biệt âm gần nhau và ràng buộc ngôn ngữ |
| Mất từ ngắn/chức năng | `làm`, `những`, `sự`, `phải`, `học` | Có thể do tín hiệu yếu, phát âm nhanh hoặc decoder bỏ qua token ngắn |
| Chèn mảnh âm | `t`, `l`, `c`, `d`, `tr`, `ph` | Một số mẫu khó sinh token rời rạc chưa thành từ |

## Câu lỗi nặng cần nghe lại

| idx | WER | reference | prediction |
| ---: | ---: | --- | --- |
| 825 | 100.0% | `tự dưng trong lòng tôi nảy nở một hình ảnh rất đẹp` | `giá nói` |
| 547 | 100.0% | `khi tôi ăn cơm con đừng ăn mãi một món ưng ý nhé` | `khi tôi ăn cơm ăn mái món ưng nhé khi tôi ăn cơm con đường ăn mái một món ưng` |
| 813 | 100.0% | `tình tiền tù tội` | `tên t tên tu tội` |

Các câu này có thể lộ lỗi audio, transcript, tốc độ nói, vùng giọng hoặc giới hạn decoder. Không nên suy từ text-only; cần nghe lại audio.

## Insight

- Sau khi fine-tune, mô hình không còn vấn đề "không học được tiếng Việt"; WER 14.13% chứng minh pipeline CTC có hiệu lực.
- Nút thắt tiếp theo là chất lượng phân biệt âm/từ, đặc biệt các cặp gần âm hoặc khác dấu.
- Vì substitution áp đảo, hướng rẻ nhất nên thử trước là beam search, language model hoặc rescoring; nếu vẫn kẹt mới tăng data/augmentation có mục tiêu.
- CER 7.67% thấp hơn WER khá nhiều, gợi ý nhiều lỗi là sai một phần từ hoặc dấu/âm gần nhau, không phải câu hoàn toàn hỏng.

## Việc cần bổ sung

1. Nghe lại idx 825, 547, 813 và nhóm WER cao.
2. Lưu `error_analysis.csv` hoặc predictions đầy đủ vào artifact local.
3. Tách WER/CER theo độ dài câu, độ dài audio, speaker/vùng giọng nếu metadata có.
4. Tạo confusion report theo âm vị/từ vựng để biết nên augment nhóm âm nào.
5. Đo lại với beam search/LM và so sánh S/D/I trước-sau.
