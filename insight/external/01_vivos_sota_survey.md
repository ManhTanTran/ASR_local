# external — khảo sát SOTA VIVOS (sensing)

> Cập nhật **2026-06-23** (web search). Đây là **mốc ngoài** để xếp run của mình vào thang, KHÔNG
> phải kết quả của mình. Số báo theo chuẩn hoá riêng của từng nhóm → so **xấp xỉ theo nấc**, không
> phải số lẻ tuyệt đối.

## Bảng WER trên VIVOS test

| Model | Tham số | WER | Kiến trúc / data |
| --- | --- | --- | --- |
| **ChunkFormer-CTC-large-vie** | 110M | **4,18%** | ChunkFormer + CTC, ~3.000h VI đa nguồn, có streaming |
| PhoWhisper-large | 1,55B | 4,67% | Whisper fine-tune, 844h VI |
| PhoWhisper-medium | — | 4,97% | Whisper fine-tune |
| wav2vec2-large-vi-vlsp2020 | ~300M | 8,61% | wav2vec2 (baseline mạnh trước PhoWhisper) |
| PhoWhisper-base | — | 8,46% | Whisper fine-tune |
| wav2vec2-base-vi-vlsp2020 | — | 9,90% | wav2vec2 |
| PhoWhisper-tiny | — | 10,41% | Whisper fine-tune |
| wav2vec2-base-vietnamese-250h | ~95M | 10,83% | wav2vec2 (baseline phổ biến nhất) |

ChunkFormer (ngoài VIVOS): Common Voice VI 6,66% · VLSP Task-1 14,09% · TB 8,31%.

## Kết luận sensing

1. **SOTA ~4,2-4,7%** trên VIVOS; mốc < 4% gần như chưa có model công khai. Vùng 4-5% là **trần thực tế**.
2. **Cỡ model KHÔNG phải nút thắt**: ChunkFormer 110M < PhoWhisper-large 1,55B mà tốt hơn.
3. **Nút thắt = DATA**: ChunkFormer 3.000h, PhoWhisper 844h; mình ~15h VIVOS. → đòn bẩy lớn nhất là gộp data.
4. ChunkFormer (110M, CTC) **cùng hạng cân** model mình (fastconformer 115M, RNNT) → đáng nghiên cứu CTC.

→ Định hướng rút ra: `proposals/01_next_directions.md`.

## Nguồn

- PhoWhisper (arXiv 2406.02555) — https://arxiv.org/pdf/2406.02555
- ChunkFormer-CTC-large-vie (HuggingFace) — https://huggingface.co/khanhld/chunkformer-ctc-large-vie
- wav2vec2-base-vietnamese-250h (Dataloop) — https://dataloop.ai/library/model/nguyenvulebinh_wav2vec2-base-vietnamese-250h/
