# 00 — INDEX: phân tích model cụ thể

Mỗi model một thư mục, gồm: cấu trúc (structure) và đặc tính + đánh giá + điểm tiến bộ so với Fast-Conformer.
Kiến thức nền dùng chung ở `../02_asr_components/`. Số liệu kiến trúc đo thật bằng `notebooks/01_explore_model_config.ipynb`.

| Thư mục | Model | Decoder | Hướng dùng |
| --- | --- | --- | --- |
| [01_parakeet/](01_parakeet/00_INDEX.md) | Parakeet-TDT-0.6b-v2 | **TDT** | Offline, rất nhanh |
| [02_nemotron/](02_nemotron/00_INDEX.md) | Nemotron-Speech-Streaming-0.6b | **RNNT cache-aware** | Streaming độ trễ điều chỉnh |
| [03_fastconformer/](03_fastconformer/00_INDEX.md) | FastConformer-Transducer-Large 115M | **RNNT offline** | Cỡ VPB; **fine-tune VI thành công** |

## Mốc so sánh: Fast-Conformer RNNT (model VPB cũ)

Dùng làm gốc so sánh xuyên suốt — chi tiết ở `stt_nvidia_nemo/docs_vpb/02_review_nemo/01_asr_domain_review.md`.

| | Fast-Conformer (VPB) | Parakeet-TDT | Nemotron-Streaming |
| --- | --- | --- | --- |
| Params | ~120M (large) | 617,8M | 618,1M |
| Encoder | FastConformer large | FastConformer XL | Cache-aware FastConformer |
| d_model / lớp | 512 / 17 | 1024 / 24 | 1024 / 24 |
| Decoder | RNNT | TDT | RNNT (cache-aware) |
| joint_out | 1025 | 1030 (+5 duration) | 1025 |
