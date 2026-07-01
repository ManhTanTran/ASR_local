# FastConformer-Transducer Large (115M) — INDEX

Model `nvidia/stt_en_fastconformer_transducer_large`: FastConformer **offline** (att_context `regular`,
full context) + RNNT thuần, ~115M — **đúng cỡ kiến trúc VPB**. Đây là model fine-tune sang tiếng Việt
THÀNH CÔNG (VIVOS WER 100% → 20,37%), khác với nemotron streaming.

| File | Nội dung |
| --- | --- |
| [01_vi_finetune_error_analysis.md](01_vi_finetune_error_analysis.md) | **Phân tích lỗi output sau fine-tune** — root-cause "rớt ký tự đầu" = tokenizer/manifest mismatch (`<unk>` ở chữ hoa) + hướng cải thiện (đã sửa code) |

Kết quả + cơ chế train: [../../06_benchmarks/02_vivos_finetune.md](../../06_benchmarks/02_vivos_finetune.md).
So sánh khó fine-tune với nemotron streaming: [../02_nemotron/04_vi_finetune_difficulty.md](../02_nemotron/04_vi_finetune_difficulty.md).
