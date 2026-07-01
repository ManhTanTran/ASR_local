# Nemotron Speech Streaming — INDEX

Họ model ASR streaming của NVIDIA: cache-aware FastConformer encoder + decoder RNNT, độ trễ điều chỉnh được. Bản phân tích: `nemotron-speech-streaming-en-0.6b`.

| File | Nội dung |
| --- | --- |
| [01_structure.md](01_structure.md) | Cấu trúc kiến trúc (số đo thật) + cơ chế cache-aware |
| [02_characteristics_eval_advances.md](02_characteristics_eval_advances.md) | Streaming, điều chỉnh độ trễ, đa ngôn ngữ, đánh giá, tiến bộ so với Fast-Conformer |
| [03_pretrained_weights.md](03_pretrained_weights.md) | Các checkpoint tải được, license, lệnh tải/smoke-test (bản 3.5 CÓ tiếng Việt) |
| [04_vi_finetune_difficulty.md](04_vi_finetune_difficulty.md) | **Vì sao KHÓ fine-tune sang tiếng Việt** — streaming + hybrid RNNT/CTC → collapse-to-blank (có config thật + bằng chứng) |

Kiến thức nền: encoder Conformer → `../../02_asr_components/05_encoder_conformer.md`; RNNT + streaming cache-aware → `../../02_asr_components/07_decode_rnnt.md`.
