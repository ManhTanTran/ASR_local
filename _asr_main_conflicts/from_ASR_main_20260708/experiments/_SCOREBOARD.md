# 📊 _SCOREBOARD — mọi run fine-tune (sinh tự động)

> Sinh bằng `uv run python -m asr_lab.registry.build_scoreboard` từ `artifacts/runs/*/results.json`. **Đừng sửa tay** — chạy lại script.

## Run của mình (sắp theo WER sau, thấp = tốt)

| run_id | model nền | WER trước | **WER sau** | RTF sau | epoch | batch | vocab | step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vivos-cv` | resume:nemotron_vivos_ft.nemo | 11,93% | **11,39%** | 0.0573 | 25 | 16 | 1024 | 20625 |
| `vivos-fc115m-v2norm` | stt_en_fastconformer_transducer_large | 100,13% | **11,93%** | 0.0675 | 50 | 16 | 1024 | 34750 |
| `vivos-fc115m-v1` | stt_en_fastconformer_transducer_large | 100,13% | **20,37%** | 0.0615 | 40 | 16 | 1024 | 27800 |
| `verify2` | stt_en_fastconformer_transducer_large | 100,49% | **81,03%** | 0.0556 | 11 | 16 | 1024 | 3048 |
| `refactorchk` | stt_en_fastconformer_transducer_large | 100,61% | **100,00%** | 0.0562 | 2 | 8 | 512 | 186 |
| `verify` | nemotron-speech-streaming-en-0.6b | 100,13% | **100,00%** | 0.2025 | 2 | 8 | 1024 | 346 |

## Mốc ngoài trên VIVOS (sensing — KHÔNG phải run của mình)

| model | tham số | WER | ghi chú |
| --- | --- | --- | --- |
| ChunkFormer-CTC-large-vie | 110M | 4,18% | SOTA cỡ nhỏ (3.000h) |
| PhoWhisper-large | 1,55B | 4,67% | Whisper 844h VI |
| wav2vec2-large-vi-vlsp2020 | ~300M | 8,61% | baseline mạnh |
| wav2vec2-base-vietnamese-250h | ~95M | 10,83% | baseline phổ biến |

## Thang baseline VIVOS (đọc verdict theo bậc)

1. **Sàn** English zero-shot ~100% · 2. **Tối thiểu** fine-tune đầu 20,37% · 3. **KPI cộng đồng** wav2vec2-base ~10,8% · 4. **Sao bắc đẩu** ChunkFormer 4,18%.
