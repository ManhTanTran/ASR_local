# Experiment Idea - FastConformer Fine-tune

## Mục tiêu

Fine-tune model NVIDIA FastConformer sang tiếng Việt với VIVOS, sau đó thử resume + thêm Common Voice VI.

## Hướng chính

1. Dùng main repo pipeline:
   - `asr_lab.train.finetune_vivos`
   - `asr_lab.train.continue_vi`
   - `asr_lab.deploy.kaggle`
2. Notebook chỉ điều phối command và đọc kết quả.
3. Artifacts chính nằm ở `ASR/artifacts/runs/<run_id>/`.

## Run quan trọng

- `vivos-fc115m-v2norm`: VIVOS full, FastConformer 115M, normalize/OOV gate.
- `vivos-cv`: resume từ VIVOS checkpoint, gộp Common Voice VI.

## Cần theo dõi

- WER trước/sau.
- RTF sau.
- OOV/tokenizer mismatch.
- Domain shift VIVOS -> Common Voice.

