# config — 00_vivos_fc115m

- **run_id (artifact):** `vivos-fc115m-v1`
- **account Kaggle:** kyhoolee (P100 GPU)
- **model nền:** `nvidia/stt_en_fastconformer_transducer_large`

| Tham số | Giá trị |
| --- | --- |
| epochs | 40 (chạy đủ 40) |
| batch | 16 |
| vocab_size | 1024 |
| lr | 2e-4 |
| freeze_encoder | False (full fine-tune) |
| max_minutes | 420 |
| train_n / test | full (11420 / 1000) |

Lệnh (dạng cũ, trước refactor — nay chạy qua `-m asr_lab.deploy.kaggle`):

```bash
uv run python -m asr_lab.deploy.kaggle push --account kyhoolee --gpu --as asr-ft-fc115m-full \
  --module asr_lab.train.finetune_vivos --script-args \
  "--pretrained nvidia/stt_en_fastconformer_transducer_large --run-id vivos-fc115m-v1 \
   --epochs 40 --batch 16 --vocab-size 1024 --lr 2e-4 --max-minutes 420"
```
