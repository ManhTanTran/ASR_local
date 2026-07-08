# config — 01_vivos_fc115m_norm

- **run_id (artifact):** `vivos-fc115m-v2norm`
- **account Kaggle:** kyhoolee (GPU)
- **model nền:** `nvidia/stt_en_fastconformer_transducer_large`
- **code:** đã build code-dataset gồm fix `normalize_vi` nhãn train + `assert_no_oov` (cổng OOV)

| Tham số | Giá trị |
| --- | --- |
| epochs | 50 |
| batch | 16 |
| vocab_size | 1024 |
| lr | 2e-4 |
| precision | 32 (ổn định RNNT) |
| max_minutes | 480 |
| train_n / test | full (11420 / 1000) |

```bash
uv run python -m asr_lab.deploy.kaggle build --account kyhoolee
uv run python -m asr_lab.deploy.kaggle push --account kyhoolee --gpu --as asr-ft-fc115m-v2norm \
  --module asr_lab.train.finetune_vivos --script-args \
  "--pretrained nvidia/stt_en_fastconformer_transducer_large --run-id vivos-fc115m-v2norm \
   --epochs 50 --batch 16 --vocab-size 1024 --lr 2e-4 --precision 32 --max-minutes 480"
```
