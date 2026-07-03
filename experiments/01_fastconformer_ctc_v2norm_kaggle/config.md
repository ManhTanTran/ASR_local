# config - 01_fastconformer_ctc_v2norm_kaggle

## Notebook

```text
notebooks/final/fastconformer/02_fastconformer_main.ipynb
```

Kaggle notebook:

```text
https://www.kaggle.com/code/tantranmanh/asr-fine-tuning/edit
```

GitHub repo duoc Kaggle clone:

```text
https://github.com/ManhTanTran/ASR_local.git
```

## Model va data

| key            | value                                     |
| -------------- | ----------------------------------------- |
| run_id         | `vivos-fc-ctc-v2norm`                   |
| pretrained     | `nvidia/stt_en_fastconformer_ctc_large` |
| model family   | FastConformer                             |
| decoder        | CTC                                       |
| dataset        | VIVOS                                     |
| train split    | VIVOS train,`train_n=0` lay het         |
| val split      | holdout tu train,`val_n=300`            |
| test split     | VIVOS test,`limit_test=0` lay het       |
| tokenizer      | SentencePiece/BPE tieng Viet              |
| vocab_size     | 1024                                      |
| text normalize | `normalize_vi`                          |

## Completion observed in report

| key | value |
| --- | --- |
| completed_epochs | 50 |
| latest_epoch_logged | 49 |
| latest_global_step | 34,751 |
| median_steps_per_epoch | khoảng 695 |
| train_seconds | 18,375.5 giây, khoảng 5.10 giờ |
| cuda | true |
| observed_gpu | Tesla T4 x2, mỗi GPU khoảng 15,360 MB VRAM |
| cpu_logical_physical | 4 / 2 |
| ram_total_available | 31.348 GB / 30.073 GB |

## Train config

| key                     | value                                    |
| ----------------------- | ---------------------------------------- |
| finetune_mode           | full encoder + decoder/head CTC          |
| freeze_encoder          | false                                    |
| epochs                  | 50                                       |
| batch_size              | 16                                       |
| learning_rate           | `2e-4`                                 |
| precision               | `32`                                   |
| max_minutes             | 660                                      |
| optimizer               | AdamW                                    |
| scheduler               | CosineAnnealing                          |
| warmup_steps            | `max(50, min(500, total_steps // 10))` |
| weight_decay            | 0.001                                    |
| max_duration            | 20.0 sec                                 |
| min_duration            | 0.1 sec                                  |
| num_workers             | 2                                        |
| devices                 | 1 GPU                                    |
| Kaggle accelerator seen | T4 x2, script uses 1 GPU                 |

## Logging va checkpoint

| key                 | value                                                |
| ------------------- | ---------------------------------------------------- |
| console_log_steps   | 0                                                    |
| visible logs        | epoch summary only                                   |
| epoch metrics       | `train_loss`, `val_loss`, `val_wer`, `lr`    |
| run log             | `/kaggle/working/runs/vivos-fc-ctc-v2norm/run.log` |
| checkpoint_steps    | 0                                                    |
| checkpoint mode     | epoch-end only                                       |
| checkpoint_keep     | 2                                                    |
| checkpoint manifest | `checkpoints/checkpoint_manifest.json`             |
| auto_resume         | true                                                 |

## Artifacts observed in report

| artifact | value |
| --- | --- |
| final_checkpoint | `/kaggle/working/runs/vivos-fc-ctc-v2norm/checkpoints/epoch-end-epoch049-step034751.ckpt` |
| exported_nemo | `/kaggle/working/runs/vivos-fc-ctc-v2norm/report/../fastconformer_vivos_ft.nemo` |
| nemo_size | 463.13 MB |
| latest_checkpoint_size | 1,388.41 MB |
| metrics_csv | `logs/version_0/metrics.csv`, 0.09 MB |
| run_log | `run.log`, 0.34 MB |

## Expected scale

Doc tu log truoc do:

```text
steps_per_epoch ~= 695
total_steps ~= 34,750
```

De chay du 50 epoch trong 660 phut can toc do toi thieu xap xi:

```text
34,750 steps / 39,600 sec = 0.88 step/s
```

Neu step time cham hon khoang 1.1 sec/step thi co the khong kip full 50 epoch trong mot session.
