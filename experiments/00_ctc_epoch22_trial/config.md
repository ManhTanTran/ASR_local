# config - 00_ctc_epoch22_trial

## Notebook

```text
notebooks/final/fastconformer/03_fastconformer_ctc_epoch22_trial.ipynb
```

Nguon ban dau:

```text
D:\Downloads\asr-training.ipynb
```

## Model va data

| key | value |
| --- | --- |
| model | `nvidia/stt_en_fastconformer_ctc_large` |
| decoder | CTC |
| dataset | VIVOS |
| tokenizer | SentencePiece unigram |
| vocab_size | 1024 |
| character_coverage | 1.0 |
| resume_from | `/kaggle/input/models/trnmnhtn/finetune-fastconformer/other/default/1/phase3_best_full_finetune.nemo` |

## Train config

| key | value |
| --- | --- |
| phase | `phase3_continue_from_epoch11` |
| trainable_mode | `full_encoder_decoder` |
| max_epochs | 12 |
| lr | `2e-5` |
| batch_size | 2 |
| accumulate_grad_batches | 16 |
| precision | `16-mixed` |
| warmup_steps | 500 |
| weight_decay | 0.001 |
| steps_per_epoch | 314 |
| estimated_total_steps | 3768 |
| runtime | Tesla T4 |

## Params

| key | value |
| --- | ---: |
| encoder_trainable_params | 115,074,560 |
| decoder_trainable_params | 525,825 |
| total_trainable_params | 115,600,385 |
| total_params | 115,600,385 |
| trainable_fraction | 1.0 |
