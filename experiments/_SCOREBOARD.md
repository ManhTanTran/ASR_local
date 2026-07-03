# Local scoreboard

> Bảng này chỉ ghi kết quả chạy trong `ASR_local`. Không copy số từ `ASR` main vào đây.
> Chỉ khi một run local đủ tốt, có nguồn số rõ, thì mới promote/update ngược sang `ASR`.

## Runs

| run_id | notebook | model | decoder | mốc epoch | val WER | test WER | test CER | RTF sau | status |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `vivos-fc-ctc-v2norm` | `notebooks/final/fastconformer/02_fastconformer_main.ipynb` | `nvidia/stt_en_fastconformer_ctc_large` | CTC | 50/50, step 34,751 | **13.10%** | **14.13%** | **7.67%** | 0.054 | complete, CTC baseline local |
| `ctc-epoch22-trial` | `notebooks/final/fastconformer/03_fastconformer_ctc_epoch22_trial.ipynb` | `nvidia/stt_en_fastconformer_ctc_large` | CTC | continue `12/12` từ ckpt epoch 11, đọc như epoch ~22 | 43.75% | **47.84%** | 22.47% | n/a | local trial, chưa promote |

## Đọc nhanh

- `vivos-fc-ctc-v2norm` là run Kaggle GitHub đã hoàn tất: FastConformer CTC full fine-tune trên VIVOS, batch 16, 50 epoch.
- WER giảm từ **100.43%** xuống **14.13%**; relative WER reduction **85.93%**.
- Error analysis 1000 câu test: **S=1579, D=94, I=115**; substitution chiếm khoảng **88.3%** lỗi word.
- Run này là baseline CTC local đáng giữ lại, nhưng chưa tự động promote sang `ASR` main cho tới khi đối chiếu thêm best-checkpoint/beam search và artifact đầy đủ.
- `ctc-epoch22-trial` giữ vai trò sanity check cũ; số lỗi chi tiết của trial đó không đáng tin vì cell error-analysis từng stringify `Hypothesis`.
