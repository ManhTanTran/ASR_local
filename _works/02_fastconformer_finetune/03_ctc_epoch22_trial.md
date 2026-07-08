# CTC epoch 22 trial

Nguon notebook: `D:\Downloads\asr-training.ipynb`.

Da gan vao local tai:

```text
notebooks/final/fastconformer/03_fastconformer_ctc_epoch22_trial.ipynb
```

## Cach doc moc epoch

Notebook khong in truc tiep `epoch=22`. Run co ten `phase3_continue_from_epoch11` va doan continue chay
`12/12` epoch. Neu tinh theo moc dang noi "epoch 22" = tiep tuc tu checkpoint epoch 11, ket qua can lay la
checkpoint best/cuoi cua doan continue nay.

## Cau hinh chinh

- Model: `nvidia/stt_en_fastconformer_ctc_large`.
- Dataset: VIVOS.
- Tokenizer: SentencePiece unigram, vocab 1024, `character_coverage=1.0`.
- Resume model: `/kaggle/input/models/trnmnhtn/finetune-fastconformer/other/default/1/phase3_best_full_finetune.nemo`.
- Phase: full encoder + decoder fine-tune.
- LR: `2e-5`.
- Batch: `2`, accumulate grad `16`.
- Trainable params: `115,600,385 / 115,600,385`.
- Runtime: Tesla T4, `16-mixed`, elapsed `181.64` phut.

## Ket qua doc tu notebook

Validation:

- Best val WER trong callback: `0.438433` = **43.84%**.
- Eval lai best `.nemo` tren val: WER `0.437540` = **43.75%**, CER `20.16%`.
- Val samples: `583`.

Test:

- Test WER: `0.478373` = **47.84%**.
- Test CER: `0.224711` = **22.47%**.
- Test samples: `760`.
- Empty prediction rate: `0.0`.
- Pred/ref word ratio: `1.00013`.

## Nhan xet

Run nay co hoc tiep va khong bi output rong, nhung WER con cao. Ket qua hien tai chua phai moc tot de
update sang `ASR`; no nen duoc xem la ban local dang chay thu/kiem nghiem notebook CTC.

Luu y: cac cell error-analysis sau eval dang bi nhieu vi `hyp` bi stringify thanh object `Hypothesis score tensor...`
thay vi text transcript. Vi vay chi nen tin cac metric tu `evaluate_nemo`, chua nen tin bang substitution/insertion
ben duoi cho den khi sua `hyp = getattr(hyp, "text", hyp)`.
