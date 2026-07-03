# spec - 00_ctc_epoch22_trial

## Muc tieu

Gan notebook `D:\Downloads\asr-training.ipynb` vao `ASR_local` nhu mot ban dang chay thu, roi chot so
ket qua o moc user goi la epoch 22.

## Cach doc moc epoch

Notebook khong co dong literal `epoch=22`. Cell chinh tao run:

```text
phase3_continue_from_epoch11
max_epochs = 12
```

Vi vay "epoch 22" duoc doc la checkpoint best/cuoi cua doan continue tu checkpoint epoch 11 sau khi chay
them 12 epoch.

## Gia thuyet

- H1: CTC FastConformer tiep tuc hoc duoc tren VIVOS, khong roi vao output rong.
- H0: Run khong on dinh, output rong hoac metric khong dang tin.

## Tieu chi doc ket qua

- Lay WER/CER tu ham `evaluate_nemo`.
- Khong lay bang error-analysis neu `hyp` bi stringify thanh `Hypothesis score tensor...`.
- Neu WER con cao, giu la local trial, chua update sang `ASR`.
