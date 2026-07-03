# spec - 01_fastconformer_ctc_v2norm_kaggle

**Trạng thái:** COMPLETE, đã chốt số từ `D:\Downloads\final_report_updated_error_analysis.docx`.

## Muc tieu

Chay notebook Kaggle GitHub sach cho FastConformer CTC tren VIVOS, tach logic vao `src/asr_lab`
thay vi notebook to, va tao artifact co the resume/pull ve local.

## Cau hoi

- H1: Full fine-tune FastConformer CTC voi tokenizer tieng Viet giam WER ro ret so voi baseline English pretrained.
- H2: Checkpoint epoch-end du de resume neu Kaggle het GPU/truncate session.
- H3: Log gon theo epoch van du thong tin de theo doi `train_loss`, `val_loss`, `val_wer`.

## Cach doc run

Run chinh:

```text
vivos-fc-ctc-v2norm
```

Thu muc Kaggle:

```text
/kaggle/working/runs/vivos-fc-ctc-v2norm/
```

File can doc khi co output:

```text
run.log
results.json
status.json
checkpoints/checkpoint_manifest.json
checkpoints/*.ckpt
fastconformer_vivos_ft.nemo
```

## Tieu chi cap nhat RESULT

Chi cap nhat so khi lay duoc tu `results.json` hoac dong eval cuoi trong `run.log`.

Can ghi:

- `wer_before`
- `wer_after`
- `rtf_before`
- `rtf_after`
- `completed_epochs`
- `train_minutes`
- `latest_checkpoint`
- co tao duoc `.nemo` hay chi co checkpoint

Report DOCX đã bổ sung thêm error analysis 1000 câu test, nên `RESULT.md` cần ghi thêm:

- corpus CER
- tổng lỗi word S/D/I
- top substitution/deletion/insertion
- nhóm câu WER cao cần nghe lại
- caveat về params/FLOPs và metadata split chưa đo đủ

## Resume

Truoc khi tat may/local browser, can Save Version + Output tren Kaggle. Lan sau add output version cu lam input,
script se tim checkpoint moi nhat trong:

```text
/kaggle/input/**/runs/vivos-fc-ctc-v2norm/checkpoints/*.ckpt
```
