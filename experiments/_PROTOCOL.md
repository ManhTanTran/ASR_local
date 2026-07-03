# Local experiment protocol

## Nguyen tac

- `ASR_local` la noi thu nghiem/notebook ca nhan.
- `ASR` main chi nhan ket qua sau khi run local co so ro, co `RESULT.md`, va du tot de giu lai.
- Khong copy scoreboard/result tu `ASR` main ve `ASR_local`, tru khi do la khung trong va khong kem so lieu.

## Moi run can co

1. `spec.md`: y do, nguon notebook, cach doc moc epoch/checkpoint.
2. `config.md`: model, data, tokenizer, hyperparameter, runtime.
3. `RESULT.md`: so that doc tu log/notebook/artifact, verdict, caveat.
4. Neu co error analysis: ghi so cau, WER/CER corpus, S/D/I, top nham lan, nhom cau xau nhat, va link sang `insight/error_analysis/`.
5. Cap nhat `_SCOREBOARD.md` chi voi so cua local.

## Cong promote sang ASR main

- Metric phai lay tu eval function/artifact dang tin, khong lay tu cell phan tich loi neu cell do dang bug.
- Can co WER/CER tren test split va mo ta split.
- Nen co artifact may doc duoc: `results.json`, `run.log`, `.nemo` hoac checkpoint, va neu co thi `error_analysis.csv`.
- Neu WER con cao hoac run chi la sanity check, giu o `ASR_local`.
- Neu promote, copy sang `ASR` bang mot experiment folder moi hoac update folder dang co, sau do moi chuan bi push GitHub.
