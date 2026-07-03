# RESULT - 00_ctc_epoch22_trial

**Notebook:** `notebooks/final/fastconformer/03_fastconformer_ctc_epoch22_trial.ipynb`

**Trang thai:** local trial, chua promote sang `ASR`.

## So chinh

| split | samples | WER | CER | ghi chu |
| --- | ---: | ---: | ---: | --- |
| val | 583 | **43.75%** | 20.16% | eval lai best `.nemo` |
| test | 760 | **47.84%** | 22.47% | metric chinh de doc run |

Chi tiet tu output notebook:

```text
best callback val_wer = 0.438433
eval val WER = 0.4375398749521501
eval val CER = 0.20160976631459263
eval test WER = 0.4783734783734784
eval test CER = 0.22471087241768636
empty_pred_rate = 0.0
pred_ref_word_ratio = 1.0001295001295
elapsed_minutes = 181.64332000925
```

## Duong val WER trong doan continue

| epoch trong continue | step | val WER |
| ---: | ---: | ---: |
| 2/12 | 628 | 47.95% |
| 3/12 | 942 | 47.15% |
| 4/12 | 1256 | 46.62% |
| 5/12 | 1570 | 46.00% |
| 8/12 | 2512 | 44.38% |
| 9/12 | 2826 | 44.30% |
| 10/12 | 3140 | 44.14% |
| 11/12 | 3454 | 44.07% |
| 12/12 | 3768 | **43.84%** |

## Verdict

- PASS ve thong luong: notebook chay duoc, model restore duoc, train/eval duoc, output khong rong.
- FAIL ve muc ket qua: WER test **47.84%** con cao, chua nen update sang `ASR` main.
- Metric dang tin la WER/CER tu `evaluate_nemo`.

## Caveat

Cell error-analysis sau eval dang bi bug: `hyp` bi normalize tu object `Hypothesis` thanh chuoi
`hypothesis score tensor...`, lam cac bang WER theo bucket, substitution, insertion bi nhieu. Can sua:

```python
hyp = getattr(hyp, "text", hyp)
```

truoc khi tinh `wer(ref_n, hyp_n)` trong cell error-analysis.
