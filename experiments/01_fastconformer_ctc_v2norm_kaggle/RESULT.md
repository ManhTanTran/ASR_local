# RESULT - 01_fastconformer_ctc_v2norm_kaggle

**Notebook:** `notebooks/final/fastconformer/02_fastconformer_main.ipynb`

**Run ID:** `vivos-fc-ctc-v2norm`

**Trạng thái:** COMPLETE, chốt số từ `D:\Downloads\final_report_updated_error_analysis.docx`.

**Artifact chính:** `/kaggle/working/runs/vivos-fc-ctc-v2norm/`

## Số chính

| metric | trước fine-tune | sau fine-tune | ghi chú |
| --- | ---: | ---: | --- |
| WER test | 100.43% | **14.13%** | giảm 86.30 điểm phần trăm |
| Relative WER reduction | - | **85.93%** | cải thiện mạnh sau fine-tune |
| Best validation WER | - | **13.10%** | đạt tại epoch 49 |
| Validation WER cuối log | - | **13.10%** | gần như trùng best |
| RTF | 0.003 | **0.054** | vẫn nhanh hơn real-time |
| Corpus CER | - | **7.67%** | đo trên test set sau fine-tune |

Chi tiết run:

```text
completed_epochs = 50
latest_epoch_logged = 49
latest_global_step = 34751
train_seconds = 18375.5 (~5.10h)
checkpoint = /kaggle/working/runs/vivos-fc-ctc-v2norm/checkpoints/epoch-end-epoch049-step034751.ckpt
nemo = /kaggle/working/runs/vivos-fc-ctc-v2norm/report/../fastconformer_vivos_ft.nemo
```

## Config đã xác nhận

| key | value |
| --- | --- |
| pretrained | `nvidia/stt_en_fastconformer_ctc_large` |
| model type | `EncDecCTCModelBPE` |
| finetune | full encoder + decoder/head CTC |
| dataset | VIVOS |
| epochs | 50 |
| batch_size | 16 |
| learning_rate | `2e-4` |
| precision | `32` |
| vocab_size | 1024 |
| cuda | true |
| observed GPU | Tesla T4 x2 |

## Error analysis

Error analysis chạy trên **1000 câu** từ `test.jsonl`, transcribe bằng `fastconformer_vivos_ft.nemo`.

| metric | value |
| --- | ---: |
| Corpus WER | 14.13% |
| Corpus CER | 7.67% |
| Substitution | 1,579 |
| Deletion | 94 |
| Insertion | 115 |
| Tổng lỗi word | 1,788 |
| Tỉ lệ substitution | 88.3% |
| Thời gian transcribe test set | 31.1 giây |

Top lỗi cho thấy vấn đề chính là **nhầm âm/từ gần giống**, không phải output rỗng:

| nhóm | ví dụ |
| --- | --- |
| substitution | `việt -> việc`, `chị -> chỉ`, `đều -> điều`, `xin -> sinh`, `dạy -> dậy` |
| deletion | `làm`, `những`, `sự`, `phải`, `học` |
| insertion | `t`, `l`, `c`, `d`, `tr`, `ph` |

Nhóm câu WER cao cần nghe lại thủ công:

| idx | WER | nhận xét nhanh |
| ---: | ---: | --- |
| 825 | 100.0% | prediction rất ngắn so với reference |
| 547 | 100.0% | lặp/chen cụm dài |
| 813 | 100.0% | câu ngắn, sai vài token là WER tối đa |

Chi tiết insight nằm ở `../../insight/error_analysis/01_fastconformer_ctc_v2norm.md`.

## Verdict

- PASS về training: chạy đủ 50 epoch, có checkpoint epoch 49 và export được `.nemo`.
- PASS về metric: WER giảm mạnh từ **100.43%** xuống **14.13%**, CER còn **7.67%**.
- PASS về tốc độ inference: RTF sau fine-tune **0.054**, vẫn nhỏ hơn 1.
- CHƯA PASS để coi là tối ưu cuối: lỗi còn chủ yếu là substitution; cần thử decoder/LM, nghe lại mẫu xấu và bổ sung profile.

→ **VERDICT: THẮNG cho baseline CTC local, chưa tự động promote làm kết quả cuối.**

## Caveat

- Report hiện chưa đo total parameters, trainable parameters và FLOPs.
- Report chưa ghi đủ metadata split: số lượng utterance, tổng giờ audio, sampling rate, số speaker cho train/validation/test.
- Cần kéo artifact máy đọc được về local (`results.json`, `run.log`, `.nemo`, checkpoint, `error_analysis.csv`) để repo tự tái lập được số.

## Hướng kế

1. So sánh `.nemo` cuối với best checkpoint theo validation WER.
2. Thử beam search/language model hoặc rescoring để giảm substitution.
3. Tạo confusion report sâu hơn theo âm vị/từ vựng.
4. Nghe lại nhóm câu WER cao, nhất là idx 825, 547, 813.
5. Chạy profiler cố định input shape để đo params/FLOPs/latency/peak VRAM.
