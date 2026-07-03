# ASR Local

Workspace cá nhân cho notebook, ghi chú và code hỗ trợ fine-tune ASR.

## Mục tiêu

- Chạy/fine-tune ASR tiếng Việt trong môi trường local + Kaggle GPU.
- Tách logic khỏi notebook vào `src/asr_lab/` để notebook chỉ còn là entrypoint chạy thí nghiệm.
- Ghi lại số liệu, báo cáo và insight theo từng run để có thể promote ngược sang `ASR` main khi đủ chắc.

## Trạng thái mới nhất

Run `vivos-fc-ctc-v2norm` đã hoàn tất trên Kaggle và được chốt lại từ báo cáo:

```text
D:\Downloads\final_report_updated_error_analysis.docx
```

| metric | value |
| --- | ---: |
| WER trước fine-tune | 100.43% |
| WER sau fine-tune/test | 14.13% |
| Best validation WER | 13.10% tại epoch 49 |
| Corpus CER | 7.67% |
| Relative WER reduction | 85.93% |

Chi tiết nằm ở `experiments/01_fastconformer_ctc_v2norm_kaggle/RESULT.md`; phần nhận định và hướng đi tiếp nằm ở `insight/`.

## Báo cáo

| Báo cáo | Nội dung |
| --- | --- |
| [reports/01_fastconformer_ctc_v2norm.md](reports/01_fastconformer_ctc_v2norm.md) | Bản Markdown từ report DOCX: fine-tune FastConformer CTC trên VIVOS, kết quả WER/CER, error analysis và checklist chạy tiếp |

## Cấu trúc

```text
_works/          Ghi chú công việc cá nhân
docs/            Tài liệu tham chiếu ASR
experiments/     Ledger số liệu, config và verdict của từng run local
insight/         Nhật ký nghiên cứu động: quyết định, error analysis, hướng đi tiếp
reports/         Báo cáo đọc liền mạch từ kết quả thí nghiệm
notebooks/final/ Hai notebook cuối cho Parakeet và FastConformer
src/asr_lab/     Package tách logic khỏi notebook
```

## Kaggle

Account mặc định trong cấu hình local:

```text
trnmnhtn - Manh Tan Tran
```

Notebook FastConformer bản main chạy trực tiếp trên Kaggle bằng cách clone code từ GitHub:

```text
notebooks/final/fastconformer/02_fastconformer_main.ipynb
```

Hướng dẫn từng bước nằm ở:

```text
docs/06_benchmarks/03_kaggle_github_fastconformer.md
```
