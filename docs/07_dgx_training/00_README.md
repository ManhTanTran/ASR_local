# 07 — Kế hoạch dữ liệu + huấn luyện ASR tiếng Việt trên DGX

- entry point cho cụm tài liệu
  - **chuyển từ khảo sát sang huấn luyện thật**
  - lab QASI đã có tài nguyên đầu tiên: máy **DGX Spark nội bộ** (`edgexpert-06b7`).
- Mục tiêu tuần:
  - (a) kéo bộ dữ liệu ASR tiếng Việt công khai về làm **tài sản data dùng chung**;
  - (b) chốt **thang huấn luyện easy→hard** để model hội tụ dần;
  - (c) tích lũy checkpoint đầu tiên cho team.

> Tài liệu này gom kết quả khảo sát (`docs/04_datasets_vi`) + research trick training + hiện trạng DGX thành một kế hoạch chạy được.
> Số GPU-hour là **ước lượng bậc độ lớn**, phải đo lại trên máy.

---

## Glossary (thuật ngữ)

- **Curriculum learning:** huấn luyện từ mẫu dễ đến khó (câu ngắn/âm sạch trước) để hội tụ nhanh, ít phân kỳ.
- **Catastrophic forgetting (quên tai hại):**
  - thêm domain mới làm model quên domain cũ.
  - Chống bằng **replay **(giữ data cũ trong mix) + theo dõi val set cũ.
- **RNNT / Transducer:**
  - kiến trúc ASR có LM ngầm, WER offline tốt; nặng bộ nhớ khi train.
- **CTC:**
  - decoder đơn giản, nhẹ, nhanh;
  - kém RNNT chút nhưng đủ cho streaming/throughput cao.
- **SpecAugment:**
  - che ngẫu nhiên dải tần/thời gian trên spectrogram để tăng bền vững, chống overfit.
- **bf16-mixed:**
  - tính nửa độ chính xác + cộng dồn trọng số FP32;
  - mặc định an toàn trên Blackwell.
- **Tarred + bucketing:**
  - đóng gói audio thành tar + gom câu cùng độ dài → giảm padding, tăng throughput.
- **Manifest NeMo:** `.jsonl`, mỗi dòng `{"audio_filepath","duration","text"}`.

---

## Hiện trạng DGX (đo 2026-07-01)

| Hạng mục       | Giá trị                                                | Ý nghĩa cho training                                                      |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| GPU            | 1× NVIDIA GB10 (Blackwell sm_121)                      | 1 GPU desktop-class — fine-tune được, KHÔNG train-from-scratch model lớn  |
| Unified memory | 121 GiB (share CPU+GPU)                                | dư bộ nhớ cho batch RNNT lớn; nghẽn ở băng thông LPDDR5X, không phải VRAM |
| CUDA / driver  | 13.0 / 580.159                                         | cần torch ≥2.9 cu13x để có kernel sm_121 native                           |
| Disk`/srv`     | 3.6 TB, trống 1.6 TB                                   | đủ cho toàn corpus (~180GB ungated + gated ~320GB)                        |
| Cache chung    | `HF_HOME=/srv/team-share/cache/hf`, `UV_CACHE_DIR=...` | mọi user dùng chung, không tải lại                                        |
| Throughput HF  | ~24 MB/s (~86 GB/h)                                    | kéo 180GB ≈ 2h                                                            |
| Tồn đọng       | ~1.1TB rác (htt210/hieutb) chưa dọn                    | không chặn việc này nhưng nên dọn sau                                     |

**Giới hạn cứng:** GB10 là 1 máy đơn. Chiến lược = **fine-tune từ checkpoint**, không pretrain từ đầu.
Model khả thi: FastConformer Transducer ~115M (thoải mái) → 600M–1B (chậm, vẫn được với grad-accum).

---

## Cụm tài liệu

- [`01_corpus_pull.md`](01_corpus_pull.md) — bộ dữ liệu chọn kéo về, license, công cụ `tools/pull_datasets`, trạng thái kéo.
- [`02_training_ladder.md`](02_training_ladder.md) — thang curriculum easy→hard, trick tăng WER, nhóm model nên train, kỳ vọng thực tế trên GB10, các bước chạy.

### Plan chi tiết thực thi (03-07)
- [`03_tokenizer_vocab.md`](03_tokenizer_vocab.md) — quyết định tokenizer: **vá NFC**, rebuild từ toàn corpus, vocab **1024**, whitelist charset, chính sách số.
- [`04_data_normalization_manifest.md`](04_data_normalization_manifest.md) — builder thống nhất (`build_corpus.py`), adapter mỗi nguồn, chuẩn hóa 16k/NFC, eval cố định, **tarred** cho bộ lớn.
- [`05_model_batch_epoch.md`](05_model_batch_epoch.md) — kiến trúc + ước lượng bộ nhớ RNNT → batch **32** + grad-accum, epoch/steps từng nấc, bf16-mixed.
- [`06_resource_budget_time.md`](06_resource_budget_time.md) — ước lượng giờ train (~54–135h cả 3 nấc), ngân sách đĩa, **giao ước chia máy chung** (tmux, max_time, preemptible, giờ thấp điểm).
- [`07_training_lifecycle.md`](07_training_lifecycle.md) — save/load/resume: bật ModelCheckpoint top-k, resume `.ckpt` vs sang-nấc `.nemo`, export + backup team-share, registry.
- [`08_new_system_design.md`](08_new_system_design.md) — **thiết kế hệ training mới** config-driven (`train/vi/` + `configs/*.yaml`): tách khỏi code cũ đóng băng, thêm dataset không sửa lõi, lộ trình lát mỏng.
- [`09_splits_eval_lineage.md`](09_splits_eval_lineage.md) — **CHỐT chống confuse**: split mọi dataset (train/val/test + phiên bản), metric đo trên tập nào (val per-run vs test cố định), lỗ hổng coverage, lineage model qua các nấc. Đọc trước khi bàn kết quả.

---

## Tóm tắt định hướng (1 đoạn)

Kéo ~180GB dataset ungated (VIVOS→CV→FLEURS→InfoRE→FOSD→VLSP→LSVSC→VietSuperSpeech→Bud500→InfoRE2) về `/srv/team-share/datasets/asr_vi`.
Huấn luyện **offline hybrid Transducer-CTC ~115M**, hai-ba nấc easy→hard với **replay** chống quên, `bf16-mixed`, cosine LR ~1e-4, tarred+bucketing, chọn checkpoint theo val WER.
Chứng minh máy bằng smoke 50-step trước khi chạy dài.
Sản phẩm: checkpoint tiếng Việt mạnh dần + bộ eval cố định (FLEURS-vi + CV test) làm thước đo chung cho lab.
