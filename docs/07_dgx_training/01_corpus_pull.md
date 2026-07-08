# 07.01 — Corpus ASR tiếng Việt: chọn lọc + kéo về DGX

Danh mục dataset công khai chọn kéo về `/srv/team-share/datasets/asr_vi/` làm tài sản chung.
Nguồn khảo sát chi tiết từng bộ: [`docs/04_datasets_vi`](../04_datasets_vi/00_INDEX.md).
Công cụ kéo: [`tools/pull_datasets`](../../tools/pull_datasets/README.md) (manifest `datasets.yaml`).

---

## Bảng dataset (đã verify HF id)

| #   | Dataset             | HF id                                           | Giờ  | ~GB | License       | Stage | Domain        | Gated |
| --- | ------------------- | ----------------------------------------------- | ---- | --- | ------------- | ----- | ------------- | ----- |
| 1   | VIVOS               | `AILAB-VNUHCM/vivos`                            | 15   | 1.5 | research-nc   | 1     | đọc           | —     |
| 2   | Common Voice VI     | `tsdocode/common_voice_13_0_vi_pseudo_labelled` | 4    | 0.1 | commercial-ok | 1     | đọc           | —     |
| 3   | FLEURS-vi           | `google/fleurs` (vi_vn)                         | 12   | 2   | commercial-ok | 1     | đọc           | —     |
| 4   | InfoRE 1            | `doof-ferb/infore1_25hours`                     | 25   | 3   | research-nc   | 1     | đọc           | —     |
| 5   | FOSD (FPT)          | `doof-ferb/fpt_fosd`                            | 100  | 1   | commercial-ok | 2     | đọc           | —     |
| 6   | VLSP2020-100h       | `doof-ferb/vlsp2020_vinai_100h`                 | 100  | 11  | commercial-ok | 2     | đọc           | —     |
| 7   | LSVSC               | `doof-ferb/LSVSC`                               | 389  | 12  | commercial-ok | 2     | tự nhiên      | —     |
| 8   | VietSuperSpeech     | `thanhnew2001/VietSuperSpeech`                  | 103  | 12  | commercial-ok | 3     | **hội thoại** | —     |
| 9   | Bud500              | `linhtran92/viet_bud500`                        | 500  | 98  | research-nc   | 3     | tự nhiên      | —     |
| 10  | InfoRE 2 audiobooks | `doof-ferb/infore2_audiobooks`                  | 415  | 40  | research-nc   | 3     | audiobook     | —     |
| —   | PhoAudiobook        | `thivux/phoaudiobook`                           | 941  | 100 | research-nc   | 3     | audiobook     | **✓** |
| —   | viVoice             | `capleaf/viVoice`                               | 1017 | 110 | research-nc   | 3     | tự nhiên      | **✓** |
| —   | VietMed             | `leduckhai/VietMed`                             | 16   | 2   | research-nc   | 3     | y tế          | **✓** |

- **Ungated (auto kéo):** 10 bộ đầu, 1.660h, **182 GB** (~2h @ 24MB/s).
- **Gated (chờ token):** 3 bộ cuối cần chấp nhận điều khoản HF + `HF_TOKEN`. Kéo bằng `pull.py --only <name> --include-gated --token <HF_TOKEN>`.

---

## Lưu ý license (quan trọng cho FCI thương mại)

- **commercial-ok** (CC0 / CC-BY / MIT): CV, FLEURS, FOSD, VLSP2020, LSVSC, VietSuperSpeech.
  → được dùng cho model bán ra (sản phẩm FCI).
- **research-nc** (NC / cấm phân phối lại): VIVOS, Bud500, InfoRE1/2, PhoAudiobook, viVoice, VietMed.
  → **chỉ nghiên cứu/tích lũy nội bộ**.
  - Khi build model thương mại phải **segregate** (train riêng tập commercial-ok, hoặc chỉ dùng NC ở giai đoạn nghiên cứu).

Cột `license` trong `datasets.yaml` giữ nhãn này để bước build manifest lọc theo mục đích.

---

## Trạng thái kéo

- Batch ungated stage 1-3 chạy nền trên DGX: `asr_vi/_tool/pull.log`, tiến độ `asr_vi/_pull_status.json`.
- Mỗi dataset xong có cờ `asr_vi/<name>/.done`; chạy lại `pull.py` sẽ bỏ qua bộ đã xong (idempotent).
- Kiểm tra nhanh: `ssh dgx 'cat /srv/team-share/datasets/asr_vi/_pull_status.json'`.

## Bước tiếp sau khi kéo xong

1. **Build manifest NeMo** cho từng bộ: đọc parquet/audio local → `{audio_filepath,duration,text}.jsonl`, chuẩn hóa text bằng `asr_lab.common.metrics.normalize_vi`, chạy cổng OOV/charset (đã có ở repo).
2. **Tách eval cố định**: FLEURS-vi test + CV test giữ nguyên xuyên mọi stage để so sánh công bằng.
3. **Chuyển tarred + bucketing** cho các bộ lớn (Bud500, InfoRE2) để tăng throughput khi train nhiều epoch.
