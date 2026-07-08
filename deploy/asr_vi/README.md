# Package ASR tiếng Việt — chạy suy luận / test độc lập

Gói tự chứa để **chạy nhận dạng tiếng nói tiếng Việt** từ model đã train (`.nemo`),
KHÔNG cần toàn bộ repo train. Chỉ gồm: 2 script (`infer.py`, `eval_wer.py`) + hàm chuẩn hoá
dùng chung (`_common.py`) + model `.nemo` (chép riêng, xem [§Model](#model)).

Model: **FastConformer-Transducer 114M**, tự gom ~14 dataset public + curriculum 3 nấc
(đọc-sạch → tự-nhiên → hội-thoại). Mục tiêu chính: **callbot / hội thoại tổng đài (FCI)**.

---

## Kết quả (S3, đủ 4 epoch — bản bàn giao hiện tại)

WER đo trên suite 9 test cố định (mỗi test 1 câu hỏi khác nhau), chuẩn hoá `normalize_vi` thống nhất.

| Test | Đặc thù | WER |
| --- | --- | --- |
| `bud500_test` | hội thoại 3 vùng miền | **6,73%** |
| `vivos_test` | đọc studio | **8,47%** |
| `lsvsc_test` | tự nhiên đa miền | **13,12%** |
| `fleurs_test` | đọc studio chuẩn quốc tế | **16,46%** |
| `cv_test` | mic đời thường (Common Voice) | **17,19%** |
| `fosd_test` | đọc FPT | **19,96%** |
| `vietsuperspeech_test` | **hội thoại (gần callbot nhất)** | **22,87%** |
| `vlsp_test` | tin tức formal | **24,81%** |
| `vietmed_test` | y tế (probe, KHÔNG có trong train) | **26,38%** |

- Curriculum S1→S2→S3 kéo callbot từ 40,00% → **22,87%** mà không quên nền đọc-sạch.
- Mốc so sánh: `cv_test` 17,19% ≈ PhoWhisper-base 74M (16,19%) — lưu ý normalizer khác nhau.
- Chi tiết lineage + before/after từng nấc: `docs/07_dgx_training/09_splits_eval_lineage.md`
  và `experiments/05..07/RESULT.md` trong repo train.

> ⚠️ **Trần loanword:** tokenizer bản này thiếu `f/j/w/z` → từ nước ngoài (wifi, facebook, zalo…)
> bị sai. Nhánh **rebuild-vocab** đang train để gỡ (34% câu callbot có ký tự này). Khi xong sẽ có
> `s3rv-fc115m-full.nemo` thay thế — cùng cách chạy, chỉ đổi đường dẫn `--nemo`.

---

## Model

Không commit `.nemo` (457MB) vào git. Lấy từ kho model của team rồi để cạnh script:

```bash
# trên DGX (hoặc scp về máy):
cp /srv/team-share/models/asr_vi/s3-fc115m-full.nemo deploy/asr_vi/
```

| File | Nấc | Ghi chú |
| --- | --- | --- |
| `s3-fc115m-full.nemo` | S3 (4 epoch) | **bản dùng hiện tại** — hội thoại/callbot |
| `s2-fc115m-full.nemo` | S2 | tự nhiên/formal (không hội thoại) |
| `s1-fc115m-full.nemo` | S1 | đọc-sạch (baseline curriculum) |
| `s3rv-fc115m-full.nemo` | S3 rebuild-vocab | *(đang train — gỡ trần loanword)* |

---

## Setup môi trường

**Cách 1 — ephemeral (không cần project), gọn nhất cho máy lạ:**

```bash
# uv tự dựng env tạm với đúng deps, chạy 1 phát. Torch bản CPU (chậm, hợp máy không GPU).
uv run --with "nemo-toolkit[asr]==2.7.3" --with soundfile python infer.py \
    --nemo s3-fc115m-full.nemo --audio mau.wav
```

**Cách 2 — trên DGX (GPU, nhanh):** dùng luôn `.venv` của repo train (đã có nemo + torch cu130):

```bash
cd /srv/team-share/projects/nvidia_asr_nemo
PYTHONPATH=deploy/asr_vi .venv/bin/python deploy/asr_vi/infer.py \
    --nemo /srv/team-share/models/asr_vi/s3-fc115m-full.nemo --audio mau.wav
```

Yêu cầu: Python ≥ 3.12, `nemo-toolkit[asr]==2.7.3`. Audio đầu vào bất kỳ sample-rate/kênh —
NeMo tự resample 16k mono khi transcribe.

---

## Dùng

### 1) Chép lời 1 file / thư mục

```bash
python infer.py --nemo s3-fc115m-full.nemo --audio cau_noi.wav          # in đúng 1 dòng text
python infer.py --nemo s3-fc115m-full.nemo --audio thu_muc_wav/ --batch 16   # "path <tab> text"
python infer.py --nemo s3-fc115m-full.nemo --audio cau_noi.wav --raw    # text thô, không normalize
```

### 2) Đo WER trên tập test (tái tạo bảng trên)

```bash
# 1 hay nhiều manifest, đặt tên tuỳ ý:
python eval_wer.py --nemo s3-fc115m-full.nemo \
    --manifests vivos=vivos.test.jsonl callbot=vietsuperspeech.test.jsonl

# hoặc quét cả thư mục manifest (mọi *.test.jsonl):
python eval_wer.py --nemo s3-fc115m-full.nemo \
    --dir /srv/team-share/datasets/asr_vi/_manifests --batch 16
```

Manifest mỗi dòng: `{"audio_filepath": "...", "duration": <giây>, "text": "nhãn"}`.
Có cột `text` thì `infer.py` (dạng `.jsonl`) và `eval_wer.py` tự tính WER.

---

## Kiểm nhanh (PASS khi)

1. `infer.py` trên 1 wav tiếng Việt → in ra câu có dấu, hợp nghĩa.
2. `eval_wer.py --dir <manifests>` → bảng WER khớp ± sai số nhỏ với bảng ở trên
   (chênh do batch/thứ tự float, không do logic).
3. Log stderr in đúng `vocab=1024` và `device=cuda` (trên DGX) hoặc `cpu`.

> Số WER chỉ khớp khi **cùng `normalize_vi`** — đừng đổi `_common.py` lệch bản gốc.
