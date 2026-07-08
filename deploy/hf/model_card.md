---
language:
- vi
license: cc-by-nc-sa-4.0
library_name: nemo
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- speech
- vietnamese
- nemo
- fastconformer
- rnnt
- conversational
---

# vi-asr-fastconformer-114m (S3 — hội thoại/callbot)

> **NỘI BỘ / RESEARCH ONLY.** Model train trên corpus trộn license gồm nhiều bộ **non-commercial** và có bộ **cấm phân phối lại** (xem [§License & nguồn data](#license--nguồn-data)). KHÔNG dùng thương mại, KHÔNG phân phối lại ra ngoài team khi chưa gỡ ràng buộc data.

Model **FastConformer-Transducer 114M** nhận dạng tiếng nói tiếng Việt, tự gom ~14 dataset public + **curriculum 3 nấc** (đọc-sạch → tự-nhiên → hội-thoại). Mục tiêu chính: **callbot / hội thoại tổng đài**. Dùng nội bộ team để suy luận và làm **khởi tạo (init) cho các task downstream** (LALM, turn-detection, projector...).

- **Kiến trúc:** `EncDecRNNTBPEModel` (FastConformer encoder d_model 512 / 17 lớp + RNNT decoder+joint), 114M tham số.
- **Tokenizer:** SentencePiece BPE 1024 (charset VIVOS-era; **thiếu f/j/w/z** → xem [§Hạn chế](#hạn-chế)).
- **Sample rate:** 16 kHz mono.
- **Lineage:** `base (English) → v1 VIVOS → v2norm (NFC) → S1 → S2 → S3` (bản này).

## Kết quả (S3, đủ 4 epoch)

WER trên suite 9 test cố định, chuẩn hoá thống nhất (giữ dấu tiếng Việt, bỏ dấu câu, NFC).

| Test | Đặc thù | WER |
| --- | --- | --- |
| `bud500` | hội thoại 3 vùng miền | **6,73%** |
| `vivos` | đọc studio | **8,47%** |
| `lsvsc` | tự nhiên đa miền | **13,12%** |
| `fleurs` | đọc studio chuẩn quốc tế | **16,46%** |
| `common_voice` | mic đời thường | **17,19%** |
| `fosd` | đọc FPT | **19,96%** |
| `vietsuperspeech` | **hội thoại (gần callbot nhất)** | **22,87%** |
| `vlsp` | tin tức formal | **24,81%** |
| `vietmed` | y tế (probe, KHÔNG train) | **26,38%** |

Curriculum kéo callbot từ 40,00% → **22,87%** mà không quên nền đọc-sạch (cả 9/9 test cải thiện qua các nấc).

## Cách dùng

Cần `nemo-toolkit[asr]` (khớp bản train: 2.7.3) + `huggingface_hub`. Repo private → cần được add collaborator + đăng nhập HF.

```python
from huggingface_hub import hf_hub_download
import nemo.collections.asr as nemo_asr

path = hf_hub_download(repo_id="kyle/vi-asr-fastconformer-114m",
                       filename="s3-fc115m-full.nemo")
model = nemo_asr.models.ASRModel.restore_from(path)   # thêm map_location="cuda" nếu có GPU
print(model.transcribe(["cau_noi.wav"]))              # wav bất kỳ SR, NeMo tự resample 16k
```

Chuẩn hoá text để so WER công bằng: dùng `normalize_vi` (giữ dấu, bỏ dấu câu, NFC) — xem `deploy/asr_vi/_common.py` trong repo train, hoặc script `infer.py`/`eval_wer.py` chạy độc lập kèm theo.

### Dùng làm init cho downstream

- **Encoder cho LALM / projector:** lấy `model.encoder` (FastConformer) làm speech-encoder, nối projector sang LLM (SLAM-ASR style).
- **Turn-detection / barge-in:** dùng đặc trưng encoder hoặc output RNNT làm tín hiệu.
- File `.nemo` gói trọn config + tokenizer → `restore_from` là đủ, không cần dựng lại kiến trúc.

## License & nguồn data

Model là **tác phẩm phái sinh** từ corpus trộn license. Phát hành nội bộ theo **CC BY-NC-SA 4.0** + nêu nguồn:

| License nguồn | Dataset (đã đưa vào train) |
| --- | --- |
| commercial-ok (CC-BY / CC0 / MIT) | common_voice_vi, fleurs_vi, fosd, vlsp2020_100h, lsvsc, vietsuperspeech |
| research-nc (CC BY-NC-SA) | vivos, bud500, viVoice, infore1, infore2_audiobooks |
| **cấm phân phối lại** | **phoaudiobook** |

- Vì có bộ **NC-SA** và **phoaudiobook cấm phân phối lại**, model **không** phát hành thương mại/permissive được.
- Muốn có bản dùng thương mại: phải train lại **chỉ trên nhóm commercial-ok** (bản riêng, chất lượng hội thoại thấp hơn do bỏ bud500/viVoice).
- `vietmed` (NC) chỉ dùng **eval**, KHÔNG train → không dính vào trọng số.

## Hạn chế

- **Trần loanword:** tokenizer thiếu `f/j/w/z` → từ nước ngoài (wifi, facebook, zalo…) bị sai. Đo thật: 34% câu callbot chứa ký tự này. Đã thử nhánh **rebuild-vocab (s3rv)** nhưng `change_vocabulary` reset decoder+joint về ngẫu nhiên → 3 epoch không đủ phục hồi, WER **xấu hơn S3 ở cả 9 test** → chưa dùng. Hướng gỡ đúng còn để mở: rebuild vocab ngay từ nấc S1, hoặc train s3rv dài hơn (5-8 epoch).
- Nhãn một số bộ lớn (viVoice/bud500) là pseudo-label → nhiễu nhãn có thể giới hạn trần.
- Offline (không streaming); adapt streaming là hướng sau.

## Phiên bản

| Version | File | Nấc | Ghi chú |
| --- | --- | --- | --- |
| v1 | `s3-fc115m-full.nemo` | S3 (4 epoch) | **bản tốt nhất hiện tại** — hội thoại/callbot |

> Nhánh `s3rv` (rebuild-vocab) đã thử nhưng WER xấu hơn S3 ở cả 9 test (reset decoder+joint không kịp phục hồi) → không phát hành. v2 để mở cho hướng gỡ loanword tốt hơn.
