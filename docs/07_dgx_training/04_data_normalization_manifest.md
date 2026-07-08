# 07.04 — Logic chuẩn hóa data → format NeMo

Cách biến data thô đã kéo (`/srv/team-share/datasets/asr_vi/<name>/`, mỗi bộ schema khác nhau) thành manifest NeMo chuẩn `{audio_filepath, duration, text}` + đóng tarred cho throughput.
Bám pattern có sẵn: `data/vivos.py` + `data/common_voice.py` (`dump_split`, `to_16k_mono`).

---

## Glossary

- **Manifest NeMo:** `.jsonl`, 1 dòng/utterance: `{"audio_filepath","duration","text"}`.
- **Tarred dataset:**
  - gói audio thành `.tar` + manifest sharded → tránh hàng triệu file lẻ (chết filesystem chung), đọc tuần tự nhanh.
  - Sinh bằng `convert_to_tarred_audio_dataset.py` (NeMo).
- **Bucketing:** gom câu cùng độ dài vào 1 bucket → giảm padding khi train.

---

## Vấn đề: mỗi nguồn một schema

Mỗi dataset HF khác nhau cột text + kiểu audio:

| Nguồn                              | Cột text                             | Audio         | Ghi chú                      |
| ---------------------------------- | ------------------------------------ | ------------- | ---------------------------- |
| VIVOS mirror                       | `transcription`                      | bytes wav     | đã xử lý (`vivos.py`)        |
| Common Voice                       | `sentence` (bỏ `whisper_transcript`) | mp3 bytes 48k | đã xử lý (`common_voice.py`) |
| FLEURS-vi                          | `transcription`/`raw_transcription`  | array/wav     | có sẵn split train/val/test  |
| doof-ferb (FOSD/VLSP/InfoRE/LSVSC) | `transcription`/`text`/`sentence`    | mp3/wav bytes | cần dò cột thật từng bộ      |
| VietSuperSpeech                    | `text`/`transcription`               | wav           | auto-label                   |
| Bud500                             | `transcription`                      | array         | ~500h, shard parquet         |

→ Không hardcode được. Cần **adapter registry**: mỗi bộ khai báo (cột text, cách lấy audio, sr gốc).

---

## Thiết kế: `asr_lab/data/build_corpus.py` (mới)

Một builder tổng, đọc `datasets.yaml` (đã có `repo_id`, `name`, `license`) + bảng adapter:

```
ADAPTERS = {
  "fleurs_vi":       {"text": ["transcription","raw_transcription"], "audio": "array", "sr": 16000},
  "fosd":            {"text": ["transcription","text","sentence"],   "audio": "bytes", "sr": None},
  "bud500":          {"text": ["transcription"],                      "audio": "array", "sr": 16000},
  ...  # dò 1 lần bằng ds.features, ghi cứng vào đây
}
```

**Luồng mỗi bộ** (tái dùng `to_16k_mono` + `normalize_vi` đã vá NFC):

1. `load_dataset("parquet", data_files=asr_vi/<name>/**/*.parquet)` — đọc từ snapshot LOCAL, không tải lại.
2. Lấy text theo adapter → `normalize_vi` (NFC + lower + bỏ dấu câu) → **lọc whitelist** (xem [03](03_tokenizer_vocab.md) QĐ-4).
3. Lọc clip: text rỗng → bỏ; `duration ∉ [0.3, 30]s` → bỏ (câu quá ngắn/dài hại RNNT); ký tự ngoài whitelist >1% → bỏ.
4. Decode audio → `to_16k_mono` → ghi `wav` (hoặc giữ để đóng tar ở bước sau).
5. Ghi manifest `asr_vi/_manifests/<name>.jsonl` + `<name>.stats.json` (giờ, #clip, charset, phân bố độ dài).

**Chuẩn hóa đồng nhất mọi nguồn:** 16kHz mono float32 wav; text NFC-lower-no-punct; duration làm tròn 3 số.

---

## Split & eval cố định (quan trọng cho so sánh công bằng)

- **Eval set khóa cứng:** FLEURS-vi test + CV test (giữ nguyên xuyên mọi stage/model). Không bao giờ train trên 2 bộ này.
- **Train/val mỗi nguồn:** nguồn có sẵn split thì theo; nguồn không có → cắt val 2-5% từ đuôi (như code hiện tại).
- **Chống rò rỉ (leakage):** cùng speaker/câu không nằm cả train lẫn eval. FLEURS/CV có split sẵn nên an toàn; các bộ tự cắt val thì chấp nhận rủi ro thấp (ghi chú).
- **Dedup:** hash (text + duration) loại trùng trong 1 nguồn và giữa các nguồn overlap (InfoRE1 vs InfoRE2).

---

## Lưu trữ: dùng tarred cho bộ lớn (bắt buộc)

1.660h @ 16k mono 16-bit ≈ **180GB wav + 1.5 triệu file lẻ** → giết filesystem chung `/srv`.

- Bộ nhỏ (S1: VIVOS/CV/FLEURS/InfoRE1) — để wav thường, tiện debug.
- Bộ lớn (VLSP/LSVSC/VietSuperSpeech/Bud500/InfoRE2) — **đóng tarred** ngay sau build:
  `python <nemo>/scripts/speech_recognition/convert_to_tarred_audio_dataset.py --manifest_path=<name>.jsonl --target_dir=asr_vi/_tarred/<name> --num_shards=64 --max_duration=30 --min_duration=0.3`
- Train đọc tarred qua `is_tarred=True` + `tarred_audio_filepaths` + `bucketing_batch_size`.

Đích: `asr_vi/_manifests/` (jsonl + stats), `asr_vi/_tarred/<name>/` (tar + sharded manifest), `asr_vi/_stage/{s1,s2,s3}.jsonl` (manifest gộp theo nấc curriculum, có trọng số upsample tập sạch nhỏ).

---

## Việc code (thứ tự)

1. Vá `normalize_vi` (NFC) — [03](03_tokenizer_vocab.md) QĐ-1.
2. Dò schema thật từng bộ (`ds.features`) → điền `ADAPTERS`.
3. Viết `build_corpus.py` (dùng lại `to_16k_mono`, `normalize_vi`), sinh manifest + stats từng bộ.
4. Dedup + build manifest gộp theo stage (+ trọng số replay).
5. Đóng tarred cho bộ lớn.
6. In bảng tổng: mỗi bộ giờ thực / #clip / #drop / charset — đối chiếu với ước lượng ở [01](01_corpus_pull.md).

## ✅ Tự kiểm nhanh

1. Vì sao cần adapter registry? 2. Vì sao đóng tarred cho bộ lớn? 3. Bộ nào KHÔNG bao giờ được train?

<details><summary>Đáp án</summary>
1. Mỗi nguồn HF có cột text + kiểu audio khác nhau, không hardcode được. 2. ~1.5tr file wav lẻ giết filesystem chung; tar đọc tuần tự nhanh. 3. FLEURS-vi test + CV test (eval cố định).
</details>
