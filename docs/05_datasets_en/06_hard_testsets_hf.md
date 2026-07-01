# 06 — Bộ test tiếng Anh KHÓ tải thẳng từ HuggingFace (để phân định model)

Mục tiêu: tìm các bộ **test split** tiếng Anh **khó hơn LibriSpeech clean** (WER kỳ vọng ~10-25%), load
được trực tiếp qua `datasets.load_dataset(...)` để chạy đo WER hàng loạt nhiều model trên CPU — giúp **phân
định model rõ ràng** (LibriSpeech dev/test-clean quá dễ, model nào cũng ~2-6% nên không tách bậc được).

> Đây là tài liệu **khảo sát** (survey), không tải dataset thật. Số liệu lấy theo nguồn gốc tại thời điểm
> 2026-06 (Open ASR Leaderboard + dataset card + paper arXiv 2510.06961 + model card Whisper-v3 / Parakeet-v2);
> chỗ ghi "cần kiểm chứng" nghĩa là nguồn không nêu rõ con số chính xác — phải tải thật ra đo lại trước khi tin.

---

## Glossary (thuật ngữ)

- **Test split:** phần dữ liệu dành riêng để đánh giá (không dùng train). Ở đây chỉ cần test split → tải nhẹ.
- **Gated (cổng khoá):** dataset (hoặc subset) bắt **đăng nhập HuggingFace + bấm "Agree" đồng ý điều khoản**
  trên trang dataset trước khi `load_dataset` chạy được. Không làm bước này sẽ bị lỗi 401/403.
- **Mở (open / public):** tải tự do, không cần đăng nhập, không cần đồng ý điều khoản.
- **trust_remote_code=True:** cho phép `datasets` chạy **script Python kèm theo dataset** để giải nén/định dạng.
  Bộ định dạng **Parquet** thuần thì KHÔNG cần; bộ có loader script cũ (như `esb/datasets`, `LIUM/tedlium`)
  thì có thể cần. Cần cẩn thận vì nó chạy code lạ.
- **streaming=True:** **không tải hết** dataset xuống đĩa; đọc từng mẫu qua mạng theo nhu cầu (lazy). Hợp khi
  chỉ cần vài chục mẫu để smoke-test, tránh tải hàng chục GB.
- **`split="test[:100]"`:** cú pháp slice của `datasets` — lấy **100 mẫu đầu** của test split (chỉ tải đúng
  phần đó nếu là Parquet/đã shard, hoặc tải shard chứa nó). Cách nhanh nhất để lấy subset nhỏ.
- **WER (Word Error Rate):** tỉ lệ lỗi từ. "WER tham khảo" dưới đây là mức **model ASR tốt** (Whisper-large-v3
  hoặc Parakeet-TDT-0.6b-v2) đạt được trên bộ đó — dùng để biết bộ nào "khó".
- **Long-form (audio dài):** AMI / Earnings là họp/cuộc gọi dài, nhiều người nói, nhiễu → KHÓ nhất.
- **Short-form (audio ngắn):** câu/đoạn ngắn đã cắt sẵn (LibriSpeech, VoxPopuli, Common Voice).

---

## Bảng các bộ test khó (xếp theo độ khó tăng dần)

WER tham khảo lấy từ 2 model tốt làm mốc: **Parakeet-TDT-0.6b-v2** (P) và **Whisper-large-v3** (W). Bộ nào WER
cao nghĩa là khó → tách bậc model tốt hơn.

| # | id HF + subset | split | giờ / mẫu test | sample rate | WER tham khảo | Mở / Gated | License |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `hf-audio/esb-datasets-test-only-sorted` · `librispeech` | `test.clean` | ~11h | 16kHz | P 1.69% / W ~2.0% | **Mở** | CC-BY-4.0 |
| 2 | `hf-audio/esb-datasets-test-only-sorted` · `spgispeech` | `test` | ~100h | 16kHz | P 2.17% | **Gated** (User Agreement) | User Agreement |
| 3 | `hf-audio/esb-datasets-test-only-sorted` · `librispeech` | `test.other` | ~11h | 16kHz | P 3.19% / W ~3.9% | **Mở** | CC-BY-4.0 |
| 4 | `hf-audio/esb-datasets-test-only-sorted` · `tedlium` | `test` | ~3h | 16kHz | P 3.38% / W ~3.9% | **Mở** | CC-BY-NC-ND 3.0 |
| 5 | `hf-audio/esb-datasets-test-only-sorted` · `voxpopuli` | `test` | ~5h | 16kHz | P 5.95% | **Mở** | CC0 |
| 6 | `hf-audio/esb-datasets-test-only-sorted` · `gigaspeech` | `test` | ~40h | 16kHz | P 9.74% | **Gated** (Apache-2.0 + agree) | apache-2.0 |
| 7 | `hf-audio/esb-datasets-test-only-sorted` · `common_voice` | `test` | ~27h | 16kHz | W ~9-12% (cần kiểm chứng) | **Gated** (CC0 + agree) | CC0-1.0 |
| 8 | `hf-audio/esb-datasets-test-only-sorted` · `earnings22` | `test` | ~5h | 16kHz | P 11.15% / W ~11.4% | **Mở** | CC-BY-SA-4.0 |
| 9 | `hf-audio/esb-datasets-test-only-sorted` · `ami` | `test` | ~9h | 16kHz | P 11.16% / **W ~16.0%** | **Mở** | CC-BY-4.0 |

**Đọc bảng này thế nào:**
- **Vùng dễ (WER < 4%):** librispeech clean/other, spgispeech, tedlium — không phân định được model tốt.
- **Vùng vừa (WER ~6-10%):** **voxpopuli** (~6%) → bắt đầu tách bậc; gigaspeech (~10%) khó hơn nhưng **gated**.
- **Vùng KHÓ (WER ~11-16%):** **earnings22** (~11%) và **AMI** (~11-16%) — chính là vùng ~10-25% cần tìm.
  AMI với Whisper-v3 ~16% là **khó nhất** trong nhóm mở (họp nhiều người, nhiễu, long-form).

> **Vì sao dùng `hf-audio/esb-datasets-test-only-sorted` là ưu tiên:** đây là bản **test-only** chính thức
> của Open ASR Leaderboard — chứa đúng 8 subset trên (cùng `common_voice`, và biến thể `voxpopuli_cleaned_aa`),
> **chỉ có test split**, đã sort theo độ dài audio. Dữ liệu lưu **Parquet** nên **KHÔNG cần trust_remote_code**.
> Một id duy nhất, đổi `name=<subset>` là chạy được nhiều bộ → rất hợp viết 1 loader chạy bench hàng loạt.

> **Cảnh báo gated (RẤT QUAN TRỌNG cho bước chạy thật):** bản thân repo `hf-audio/...` không gated, NHƯNG
> ba subset **`common_voice`, `gigaspeech`, `spgispeech`** kế thừa điều khoản gốc → phải vào trang dataset gốc
> bấm "Agree" + `huggingface-cli login` thì mới `load_dataset` được. Nếu chưa làm → BỎ ba bộ này, chỉ dùng
> các bộ MỞ.

---

## Cách load test split (snippet thực tế)

### Bộ 1 — AMI (khó nhất, MỞ) — lấy 100 mẫu đầu

```python
# uv add datasets soundfile
from datasets import load_dataset

# Parquet → KHÔNG cần trust_remote_code. AMI mở, không cần login.
ds = load_dataset("hf-audio/esb-datasets-test-only-sorted", "ami", split="test[:100]")

ex = ds[0]
audio = ex["audio"]          # dict: {"array": np.ndarray, "sampling_rate": 16000, "path": ...}
wav = audio["array"]         # mảng numpy float32, mono
sr = audio["sampling_rate"]  # 16000
ref = ex["text"]             # transcript tham chiếu (tên cột là "text" cho cả 8 subset của repo này)
print(sr, len(wav), ref[:80])
```

### Bộ 2 — Earnings-22 (khó, MỞ) — streaming + lấy 50 mẫu (không tải hết)

```python
from datasets import load_dataset
from itertools import islice

ds = load_dataset("hf-audio/esb-datasets-test-only-sorted", "earnings22",
                  split="test", streaming=True)   # lazy, không tải hết
for ex in islice(ds, 50):
    wav = ex["audio"]["array"]
    sr  = ex["audio"]["sampling_rate"]   # 16000
    ref = ex["text"]
    # ... đưa wav vào model, so với ref để tính WER
```

### Bộ 3 — VoxPopuli (vừa, MỞ) — lấy cả split test rồi slice

```python
from datasets import load_dataset

ds = load_dataset("hf-audio/esb-datasets-test-only-sorted", "voxpopuli", split="test")
small = ds.select(range(100))            # 100 mẫu đầu
for ex in small:
    wav, sr, ref = ex["audio"]["array"], ex["audio"]["sampling_rate"], ex["text"]
```

### (Tuỳ chọn) Bộ lẻ ngoài repo gộp — nếu muốn dùng id riêng

```python
# LibriSpeech test-other (id chính chủ, MỞ, KHÔNG cần trust_remote_code — bản parquet mới):
ld = load_dataset("openslr/librispeech_asr", "other", split="test[:100]")
# field text ở bản này tên là "text"

# TED-LIUM release3 (MỞ; bản loader cũ có thể CẦN trust_remote_code):
# tl = load_dataset("LIUM/tedlium", "release3", split="test[:50]", trust_remote_code=True)
# field text ở TED-LIUM tên là "text"
```

> **Lưu ý field audio:** mọi bộ trên HF trả audio dạng dict `{"array", "sampling_rate", "path"}`. Tất cả đã
> 16kHz nên KHÔNG cần resample. Nếu một bộ lẻ khác trả sr ≠ 16000 → dùng `ds.cast_column("audio",
> Audio(sampling_rate=16000))` trước khi đọc.

---

## KHUYẾN NGHỊ shortlist cho lab CPU

Chỉ chọn bộ **MỞ** (tránh gated để khỏi vướng login khi chạy bench), khó dần, chạy subset ~50-100 mẫu cho nhanh
trên CPU. Tất cả qua **một id** `hf-audio/esb-datasets-test-only-sorted`, **KHÔNG cần trust_remote_code**:

| Thứ tự | subset | split gợi ý | WER kỳ vọng (model tốt) | Vai trò |
| --- | --- | --- | --- | --- |
| 1 | `voxpopuli` | `test[:100]` | ~6% | Mức "vừa" — bắt đầu tách bậc model |
| 2 | `earnings22` | `test[:100]` | ~11% | KHÓ — giọng tài chính đa quốc, vào đúng vùng 10-25% |
| 3 | `ami` | `test[:100]` | **~11-16%** | **KHÓ NHẤT (mở)** — họp nhiều người, nhiễu, long-form |
| (phụ) | `librispeech` | `test.other` | ~3% | Mốc "dễ" để đối chiếu, xác nhận loader đúng |

**Ước tính vùng WER ~15-20%:** **AMI** là bộ sát nhất (Whisper-v3 ~16%; model yếu hơn sẽ ≥20%). Earnings22
(~11%) là bậc dưới. Hai bộ này đủ để xếp hạng model rõ ràng trên CPU mà không cần đụng bộ gated.

**Bộ GATED phải bỏ (nếu không đăng nhập + đồng ý điều khoản):**
`common_voice`, `gigaspeech`, `spgispeech`. Trong đó `gigaspeech` (~10%) và `common_voice` (~9-12%) khá khó —
nếu sau này chịu khó làm bước đăng nhập HF + Agree thì bổ sung được; còn `spgispeech` (~2%) dễ nên không tiếc.

---

## Lưu ý field text + chuẩn hoá (tránh tính WER sai)

**Với repo `hf-audio/esb-datasets-test-only-sorted`** (khuyến nghị dùng): mọi subset đã chuẩn hoá về **cùng tên
cột `text`** → viết loader một lần dùng cho cả 8 bộ. Đây là lợi thế lớn so với dùng id lẻ.

**Nếu dùng id dataset LẺ thì tên cột transcript KHÁC NHAU** — phải map đúng kẻo đọc nhầm cột rỗng:

| Dataset (id lẻ) | Tên cột transcript | Ghi chú chuẩn hoá |
| --- | --- | --- |
| `openslr/librispeech_asr` | `text` | đã chữ HOA hết, không dấu câu |
| `LIUM/tedlium` | `text` | có token `<unk>`; câu "ignore_time_segment_in_scoring" cần loại |
| `facebook/voxpopuli` | `normalized_text` (và `raw_text`) | dùng `normalized_text` để so cho khớp |
| `mozilla-foundation/common_voice_*` | `sentence` | CÓ dấu câu + chữ hoa → phải chuẩn hoá mạnh |
| `speechcolab/gigaspeech` | `text` | có token đặc biệt `<COMMA> <PERIOD> <QUESTIONMARK>` cần xử lý |
| `distil-whisper/earnings22` | `transcription` | kiểm tra lại tên cột theo bản phát hành |

**Chuẩn hoá tối thiểu trước khi tính WER (cho cả ref lẫn hypothesis của model):**
- Hạ thường (lowercase).
- Bỏ dấu câu (`. , ? ! " ' :` ...).
- Gộp khoảng trắng thừa.
- Loại token đặc biệt của từng bộ (`<unk>`, `<COMMA>`, `<PERIOD>` ...).
- KHÔNG so trực tiếp text thô của Common Voice/Gigaspeech với output model → WER sẽ phồng giả tạo.

> Nên dùng bộ chuẩn hoá thống nhất (vd hàm normalizer của Whisper, hoặc `jiwer` + transform) áp **cùng một
> cách** cho cả tham chiếu và dự đoán — đây là cách Open ASR Leaderboard làm để số liệu so sánh được.

---

## ✅ Tự kiểm nhanh

1. Trong các bộ MỞ (không gated), bộ nào khó nhất và WER tham khảo khoảng bao nhiêu?
2. Ba subset nào của `hf-audio/esb-datasets-test-only-sorted` bị gated phải bỏ nếu không đăng nhập HF?

<details>
<summary>Đáp án</summary>

1. **AMI** (`ami`, split `test`) — bộ họp nhiều người nói, nhiễu, long-form; model tốt như Whisper-large-v3
   vẫn ~16% WER (Parakeet-v2 ~11%). Là bộ mở khó nhất, sát vùng 15-20% cần tìm.
2. **`common_voice`, `gigaspeech`, `spgispeech`** — kế thừa điều khoản gốc, phải `huggingface-cli login` +
   bấm "Agree" trên trang dataset gốc. Không làm thì `load_dataset` lỗi 401/403 → bỏ, dùng các bộ mở
   (`voxpopuli`, `earnings22`, `ami`, `librispeech`). Cả repo dùng cột text tên `text`, lưu Parquet nên
   **KHÔNG cần trust_remote_code**.

</details>
