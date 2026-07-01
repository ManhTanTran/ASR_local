# 08 — infore (25h + audiobooks)

Hai bộ liên quan, hay được nhắc trong các model wav2vec2 tiếng Việt (vd `wav2vec2-base-vietnamese-250h`).
Trên HuggingFace do `doof-ferb` mirror.

---

## Số liệu

- **infore1_25hours** (`doof-ferb/infore1_25hours`):
  - **Loại:** đọc văn bản.
  - **Giờ audio:** ~**25h**.
  - **Transcript:** có.
- **infore2_audiobooks** (`doof-ferb/infore2_audiobooks`):
  - **Loại:** **sách nói (audiobook)** — đoạn dài hơn.
  - **Giờ audio:** lớn hơn nhiều (cần kiểm chứng con số chính xác trên thẻ).
  - **Transcript:** có.
- **Sample rate / định dạng:** **cần kiểm chứng** (đo file thật / xem `ds.features`).
- **License:** **cần kiểm chứng** — thẻ mirror không nêu rõ ràng. **Trước khi dùng thương mại phải xác minh license gốc của InfoRe.**

## Cảnh báo license

License của hai bộ này **không rõ ràng** trên mirror. Coi như **chỉ nghiên cứu** cho tới khi xác minh được
điều khoản gốc từ InfoRe Technology. Không dùng cho sản phẩm thương mại khi chưa rõ.

## Cách tải

```bash
# infore1 (25h đọc)
uv run python -c "
from datasets import load_dataset
ds = load_dataset('doof-ferb/infore1_25hours', split='train', streaming=True)
print(next(iter(ds)))
"

# infore2 (audiobooks)
uv run python -c "
from datasets import load_dataset
ds = load_dataset('doof-ferb/infore2_audiobooks', split='train', streaming=True)
print(next(iter(ds)))
"
```

## Convert sang manifest NeMo

Pattern giống VIVOS (ghi wav + sinh `.jsonl`). Lưu ý infore2 (audiobook) có **đoạn dài** →
nếu dùng để test/train có thể cần cắt đoạn (segment) cho phù hợp memory CPU.

## Lưu ý

- infore1 (25h) cỡ tương tự FOSD → một lựa chọn smoke-test thay thế, nhưng **license kém rõ ràng hơn FOSD** → ưu tiên FOSD.
- infore2 (audiobook) đoạn dài → ít hợp smoke-test nhanh, hợp làm dữ liệu train thêm.

## ✅ Tự kiểm nhanh

1. Vì sao nên ưu tiên FOSD hơn infore1 dù cỡ tương đương?
2. Trở ngại khi dùng infore2 để smoke-test là gì?

<details>
<summary>Đáp án</summary>

1. Vì **license của infore không rõ ràng**, còn FOSD là CC BY 4.0 minh bạch (cho thương mại).
2. infore2 là **audiobook đoạn dài** → tốn memory, không hợp smoke-test nhanh; cần cắt đoạn trước.

</details>
