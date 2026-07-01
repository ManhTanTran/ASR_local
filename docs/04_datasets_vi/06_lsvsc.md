# 06 — LSVSC (Large Scale Vietnamese Speech Corpus)

Bộ ~100h có **metadata phong phú** (giới tính, vùng miền, cảm xúc, nhóm tuổi, chủ đề).

---

## Số liệu

- **Loại:** đọc — gồm **truyện, sách nói, tin tức**. Giọng chủ yếu miền Bắc.
- **Giờ audio:** ~**100h**.
- **Số mẫu:** ~56.823 (45.5k train + 5.68k validation + 5.68k test).
- **Sample rate / định dạng:** trên HF lưu **parquet**. Sample rate gốc **cần kiểm chứng**.
- **Transcript:** có. Kèm metadata: gender, dialect, emotion, age group, topic.
- **License:** **CC BY 4.0** (theo thẻ HF mirror) — **cho phép thương mại**; mirror ghi là "unofficial mirror" nên nếu dùng cho sản phẩm nên truy nguồn gốc để đối chiếu.

## Cách tải

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('doof-ferb/LSVSC', split='test', streaming=True)
print(next(iter(ds)))
"
```

## Convert sang manifest NeMo

Giống pattern VIVOS — kiểm `ds.features` để biết tên cột audio/transcript, rồi ghi wav + sinh `.jsonl`.
Metadata (emotion, dialect, ...) có thể giữ thêm vào manifest nếu muốn phân tích WER theo nhóm:

```python
# ngoài audio_filepath/duration/text, NeMo cho phép thêm field tùy ý:
record = {
    "audio_filepath": path,
    "duration": dur,
    "text": text,
    "dialect": row.get("dialect"),     # field phụ — NeMo bỏ qua khi train, hữu ích để slice WER
    "emotion": row.get("emotion"),
}
```

## Lưu ý

- Metadata vùng miền/cảm xúc giúp **phân tích WER theo nhóm** (vd model yếu giọng miền nào).
- Vẫn là giọng đọc — không phải hội thoại điện thoại.

## ✅ Tự kiểm nhanh

1. LSVSC có gì khác biệt so với các bộ đọc khác?
2. Field phụ trong manifest dùng để làm gì?

<details>
<summary>Đáp án</summary>

1. Có **metadata phong phú** (giới tính, vùng miền, cảm xúc, tuổi, chủ đề).
2. Để **slice/phân tích WER theo nhóm** (vd so WER giọng Bắc vs Nam); NeMo bỏ qua field phụ khi train.

</details>
