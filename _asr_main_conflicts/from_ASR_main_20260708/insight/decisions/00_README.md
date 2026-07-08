# decisions — sổ quyết định

> Mỗi quyết định lớn = 1 file ngắn: bối cảnh → các lựa chọn → bằng chứng (link experiment/run_id) →
> chốt. Phân tích chi tiết để ở `docs/`; đây chỉ là bản ghi "vì sao chọn".

| # | Quyết định | Trạng thái |
| --- | --- | --- |
| [01](01_offline_not_streaming.md) | Dùng model **offline** (fastconformer) thay vì **streaming** (nemotron) để fine-tune đổi-vocab | Chốt |
| [02](02_why_fastconformer_115m.md) | Chọn fastconformer-transducer-large **115M** làm model nền chính | Chốt |
