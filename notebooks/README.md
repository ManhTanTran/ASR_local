# notebooks

Notebook explore cấu hình model NeMo ASR (chạy CPU).

## Cấu trúc

```
notebooks/
├── README.md
├── utils/
│   └── inspect_helpers.py   # hàm load model + in params/encoder/decoder/tokenizer gọn
└── 01_explore_model_config.ipynb   # so cấu hình small Conformer / Parakeet-TDT / Nemotron
```

## Chạy

- **VSCode (khuyến nghị):** mở file `.ipynb`, chọn kernel là `.venv` của repo (Python 3.12). `ipykernel` đã cài sẵn.
- **Jupyter Lab:** `uv run jupyter lab` (cài thêm nếu cần: `uv add jupyterlab`).

## Lưu ý

- Tất cả chạy CPU (`map_location='cpu'`), không cần GPU.
- Model `parakeet` / `nemotron` mỗi cái ~2.4GB, tải lần đầu rồi cache ở `~/.cache/huggingface`.
- Notebook import helper từ `utils/`; đã test các hàm trên model nhỏ + Parakeet (chạy được).

## Helper chính (`utils/inspect_helpers.py`)

- `load(name)` — tải model pretrained về CPU.
- `print_overview(model, name)` — in params theo khối, encoder, kiểu giải mã (tự nhận diện CTC/RNNT/TDT), tokenizer.
- `summary_row(model, name)` — một dòng cho bảng so sánh (pandas).
- `config_yaml(model, section)` — in YAML một phần config (encoder/decoder/joint/decoding).
