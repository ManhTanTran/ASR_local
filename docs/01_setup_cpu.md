# 00 — Setup môi trường CPU bằng uv

Hướng dẫn dựng môi trường để chạy NeMo ASR trên CPU và soi kiến trúc model.
Quản môi trường bằng `uv`; torch lấy bản CPU (đã cấu hình trong `pyproject.toml`).

---

## 1. Yêu cầu

- `uv` đã cài (kiểm tra: `uv --version`).
- Python ≥ 3.12 (uv tự tải nếu thiếu).
- Không cần GPU.

---

## 2. Dựng môi trường

```bash
cd ~/work/startup/_0_iruka/_1_backend/nvidia_asr_nemo

# Ghim Python 3.12 cho repo (uv tự tải bản phù hợp nếu máy chưa có)
uv python pin 3.12

# Đồng bộ: tạo .venv + cài deps theo pyproject + khóa uv.lock
# torch/torchaudio tự lấy từ index CPU, không kéo bản CUDA
uv sync
```

- **Lưu ý mạng yếu** — `uv sync` lần đầu tải torch CPU + NeMo + phụ thuộc (vài trăm MB). Nên chạy khi mạng ổn; uv cache lại nên lần sau nhanh.

---

## 3. Chạy

```bash
# Soi kiến trúc + đếm tham số (không cần audio)
uv run python -m asr_lab.model.inspect_arch

# Smoke test phiên âm 1 file wav (mono 16kHz)
uv run python -m asr_lab.eval.smoke path/to/sample.wav
```

- `uv run` tự kích hoạt `.venv`, không cần `activate` thủ công.

---

## 4. Thêm/bớt thư viện về sau

```bash
uv add <package>        # thêm, tự cập nhật pyproject + uv.lock
uv remove <package>     # bớt
uv sync --upgrade       # nâng cấp trong khoảng version đã khai báo
```

- Nâng cấp NeMo lên release mới: sửa version trong `pyproject.toml` rồi `uv sync`, sau đó chạy lại script kiểm tra.

---

## 5. Trace code NeMo (ctrl-click)

- Sau `uv sync`, source NeMo nằm trong `.venv/lib/python3.12/site-packages/nemo/...` (Python thuần).
- Trỏ interpreter của IDE vào `.venv` này → ctrl-click "Go to Definition" nhảy thẳng vào source NeMo để đọc.

---

## ✅ Tự kiểm nhanh

1. Vì sao torch không kéo bản CUDA dù không khai báo gì lúc chạy lệnh?

<details><summary>Đáp án</summary>

Vì `pyproject.toml` đã khai báo `[tool.uv.sources]` buộc torch/torchaudio lấy từ index `pytorch-cpu`, nên `uv sync` luôn lấy bản CPU.
</details>

2. Nâng cấp NeMo đúng cách gồm những bước nào?

<details><summary>Đáp án</summary>

Sửa version trong `pyproject.toml`, chạy `uv sync`, rồi chạy lại script kiểm tra để xác nhận không vỡ.
</details>
