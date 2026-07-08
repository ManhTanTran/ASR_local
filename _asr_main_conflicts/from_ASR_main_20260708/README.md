# nvidia_asr_nemo

Lab cá nhân để tìm hiểu NeMo ASR (NVIDIA) ở mức **đọc hiểu kiến trúc + chạy thử suy luận**, không huấn luyện.

## Mục tiêu

- Chạy suy luận thử (smoke test) với model pretrained để thông luồng.
- Soi kiến trúc và đếm số tham số bằng code thật (`print(model)`, `summarize()`).
- Trace code NeMo qua ctrl-click (NeMo là Python thuần, đọc thẳng trong `site-packages`).

## Ràng buộc đã chốt

- **Chạy CPU** — không cần GPU; torch lấy bản CPU (cấu hình sẵn trong `pyproject.toml`).
- **Quản môi trường bằng `uv`** — chuẩn mới, nhanh; deps khai báo trong `pyproject.toml`, khóa trong `uv.lock`.
- **Dùng NeMo như thư viện** — code repo này độc lập, nâng cấp NeMo = đổi version, không sửa lõi NeMo in-tree.
- **Pin theo release ổn định** (hiện tại v2.7.3), không bám `main`.

## Cấu trúc

```
pyproject.toml   Khai báo deps + index torch CPU (uv)
docs/            Tài liệu (xem docs/00_INDEX.md)
  01_setup_cpu.md       Dựng môi trường CPU bằng uv
  02_asr_components/    Kiến thức ASR dùng chung (clone từ stt_nvidia_nemo)
  03_models/           Phân tích model cụ thể (Parakeet, Nemotron)
src/             Script: soi kiến trúc, smoke test phiên âm
notebooks/       Notebook explore cấu hình model
```

Bắt đầu: đọc `docs/00_INDEX.md` (điểm vào toàn bộ tài liệu).
