# 00 — INDEX tài liệu nvidia_asr_nemo

Điểm vào (entry point) cho toàn bộ tài liệu của lab. Đọc theo thứ tự dưới.

> 🧭 **Bắt đầu session:** đọc [`../STATE.md`](../STATE.md) (đang ở đâu / việc kế). Quy trình nghiên
> cứu lũy tiến (phỏng `numerai/lab_v2`): [`../experiments/`](../experiments/_RUN_STANDARD.md)
> (1-exp-1-folder, [_PROTOCOL](../experiments/_PROTOCOL.md), [_SCOREBOARD](../experiments/_SCOREBOARD.md))
> · [`../insight/`](../insight/README.md) (concepts/proposals/decisions/external).
> Phân vai: **`docs/`** = kiến thức ổn định; **`experiments/`+`insight/`** = nhật ký nghiên cứu động.

---

## Cấu trúc

| Mục | Nội dung |
| --- | --- |
| [01_setup_cpu.md](01_setup_cpu.md) | Dựng môi trường CPU bằng `uv`, cách chạy + trace code |
| [02_asr_components/](02_asr_components/00_INDEX.md) | **Kiến thức ASR dùng chung** giữa các model (tokenizer, mel, Conformer, CTC/RNNT/AED, WER) |
| [03_models/](03_models/00_INDEX.md) | Phân tích từng họ model cụ thể (Parakeet, Nemotron) |
| [04_datasets_vi/](04_datasets_vi/00_INDEX.md) | **Dataset tiếng Việt dùng chung** để test/đo WER/fine-tune (VIVOS, Common Voice, FOSD, VietBud500, ...) |
| [05_datasets_en/](05_datasets_en/00_INDEX.md) | **Dataset tiếng Anh dùng chung** để smoke-test luồng (sample wav NeMo, AN4, LibriSpeech, ...) |
| [06_benchmarks/](06_benchmarks/README.md) | Đo RTF + WER + **fine-tune tiếng Việt**: [00 thông luồng](06_benchmarks/00_first_smoke_bench.md) · [01 test khó](06_benchmarks/01_hard_testsets_matrix.md) · [02 fine-tune VIVOS (100%→20%)](06_benchmarks/02_vivos_finetune.md) |
| [../notebooks/](../notebooks/README.md) | Notebook explore cấu hình model thực tế |

---

## Cấu trúc code (`src/asr_lab/` — package, chạy `python -m asr_lab.<module>`)

Tổ chức theo tầng để trace nhanh từng thành phần (data / model / eval / train / deploy):

| Module | File | Vai trò |
| --- | --- | --- |
| `asr_lab.common` | `metrics.py` · `models.py` | Dùng chung: `wer`, `normalize_en/vi`, `extract_text` · danh sách MODELS |
| `asr_lab.data` | `vivos.py` · `hf_testset.py` · `librispeech.py` | Tải data → manifest NeMo (`dump_split`...) |
| `asr_lab.model` | `inspect_arch.py` | Soi kiến trúc + đếm tham số 1 model |
| `asr_lab.eval` | `bench.py` · `sweep.py` · `vivos.py` · `smoke.py` | Đo RTF+WER (Anh: bench/sweep · Việt: vivos) · smoke 1 wav |
| `asr_lab.train` | `finetune_vivos.py` | Fine-tune đổi-vocab sang tiếng Việt (có cổng `assert_no_oov`) |
| `asr_lab.deploy` | `kaggle.py` | Adapter deploy Kaggle GPU (build/push/poll/pull) |
| `asr_lab.registry` | `build_scoreboard.py` | Sinh `experiments/_SCOREBOARD.md` từ `artifacts/runs/*/results.json` |
| `asr_lab.analytics` | `report.py` · `compare.py` · `verdict.py` | Đọc artifact → report 1 run / so 2 run / verdict 3 cổng (không train lại) |

Cài editable qua `uv sync` (pyproject có `[build-system]` + `[tool.setuptools.packages.find]`).
Data nặng ở `data/` (gitignore, kéo bằng module `asr_lab.data.*`); artifact ở `artifacts/` (gitignore).

## Định hướng đọc

- **Mới bắt đầu** — đọc `01_setup_cpu.md` để dựng môi trường, rồi chạy notebook explore.
- **Ôn kiến thức nền ASR** — vào `02_asr_components/` (các khái niệm chung, không gắn model cụ thể).
- **Tìm hiểu model cụ thể** — vào `03_models/`.

> `02_asr_components/` được clone từ cụm deep-dive bên `stt_nvidia_nemo` vì phần lớn thành phần ASR (encoder Conformer, tokenizer, các kiểu giải mã) dùng chung giữa Fast-Conformer cũ và Parakeet/Nemotron mới. Khi phân tích model mới, tham chiếu lại các file ở đây thay vì lặp lại.
