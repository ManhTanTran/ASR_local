# Parakeet — các bộ pretrained weight

Khảo sát các checkpoint Parakeet có thật trên HuggingFace để chọn bản chạy smoke-test phiên âm trên CPU. Số liệu lấy từ model card HuggingFace `nvidia/parakeet-*` (tra 2026-06-19).

---

## Glossary

- **checkpoint** — bộ trọng số đã huấn luyện sẵn (file `.nemo` hoặc `model.safetensors`) để nạp vào model mà không cần train lại.
- **NGC** — NVIDIA GPU Cloud, kho model/container của NVIDIA; nhiều model Parakeet đồng thời nằm trên NGC và HuggingFace.
- **from_pretrained** — hàm NeMo `ASRModel.from_pretrained("id")` tự tải checkpoint từ HuggingFace/NGC về cache rồi dựng model.
- **license** — giấy phép sử dụng. `CC-BY-4.0` cho phép dùng thương mại nếu ghi nguồn.
- **decoder** — đầu giải mã: CTC (đơn giản, nhanh) · RNNT (transducer thuần) · TDT (transducer có duration, nhanh hơn RNNT) · TDT-CTC (lai, một encoder hai đầu).
- **XL / XXL** — cỡ FastConformer: XL ≈ 0.6B tham số (d_model 1024, 24 lớp), XXL ≈ 1.1B.
- **PnC** — Punctuation and Capitalization: model tự thêm dấu câu và viết hoa.

---

## Bảng các bộ weight

| id HF | params | decoder | ngôn ngữ | license | dung lượng | tải được CPU? |
| --- | --- | --- | --- | --- | --- | --- |
| `nvidia/parakeet-tdt-0.6b-v2` | ~0.6B | TDT | English | CC-BY-4.0 | ~2.4–2.5 GB (cần kiểm chứng) | Có |
| `nvidia/parakeet-tdt-0.6b-v3` | ~0.6B | TDT | 25 ngôn ngữ châu Âu (auto-detect) | CC-BY-4.0 | ~2.4–2.5 GB (cần kiểm chứng) | Có |
| `nvidia/parakeet-ctc-0.6b` | ~0.6B | CTC | English (chữ thường) | CC-BY-4.0 | ~2.4 GB (cần kiểm chứng) | Có |
| `nvidia/parakeet-rnnt-0.6b` | ~0.6B | RNNT | English (chữ thường) | CC-BY-4.0 | ~2.4 GB (cần kiểm chứng) | Có |
| `nvidia/parakeet-tdt_ctc-110m` | ~0.114B | TDT-CTC (lai, PnC) | English | CC-BY-4.0 | ~0.45 GB (cần kiểm chứng) | Có (nhẹ nhất) |
| `nvidia/parakeet-tdt_ctc-0.6b-ja` | ~0.6B | TDT-CTC (PnC) | Japanese | CC-BY-4.0 | ~2.4 GB (cần kiểm chứng) | Có |
| `nvidia/parakeet-tdt-1.1b` | ~1.1B | TDT | English | CC-BY-4.0 | ~4.5 GB (cần kiểm chứng) | Được nhưng nặng |
| `nvidia/parakeet-ctc-1.1b` | ~1.1B | CTC | English | CC-BY-4.0 | ~4.5 GB (cần kiểm chứng) | Được nhưng nặng |
| `nvidia/parakeet-rnnt-1.1b` | ~1.1B | RNNT | English | CC-BY-4.0 | ~4.5 GB (cần kiểm chứng) | Được nhưng nặng |
| `nvidia/parakeet-tdt_ctc-1.1b` | ~1.1B | TDT-CTC (PnC) | English | CC-BY-4.0 | ~4.5 GB (cần kiểm chứng) | Được nhưng nặng |

Ghi chú:
- Cột dung lượng là ước lượng theo số tham số (fp32 ≈ 4 byte/tham số: 0.6B ≈ 2.4 GB, 1.1B ≈ 4.5 GB). Model card không in con số GB chính xác → cần xem mục "Files and versions" trên HuggingFace để chốt.
- Bản `0.6b-v2` và bản cấu trúc đã mổ ở `01_structure.md` là cùng họ (FastConformer XL + TDT, joint_out 1030).
- `parakeet-tdt-0.6b-v3` giữ nguyên kiến trúc v2, chỉ mở rộng dữ liệu sang 25 ngôn ngữ + thêm auto language detection.

---

## Hỗ trợ tiếng Việt

Nói thẳng: **không có bản Parakeet nào hỗ trợ tiếng Việt.**

- `parakeet-tdt-0.6b-v2`, `parakeet-ctc-0.6b`, `parakeet-rnnt-0.6b`, các bản `1.1b`, `tdt_ctc-110m` — **English-only**.
- `parakeet-tdt_ctc-0.6b-ja` — **Japanese**, không phải tiếng Việt.
- `parakeet-tdt-0.6b-v3` — **25 ngôn ngữ châu Âu**: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian. **Tiếng Việt KHÔNG nằm trong danh sách này.**

Hệ quả cho bài toán callbot tiếng Việt (giống VPB cũ): Parakeet chỉ dùng được như mục đích học kiến trúc/đo CPU, **không phiên âm được tiếng Việt** nếu chưa fine-tune lại. Muốn tiếng Việt ngay, xem họ Nemotron (`02_nemotron/03_pretrained_weights.md`) — bản `nemotron-3.5-asr-streaming-0.6b` có tiếng Việt sẵn.

---

## Lệnh tải + smoke test

Repo đã có sẵn `src/asr_lab/eval/smoke.py`. Cú pháp:

```bash
uv run python -m asr_lab.eval.smoke <duong_dan.wav> [--model <id_HF>]
```

Yêu cầu file wav: **mono, 16kHz** (resample trước nếu khác sample_rate).

Tải + chạy bản nhẹ nhất để smoke-test CPU (English):

```bash
# Bản 110M lai TDT-CTC, nhẹ nhất, có dấu câu
uv run python -m asr_lab.eval.smoke mau.wav --model nvidia/parakeet-tdt_ctc-110m
```

Tải bản 0.6b-v2 (English, đúng họ đã mổ kiến trúc):

```bash
uv run python -m asr_lab.eval.smoke mau.wav --model nvidia/parakeet-tdt-0.6b-v2
```

Tải bản đa ngôn ngữ v3 (auto-detect 25 ngôn ngữ châu Âu, KHÔNG có tiếng Việt):

```bash
uv run python -m asr_lab.eval.smoke mau.wav --model nvidia/parakeet-tdt-0.6b-v3
```

Bên trong `from_pretrained` tự tải checkpoint về cache HuggingFace (`~/.cache/huggingface`) lần đầu. Tải thủ công nếu muốn (không bắt buộc):

```bash
huggingface-cli download nvidia/parakeet-tdt_ctc-110m
```

---

## Khuyến nghị cho lab CPU

Mục tiêu là smoke-test luồng tải + phiên âm chạy thông trên CPU, không phải đo chất lượng.

1. **Bản test trước tiên: `nvidia/parakeet-tdt_ctc-110m`** — nhẹ nhất (~0.114B, tải nhanh, RAM thấp), kiểm tra luồng nhanh nhất.
2. **Bản đại diện họ: `nvidia/parakeet-tdt-0.6b-v2`** — đúng kiến trúc đã mổ ở `01_structure.md` (TDT, joint_out 1030). Chạy được trên CPU nhưng chậm hơn 110M.
3. **Nếu cần thử đa ngôn ngữ:** `nvidia/parakeet-tdt-0.6b-v3` (vẫn không có tiếng Việt).
4. **Bỏ qua cho smoke-test:** tất cả bản `1.1b` — nặng (~4.5 GB, RAM cao), CPU chạy chậm, không cần thiết để xác nhận luồng.

Lưu ý quan trọng: nếu mục tiêu cuối là **tiếng Việt**, Parakeet không phải lựa chọn (không có checkpoint tiếng Việt). Dùng để học/đo CPU thì được.

---

## ✅ Tự kiểm nhanh

1. Bản Parakeet nào nhẹ nhất để smoke-test CPU, và vì sao chọn nó?

<details><summary>Đáp án</summary>

`nvidia/parakeet-tdt_ctc-110m` (~0.114B tham số). Nhẹ nhất nên tải nhanh, RAM thấp, xác nhận luồng tải + phiên âm chạy thông nhanh nhất trước khi thử bản 0.6B.
</details>

2. Parakeet V3 hỗ trợ 25 ngôn ngữ — trong đó có tiếng Việt không?

<details><summary>Đáp án</summary>

Không. V3 là 25 ngôn ngữ châu Âu (Bulgarian, Croatian, ... Russian, Ukrainian). Tiếng Việt không nằm trong danh sách. Muốn tiếng Việt phải xem Nemotron-3.5.
</details>

3. License của các bản Parakeet là gì, có dùng thương mại được không?

<details><summary>Đáp án</summary>

CC-BY-4.0. Cho phép dùng thương mại nếu ghi nguồn (attribution).
</details>
