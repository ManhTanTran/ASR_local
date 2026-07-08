# decision 01 — offline thay vì streaming cho fine-tune đổi-vocab

**Bối cảnh:** Kỳ ban đầu chọn `nemotron-speech-streaming-en-0.6b`. Fine-tune đổi-vocab sang tiếng Việt.

**Lựa chọn:** (A) nemotron streaming 0.6B · (B) fastconformer offline 115M.

**Bằng chứng:**
- A (`artifacts/runs/verify`): loss kẹt mức ngẫu nhiên (~100), **collapse-to-blank**, WER giữ 100% dù
  full/freeze, dù precision khác nhau.
- B (`artifacts/runs/verify2` → `vivos-fc115m-v1`): cùng recipe, loss giảm đều, WER 100%→81%→**20,37%**.

**Chốt:** Recipe đổi-vocab hợp model **offline** (`att_context_style: regular`). Model **streaming**
(`chunked_limited` + conv `causal`) cần cách fine-tune riêng (giữ vocab + langid với bản đa ngữ 3.5).

**Chi tiết kỹ thuật:** `docs/03_models/02_nemotron/04_vi_finetune_difficulty.md`.
