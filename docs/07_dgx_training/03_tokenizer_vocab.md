# 07.03 — Quyết định tokenizer + tập từ điển (vocab)

Chốt cách dựng tokenizer cho corpus đa-nguồn ~1.660h (khác hẳn giai đoạn VIVOS-only 15h).
Bám code hiện có: `finetune_vivos.build_vi_tokenizer` (SentencePiece BPE) + `metrics.normalize_vi`.

---

## Glossary

- **BPE (Byte-Pair Encoding):** gộp cặp ký tự/subword hay đi cùng thành 1 token → giảm độ dài chuỗi.
- **char_coverage:** tỉ lệ ký tự corpus mà SentencePiece cam kết phủ; 1.0 = phủ mọi ký tự thấy được.
- **NFC/NFD:** chuẩn Unicode. Tiếng Việt "ệ" có thể là 1 codepoint (NFC) hoặc "e"+2 dấu tổ hợp (NFD). Không thống nhất → cùng 1 từ ra 2 chuỗi byte khác nhau → vocab phình + OOV giả.
- **OOV (<unk></unk>):** ký tự/từ ngoài vocab → token `<unk>` → model học phát `<unk>`, WER phồng ngầm.

---

## Hiện trạng (đọc từ code)

- `build_vi_tokenizer`: SentencePiece **BPE**, `vocab_size=512`, `char_coverage=1.0`, `unk_id=0`, không bos/eos/pad. Train **chỉ từ transcript VIVOS**.
- `normalize_vi`: `lower()` + bỏ ký tự không phải `[\w\s]` (giữ chữ có dấu + **số**), gộp khoảng trắng.
  **KHÔNG có bước NFC** — đây là lỗ hổng phải vá trước khi gộp nhiều nguồn.
- Cổng `assert_no_oov` / `assert_charset_ok` chặn nhãn sinh `<unk>` trước khi tốn GPU (giữ nguyên, tốt).

---

## Quyết định

### QĐ-1 — Vá `normalize_vi`: thêm NFC (bắt buộc, ưu tiên cao nhất)

Nhiều mirror HF trộn NFC/NFD. Không chuẩn hóa → tokenizer thấy "ệ" hai kiểu, vocab loạn, cổng OOV báo giả.

```python
import unicodedata
def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)   # THÊM: thống nhất codepoint tiếng Việt
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
```

Ảnh hưởng: mọi manifest + tokenizer + eval dùng chung 1 chuẩn. Chạy lại tokenizer sau khi vá.

### QĐ-2 — Tokenizer build từ TOÀN corpus đã gộp, không phải VIVOS-only

Corpus mới có loanword (facebook/wifi) chứa **f/j/w/z** mà VIVOS không có.
Nếu giữ tokenizer VIVOS 512, CV/Bud500 bị drop clip loanword (như exp 04 đã phải lọc).
Giải: **train tokenizer trên text đã normalize của cả S1+S2+S3** để phủ đủ charset tiếng Việt thực tế.

### QĐ-3 — Vocab size = **1024** (GIỮ NGUYÊN — ckpt hiện đã 1024)

> Đính chính (inspect 2026-07-01): ckpt `vivos-fc115m-v2norm` thực tế đã có vocab **1024** (default code 512 đã bị override). Đây là GIỮ cỡ, chỉ rebuild lại từ toàn corpus (QĐ-2), không phải tăng từ 512.

- 1024 cân bằng độ dài chuỗi ↔ phủ charset; hợp mức ~1.660h.
- Không nhảy 2048 vội: vocab lớn cần nhiều data/lớp, lợi giảm dần. 1024 là mốc an toàn, đo lại sau.

### QĐ-4 — Charset tường minh thay vì `char_coverage=1.0` mù

`1.0` phủ cả rác (ký tự lạ, emoji sót, chữ Hán lẫn). Thay bằng: **định nghĩa whitelist**, làm sạch text NGOÀI whitelist trước, rồi train tokenizer coverage 1.0 trên text đã sạch.

- Whitelist = 29 chữ cái tiếng Việt (đủ dấu thanh + â/ă/ê/ô/ơ/ư/đ) + `a-z` (loanword) + **số 0-9** + khoảng trắng.
- Clip có >1% ký tự ngoài whitelist → drop (log lại). Ký tự lẻ ngoài whitelist trong clip tốt → thay space.

### QĐ-5 — Số (digits): v1 GIỮ NGUYÊN dạng số, verbalize để sau

- Các nguồn trộn "2020" và "hai nghìn hai mươi". Verbalize (đọc số thành chữ) chuẩn hơn nhưng tốn công + rủi ro.
- **v1:** giữ số như nhãn gốc (đã trong whitelist), chấp nhận lệch nhỏ khi đo WER.
- **Cải tiến sau:** thêm bước ITN/verbalize thống nhất train+eval (ghi tech-debt).

### QĐ-6 — Base model & số phận decoder cũ

- Corpus đã lớn → **rebuild decoder+joint theo tokenizer 1024 mới** qua `change_vocabulary()`. Decoder học lại nhanh; không tiếc decoder VIVOS-512.
- Encoder: khởi từ **checkpoint tốt nhất sẵn có** (ckpt 115M đã hội tụ offline của lab) để không mất biểu diễn âm học. Cân nhắc song song một nhánh từ NGC VI-capable (`nvidia/parakeet-*`) ở nhóm model B.

---

## Việc code (khi corpus về)

1. Vá `normalize_vi` (QĐ-1) + test lại cổng OOV trên manifest cũ.
2. Hàm `build_vi_tokenizer` nhận **nhiều manifest** (list) để gom text toàn corpus; thêm bước lọc whitelist (QĐ-4).
3. Tham số hóa `vocab_size` mặc định 1024; giữ `unk_id=0`, không bos/eos/pad (chuẩn ASR-BPE — không đổi).
4. Sinh `tokenizer_vi_1024/` + `vocab.txt` để soi; log charset + phân bố độ dài token.

## ✅ Tự kiểm nhanh

1. Vì sao NFC bắt buộc? 2. Vì sao rebuild tokenizer thay vì giữ 512 VIVOS? 3. Vocab bao nhiêu, vì sao không 2048?

<details><summary>Đáp án</summary>
1. Trộn NFC/NFD làm cùng 1 từ ra 2 chuỗi byte → vocab loạn + OOV giả. 2. Corpus mới có loanword f/j/w/z + nhiều giờ → 512 quá nhỏ, phải phủ charset thật. 3. 1024 — cân bằng độ dài chuỗi ↔ phủ; 2048 lợi giảm dần, cần nhiều data hơn.
</details>
