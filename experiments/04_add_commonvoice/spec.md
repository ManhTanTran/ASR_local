# spec — 04_add_commonvoice (PRE-REGISTERED)

> Chốt TRƯỚC khi chạy GPU. Không sửa sau khi thấy số. Phương án **A**: giữ nguyên tokenizer VIVOS,
> resume từ ckpt `vivos-fc115m-v2norm` (.nemo), gộp thêm Common Voice VI vào train.

## Bối cảnh + nguồn data (đã verify local)

- Mirror parquet (datasets v5 OK, không gated): **`tsdocode/common_voice_13_0_vi_pseudo_labelled`**.
  Cột transcript THẬT = `sentence` (bỏ `whisper_transcript` = nhãn-giả). Audio mp3 nhúng (bytes).
- Số dòng đã đếm: **train 2.462 · validation 392 · test 1.225** (~5s/clip → train ~3,4h audio).
- Độ sạch text (sau `normalize_vi`, đo local): **0% câu có chữ số**, chỉ ~1 câu train chứa f/j/w/z.
- **Cổng OOV thật (CV ↔ tokenizer VIVOS), verify local:** train **0,0085%** (2/23524 — đúng câu
  "facebook"), test 0,0425% (loanword firefox/zoom/word chứa f/j/w/z VIVOS không phủ).

## Giả thuyết

- **H1:** Model resume + gộp CV sẽ **giảm mạnh WER trên CV test** (model hiện chưa từng thấy domain
  CV mic-tại-nhà), đồng thời **giữ hoặc cải thiện nhẹ WER VIVOS test** (thêm ~21% data + đa dạng).
- **H0a (rủi ro):** CV kéo lệch domain → WER VIVOS test **xấu đi** (domain shift > lợi ích data).
- **H0b:** CV quá ít (~3,4h so với VIVOS 15h) → hầu như không đổi gì trên cả hai test.

## Dự đoán (số cụ thể — sẽ đối chiếu)

| ckpt | VIVOS test (1000) | CV test (1225) |
| --- | --- | --- |
| `v2norm` (hiện tại, TRƯỚC) | 11,93% (đã biết) | **dự đoán 35-55%** (chưa thấy CV) |
| `v2norm` + CV (SAU) | **dự đoán 10,5-12,5%** (đi ngang ±) | **dự đoán 18-30%** (giảm rõ) |

> Lý do CV test vẫn cao kể cả sau: CV mic-tạp + loanword f/j/w/z tokenizer VIVOS không spell được.

## Tiền xử lý CV (chi tiết — để DÙNG CHUNG tokenizer VIVOS)

1. `normalize_vi` DUY NHẤT (NFC → lower → bỏ `[^\w\s]` → gộp space) ở mọi ranh giới.
2. Audio: bytes mp3 → `sf.read`/fallback `librosa` → 16kHz mono → wav (như loader VIVOS).
3. **TRAIN:** lọc theo charset tokenizer VIVOS — drop clip có ký tự ngoài vocab (≈1-3 clip). Bảo đảm
   OOV = 0 *by construction*. Cổng `assert_no_oov` xác nhận lại trước khi train.
4. **TEST (CV + VIVOS):** KHÔNG lọc — đo thật, kể cả loanword model spell sai (phản ánh giới hạn thật).

## Đòn bẩy thay đổi (cô lập)

So với `01`: đổi đúng **(a) resume từ .nemo thay vì from_pretrained**, **(b) train set = VIVOS + CV**.
Giữ nguyên: tokenizer (1024), batch 16, lr, precision 32, normalize_vi.

## Chi phí ước lượng (deliverable #3)

- **Train:** combined ~13.880 clip → ~868 step/epoch. Đo thật từ `v2`: ~0,44s/step.
  Resume cần ít epoch hơn (decoder đã chín) → mục tiêu **20-30 epoch** ⇒ ~17k-26k step ⇒ **~2,1-3,2h**
  GPU (P100). `max_minutes` cap để luôn kịp eval+save trong khung 8h Kaggle.
- **Deploy:** code-dataset rebuild vài giây; **KHÔNG upload audio** (kernel tự tải CV+VIVOS từ HF);
  `.nemo` resume **tham chiếu thẳng output kernel `asr-ft-fc115m-v2norm`** (kernel_sources) — không upload lại.
- **Pull về:** ~457MB .nemo mới (vài phút). Tổng wall-clock ~3-4h, quota Kaggle free.

## Tiêu chí nghiệm thu (cứng)

1. Cổng OOV trên train gộp PASS (≈0%).
2. Đo đủ **4 ô** bảng 2×2 (VIVOS+CV × trước+sau) trong **một** run.
3. CV test SAU < CV test TRƯỚC, |Δ| ≥ 5% (kỳ vọng giảm mạnh — cổng chính của H1).
4. VIVOS test SAU không xấu quá 2% so với 11,93% (cổng bảo vệ — bắt H0a domain shift).
5. Soi 8 cặp ref/hyp mỗi test set: không `⁇`/`<unk>` ở chữ hoa.

## Confounder cần loại

- Speaker overlap CV train/test: CV vốn tách theo split — giữ nguyên 3 split gốc.
- "pseudo_labelled": dùng `sentence` (gốc người đọc), KHÔNG dùng `whisper_transcript`.
- Best-checkpoint vẫn eval epoch cuối (chưa chọn theo val) → để experiment `02`.
