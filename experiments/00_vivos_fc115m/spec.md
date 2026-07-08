# spec — 00_vivos_fc115m (hồi tố)

> Folder hồi tố cho run `vivos-fc115m-v1` (đã chạy TRƯỚC khi có khung này). Spec viết lại theo
> ý đồ thực tế lúc đó để khép vào chuẩn.

## Giả thuyết

- **H1:** Model NeMo English-only `stt_en_fastconformer_transducer_large` (115M, offline RNNT,
  đúng cỡ kiến trúc VPB) có thể fine-tune đổi-vocab sang tiếng Việt và **hội tụ** trên VIVOS.
- **H0:** Không hội tụ (như nemotron streaming — collapse-to-blank, WER 100%).

## Dự đoán

WER sau fine-tune **giảm rõ rệt** từ ~100% (đầu English) xuống vùng "có học được" (< 50%).

## Tiêu chí nghiệm thu

- WER sau < WER trước ít nhất vài chục điểm %.
- Round-trip GPU→CPU đáng tin: eval lại bằng model kéo về cho **cùng** WER.

## Scope / out-of-scope

- Trong scope: chứng minh luồng deploy Kaggle + recipe đổi-vocab chạy được trên model offline.
- Ngoài scope: tối ưu WER xuống mức cộng đồng/SOTA (để các experiment sau).

## Confounder (lúc đó CHƯA kiểm hết — bài học)

- **Chuẩn hoá nhãn vs tokenizer**: lúc này nhãn train để RAW → sinh `<unk>` ở chữ hoa (phát hiện
  sau, là root-cause "rớt ký tự đầu"). → động lực cho experiment `01`.
