# spec — 01_vivos_fc115m_norm (PRE-REGISTERED)

> Chốt TRƯỚC khi run xong (run đang chạy: `vivos-fc115m-v2norm`). Không sửa sau khi thấy số.

## Giả thuyết

- **H1:** Sửa lệch chuẩn hoá tokenizer↔nhãn (normalize nhãn train khớp tokenizer) **loại bỏ `<unk>`
  ở chữ hoa** → khử lỗi "rớt ký tự đầu câu" → WER giảm rõ so với run `00` (20,37%).
- **H0:** WER không cải thiện có ý nghĩa (ΔWER trong vùng nhiễu < 1%).

## Dự đoán (số cụ thể)

- OOV rate trên nhãn train = **0%** (cổng `assert_no_oov` PASS). *(đã verify local trước khi phóng)*
- WER sau **< 11%** (chạm bậc 3 "KPI cộng đồng" của `_PROTOCOL`). Vùng kỳ vọng thực tế 8-15%.

## Đòn bẩy thay đổi (cô lập — OFAT)

So với `00`, đổi đúng các biến: **(a) normalize nhãn train/val** + **(b) cổng OOV** + epoch 40→50.
*Lưu ý confounder:* epoch 40→50 là biến thứ 2 → nếu thắng, không quy 100% cho fix `<unk>`; muốn cô
lập tuyệt đối cần một run 40-epoch + fix. Chấp nhận nhẹ vì mục tiêu vòng này là "có cải thiện rõ".

## Tiêu chí nghiệm thu (cứng)

1. Cổng OOV PASS (0%).
2. ΔWER vs `00` âm và **|ΔWER| ≥ 1%** (cổng tin-cậy proxy, `_PROTOCOL` §3).
3. Không hồi quy RTF (≤ 0,30).
4. Soi 5-10 cặp ref/hyp thật: **không còn `⁇`/`<unk>`** ở chữ hoa.

## Confounder cần loại

- Speaker overlap train/test (VIVOS vốn tách — giữ nguyên split).
- Best-checkpoint: run này vẫn eval epoch cuối (chưa chọn theo val) → để experiment `02`.
