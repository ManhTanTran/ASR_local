# ⚖️ _PROTOCOL — giao thức đánh giá & so sánh ASR

> Phỏng `comparison_protocol` của `numerai/lab_v2`, chế cho WER. Mục tiêu: phán quyết "cái nào tốt
> hơn" thành **nhị phân, không nước đôi**.

## 1. Eval chuẩn (cố định để mọi run so được)

- **Test set chuẩn:** `data/manifests/vivos_test.jsonl` (1000 câu) — KHÔNG đổi giữa các run.
- **Chuẩn hoá:** `asr_lab.common.metrics.normalize_vi` áp cho **cả ref lẫn hyp** (hạ thường, bỏ dấu
  câu, GIỮ dấu tiếng Việt). Đây cũng là chuẩn build tokenizer → tránh `<unk>`.
- **Chỉ số báo:** WER (chính) · CER (phụ, bắt lỗi cấp ký tự) · RTF (tốc độ).
- **Công cụ:** `uv run python -m asr_lab.eval.vivos --manifest ... --model ...` → ghi vào artifact.

## 2. Thang baseline 4 bậc (đọc kết quả theo bậc, không nhìn số trần)

| Bậc | Mốc | WER VIVOS |
| --- | --- | --- |
| 1. Sàn | English zero-shot (engine sống, chưa biết tiếng Việt) | ~100% |
| 2. Công sức tối thiểu | fine-tune đầu tiên (`vivos-fc115m-v1`) | 20,37% |
| 3. KPI cộng đồng | wav2vec2-base-vietnamese-250h | ~10,8% |
| 4. Sao bắc đẩu | ChunkFormer-CTC-large / PhoWhisper-large | 4,2 / 4,7% |

Nguồn mốc ngoài: `insight/external/01_vivos_sota_survey.md`.

## 3. Verdict 3 cổng (so candidate vs base)

Cài trong `asr_lab.analytics.verdict`:

1. **DẤU** — `WER_cand < WER_base`.
2. **TIN CẬY** — chênh đủ lớn so với nhiễu.
   - Chuẩn đầy đủ: **bootstrap CI95 trên WER mức-từng-câu** (CI không qua 0).
   - Proxy hiện tại: `|ΔWER| ≥ 1%` tuyệt đối (vì results.json mới lưu WER gộp). Muốn CI thật phải
     **dump hyp/ref từng câu** — TODO: thêm `--dump` cho `eval.vivos`.
3. **KHÔNG HỒI QUY RTF** — `RTF_cand ≤ ngân sách deploy` (mặc định 0,30; chậm hơn = mất gần real-time).

**Verdict tổng:** THẮNG (qua cả 3) · NGANG (ΔWER trong vùng nhiễu) · ĐÁNH ĐỔI (WER tốt, RTF hồi quy) · THUA.

## 4. Anti-leak / confounder (kỷ luật bắt buộc khi tuyên bố "thắng")

- **Speaker disjoint:** train/test KHÔNG chung người nói (VIVOS vốn tách — verify khi đổi data).
- **Không đụng test khi tune:** chọn siêu tham số + best-checkpoint theo **val**.
- **Chuẩn hoá nhất quán** vocab↔nhãn↔ref↔hyp (cổng OOV).
- **Cô lập biến (OFAT):** một experiment đổi MỘT đòn bẩy. ΔWER mơ hồ "do fix `<unk>` hay do thêm
  epoch?" = chưa cô lập → audit chưa đạt.
- **Kết quả tốt bất ngờ → tăng nghi ngờ**, soi positive/negative control trước khi tin.
