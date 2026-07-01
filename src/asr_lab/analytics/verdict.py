"""3 cổng verdict so 2 run ASR (phỏng comparison_protocol numerai lab_v2, chế cho WER).

Cổng:
  1. DẤU      — WER candidate < WER base (cải thiện đúng chiều).
  2. TIN CẬY  — chênh WER đủ lớn so với nhiễu. CHUẨN: bootstrap CI95 trên WER mức-từng-câu.
                Hiện results.json chỉ lưu WER gộp -> dùng ngưỡng tối thiểu |ΔWER| >= MIN_DELTA
                làm proxy; muốn CI thật phải dump hyp/ref từng câu (xem ghi chú dưới).
  3. KHÔNG HỒI QUY — RTF candidate không vượt ngân sách deploy (chậm hơn = hỏng real-time).

Verdict tổng: THẮNG (qua cả 3) · NGANG (ΔWER trong vùng nhiễu) · ĐÁNH ĐỔI (WER tốt nhưng RTF hồi quy).
"""
from __future__ import annotations

MIN_DELTA = 0.01      # 1% WER tuyệt đối — proxy "khác biệt thật" khi chưa có CI từng câu
RTF_BUDGET = 0.30     # ngưỡng RTF deploy mặc định (chậm hơn = không kịp gần real-time)


def verdict(wer_base: float, wer_cand: float,
            rtf_base: float | None = None, rtf_cand: float | None = None,
            min_delta: float = MIN_DELTA, rtf_budget: float = RTF_BUDGET) -> dict:
    d_wer = wer_cand - wer_base                      # âm = tốt
    gate_sign = d_wer < 0
    gate_conf = abs(d_wer) >= min_delta
    gate_rtf = (rtf_cand is None) or (rtf_cand <= rtf_budget)

    if not gate_conf:
        label = "NGANG (ΔWER trong vùng nhiễu)"
    elif gate_sign and gate_rtf:
        label = "THẮNG"
    elif gate_sign and not gate_rtf:
        label = "ĐÁNH ĐỔI (WER tốt nhưng RTF hồi quy)"
    else:
        label = "THUA"
    return {
        "d_wer": round(d_wer, 4), "d_wer_pct": round(d_wer * 100, 2),
        "gate_sign": gate_sign, "gate_conf": gate_conf, "gate_rtf": gate_rtf,
        "rtf_cand": rtf_cand, "verdict": label,
        "note_ci": "proxy ngưỡng |ΔWER|>=%.0f%%; CI95 từng câu cần dump hyp/ref (eval --dump)"
                   % (min_delta * 100),
    }
