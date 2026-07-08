# RESULT — 04_add_commonvoice

**Artifact:** `artifacts/runs/vivos-cv/` · ✅ **COMPLETE** (25/25 epoch, 20.625 step, ~2,45h GPU P100).
Resume `vivos-fc115m-v2norm.nemo` + gộp Common Voice VI (n_train=13.187). Chạy trên acc2 (maymilo).

> ⚠️ **Mục đích run này = THÔNG LUỒNG** resume-từ-.nemo + gộp dataset ngoài (cơ chế deploy), KHÔNG phải
> training tối ưu lâu dài. Số 2×2 đủ để xác nhận luồng đúng + chiều hướng H1; muốn đẩy CV-WER thật sự
> thì cần nhiều giờ CV + tuning (xem Hướng kế).

## Số thật (bảng 2×2 — đo trong MỘT run)

| | VIVOS test (1000) | CV test (1225) |
| --- | --- | --- |
| **TRƯỚC** (ckpt v2norm) | 11,93% | 42,95% |
| **SAU** (resume + gộp CV, 25ep) | **11,39%** | **36,85%** |
| Δ | −0,54% (tốt nhẹ) | **−6,10%** (giảm rõ) |

RTF sau: VIVOS 0,057 · CV 0,041 (không hồi quy).

**Round-trip xác nhận:** ô TRƯỚC-VIVOS đo lại = **11,93%**, khớp tuyệt đối baseline `v2norm` đã biết
→ pipeline eval + data CV nhất quán, không lệch harness.

## Verdict (3 cổng nghiệm thu spec)

- [x] **Cổng chính H1** — CV test SAU < TRƯỚC, |Δ| ≥ 5%: **−6,10%** (42,95→36,85) ✔
- [x] **Cổng bảo vệ H0a** — VIVOS SAU không xấu quá 2% so 11,93%: **cải thiện 0,54%** (không lệch domain) ✔
- [x] **Sạch `<unk>`/`⁇`**: log không có token lỗi; soi ref/hyp CV+VIVOS lỗi chỉ là nhầm âm
  ("quân à"→"quân đà", rớt từ cuối) ✔

→ **VERDICT: THẮNG (cổng PASS), nhưng mức lợi KHIÊM TỐN hơn kỳ vọng.**

## Đối chiếu dự đoán (pre-registered)

| ô | dự đoán spec | thực đo | khớp? |
| --- | --- | --- | --- |
| VIVOS SAU | 10,5–12,5% | 11,39% | ✔ trong khoảng |
| CV TRƯỚC | 35–55% | 42,95% | ✔ trong khoảng |
| CV SAU | **18–30%** | **36,85%** | ✗ giảm đúng chiều nhưng **CAO hơn dự đoán** |

## Insight (trung thực)

- **H1 đúng về chiều**: thêm domain CV kéo CV-WER xuống rõ (−6,1%) mà KHÔNG hại VIVOS (còn tốt nhẹ).
  Đây là bằng chứng "data > model" có thật — nhưng **không mạnh như kỳ vọng**.
- **H0b đúng một phần**: CV chỉ ~3,4h audio so VIVOS ~15h → lượng data CV quá ít để model "thuộc" domain
  mic-tại-nhà; CV-WER còn 36,85% (cao). Muốn xuống vùng 18–30% cần **nhiều giờ CV hơn** (hoặc augment).
- Lỗi CV còn lại không phải `<unk>` mà là **nhầm âm + rớt từ cuối câu** — giới hạn âm học/độ dài, không
  phải vocab. Tokenizer VIVOS đủ phủ (cổng charset PASS, OOV~0).

## Hướng kế (đề xuất)

1. **Tăng giờ CV**: gộp thêm mirror CV lớn hơn (CV 17/22 nếu lấy được parquet) hoặc VLSP → kiểm H0b.
2. **SpecAugment** (`03`): chống overfit khi data lệch tỉ lệ VIVOS:CV ~6:1.
3. **Best-checkpoint theo val** (`02`): có thể nhặt thêm ~0,5–1% rẻ.

## Tái lập (lệnh đã chạy)

```bash
# upload .nemo resume thành dataset (cùng account, tránh kernel cross-account private)
uv run python -m asr_lab.deploy.kaggle upload-data --account acc2 \
    --file artifacts/runs/vivos-fc115m-v2norm/nemotron_vivos_ft.nemo --as asr-v2norm-nemo
uv run python -m asr_lab.deploy.kaggle build --account acc2
uv run python -m asr_lab.deploy.kaggle push --account acc2 --gpu --as asr-cv-fc115m \
    --module asr_lab.train.continue_vi \
    --script-args "--resume-from /kaggle/input --run-id vivos-cv --epochs 25 --batch 16 --precision 32" \
    --input-dataset asr-v2norm-nemo
uv run python -m asr_lab.deploy.kaggle pull --account acc2 --kernel asr-cv-fc115m
```

> Ghi chú deploy: kyhoolee hết quota GPU tuần → chạy acc2. Bug đã fix giữa chừng: `resolve_nemo` dùng
> `p.is_file()` (không `p.exists()`) vì `/kaggle/input` là THƯ MỤC — phải rglob `.nemo` bên trong.
