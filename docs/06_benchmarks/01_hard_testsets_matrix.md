# Ma trận WER trên test set KHÓ (tiếng Anh, CPU)

Đo trên các test set khó hơn LibriSpeech clean để **phân định model** — bộ clean quá dễ nên
WER mọi model đều ~2-6%, không tách bậc. Các bộ khó cho WER 9-23%, lộ rõ khác biệt.

Ngày đo: 2026-06-19. Script: `src/asr_lab/eval/sweep.py` + `src/asr_lab/data/hf_testset.py`. Nguồn test:
`hf-audio/esb-datasets-test-only-sorted` (test-only, Parquet, không cần torchcodec/trust_remote_code).

---

## Glossary

- **clean** — LibriSpeech dev-clean (giọng đọc sạch, dễ) — mốc đối chiếu.
- **voxpop** — VoxPopuli, phát biểu nghị viện EU, giọng đa dạng (vừa).
- **earn22** — Earnings22, họp báo tài chính, giọng đa quốc (khó).
- **ami** — AMI, ghi âm họp nhiều người + nhiễu phòng (khó).
- **RTF** — thời gian xử lý / thời lượng audio; thấp = nhanh.

---

## 1. Cấu hình đo

- **CPU**, cap 4 thread; mỗi model nạp 1 lần, chạy hết 4 bộ rồi giải phóng.
- **12 utterance/bộ**, lấy các utt DÀI nhất của mỗi bộ (slice `test[:12]` — bộ sort dài→ngắn).
  Audio: clean ~1,3 phút · voxpop 7,4 · earn22 6,3 · ami 4,4.
- **Chuẩn hoá WER:** hạ thường + bỏ dấu câu (model có PnC, ref thì không).

---

## 2. Ma trận WER%

| Model | Params·Decoder | clean | voxpop | earn22 | ami | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `stt_en_conformer_ctc_small` | 13M · CTC | 2,50 | 14,42 | 21,31 | 13,12 | 0,048 |
| `stt_en_fastconformer_transducer_large` | 115M · RNNT (VPB) | 2,00 | 12,09 | 22,91 | 17,12 | 0,053 |
| `parakeet-tdt-0.6b-v2` | 618M · TDT | **1,50** | **11,26** | **13,86** | 8,99 | 0,163 |
| `nemotron-speech-streaming-en-0.6b` | 618M · RNNT streaming | 6,00 | 14,60 | 16,38 | **8,84** | 0,169 |

---

## 3. Đọc kết quả (điểm cốt lõi)

- **Trên dữ liệu KHÓ, model lớn 0,6B thắng rõ — ngược hẳn với bộ clean.**
  - earn22: parakeet 13,86 / nemotron 16,38 << small 21,31 / fastconformer 22,91.
  - ami: parakeet 8,99 / nemotron 8,84 << small 13,12 / fastconformer 17,12.
  - Đây chính là điều bộ clean che mất: dung lượng dư của model 0,6B phát huy khi audio khó (nhiễu, giọng lạ, hội thoại), không phát huy trên giọng đọc sạch.
- **FastConformer-large 115M (cỡ model VPB) đuối trên audio khó:** 17-23% ở earn22/ami, tụt khá xa so với 0,6B. Trên giọng đọc sạch nó ngang ngửa, nhưng gặp họp/nhiễu/giọng đa quốc thì model lớn mới gánh được.
- **Nemotron lạ ở bộ clean (6,0%)** — cao hơn các model khác trên giọng đọc sạch, nhưng lại tốt nhất ở ami (8,84). Nó được tối ưu cho streaming/hội thoại hơn là đọc sạch. (Lưu ý: ở đây chạy offline, không phải streaming.)
- **Tốc độ:** small/115M nhanh hơn ~3 lần 0,6B (RTF ~0,05 so với ~0,16). RTF lần này thấp hơn lần đo trước vì utt dài + batch hiệu quả hơn — chỉ so tương đối trong cùng lần đo.

**Tóm lại:** chọn model là đánh đổi theo domain. Audio sạch → 115M là điểm ngọt (nhanh, đủ tốt). Audio khó thật (callbot, họp, nhiễu) → 0,6B đáng giá vì giảm WER rõ rệt.

## 4. Hạn chế (không thổi phồng)

- **Chỉ 12 utt/bộ → nhiễu thống kê lớn.** Vài chỗ đảo bậc khả năng do nhiễu (vd small 13,12 < fastconformer 17,12 ở ami; small 21,31 < fastconformer 22,91 ở earn22). Tin được là xu hướng TỔNG (0,6B << nhỏ ở earn22/ami), không phải con số lẻ.
- **Chỉ lấy utt DÀI nhất mỗi bộ** (slice đầu) — không đại diện toàn test split; utt ngắn (sub-giây) ở đuôi là mẩu vụn nên đã bỏ.
- **Nemotron chạy offline**, không dùng streaming → không phản ánh thế mạnh thật.
- **Tiếng Anh.** Đây là đo thông luồng + so sánh tương đối, không phải số leaderboard.

## 5. Cách chạy lại

```bash
# Tải lại test set (đuôi/đầu tuỳ slice) -> manifest
uv run python -m asr_lab.data.hf_testset ami --slice "test[:12]"
uv run python -m asr_lab.data.hf_testset earnings22 --slice "test[:12]"
uv run python -m asr_lab.data.hf_testset voxpopuli --slice "test[:12]"
# Quét 4 model x 4 bộ
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
uv run python -m asr_lab.eval.sweep
```

---

## ✅ Tự kiểm nhanh

1. Vì sao trên bộ clean model 0,6B không hơn model 115M, nhưng trên ami/earn22 lại hơn rõ?

<details><summary>Đáp án</summary>

Giọng đọc sạch quá dễ, model 115M đã gần trần (~2%) nên dung lượng dư của 0,6B không có chỗ phát huy. Audio khó (nhiễu, hội thoại, giọng đa quốc) mới cần năng lực biểu diễn lớn → 0,6B giảm WER rõ (vd ami 9% so với 13-17%).
</details>

2. Vì sao không nên tin các con số WER lẻ trong bảng này?

<details><summary>Đáp án</summary>

Mỗi bộ chỉ 12 utt nên sai số thống kê lớn; vài chỗ đảo bậc là do nhiễu. Chỉ nên đọc xu hướng tổng (0,6B tốt hơn hẳn trên dữ liệu khó), không phải từng số.
</details>
