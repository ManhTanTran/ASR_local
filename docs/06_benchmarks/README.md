# Benchmark ASR — tổng kết khảo sát (để thảo luận team)

Kết quả đo **tốc độ + WER tiếng Anh** trên CPU, làm mốc trước khi plan train tiếng Việt.
Đây là khảo sát THÔNG LUỒNG + so sánh TƯƠNG ĐỐI, không phải số leaderboard.

> ⚠️ Tất cả model đo ở đây đều **English-only**. KHÔNG có pretrained tiếng Việt cho
> FastConformer/Parakeet/Nemotron-en. Bản duy nhất có vi-VN là `nemotron-3.5-asr-streaming-0.6b`
> (chưa test). Số dưới đây dùng để hiểu kiến trúc/decoder/size, KHÔNG dùng trực tiếp cho callbot Việt.

## Các báo cáo chi tiết

- [00_first_smoke_bench.md](00_first_smoke_bench.md) — thông luồng trên LibriSpeech clean (dễ).
- [01_hard_testsets_matrix.md](01_hard_testsets_matrix.md) — ma trận trên test set khó (VoxPopuli/Earnings22/AMI).
- [02_vivos_finetune.md](02_vivos_finetune.md) — **fine-tune sang tiếng Việt (VIVOS) trên Kaggle GPU:
  WER 100% → 20,37%** với model 115M offline; cơ chế deploy Kaggle + nhật ký debug.

## Bảng gộp (WER%, CPU, 12 utt/bộ ở cột khó)

| Model | Params·Decoder | clean | voxpop | earn22 | ami | RTF (khó) |
| --- | --- | --- | --- | --- | --- | --- |
| conformer-ctc-small | 13M · CTC | 2,50 | 14,42 | 21,31 | 13,12 | 0,048 |
| fastconformer-transducer-large | 115M · RNNT (cỡ VPB) | 2,00 | 12,09 | 22,91 | 17,12 | 0,053 |
| parakeet-tdt-0.6b-v2 | 618M · TDT | 1,50 | 11,26 | 13,86 | 8,99 | 0,163 |
| nemotron-speech-streaming-en-0.6b | 618M · RNNT stream | 6,00 | 14,60 | 16,38 | 8,84 | 0,169 |

## 3 điểm rút ra (cho thảo luận)

1. **Dữ liệu dễ che mất khác biệt.** Trên giọng đọc sạch mọi model ~2-6% → không tách bậc.
   Phải dùng test khó (WER 9-23%) mới so được model.
2. **Audio khó → size đáng giá.** Hai model 0,6B (Parakeet/Nemotron) giảm WER rõ trên Earnings22/AMI
   (9-16%) so với 13M/115M (13-23%). Ngược với bộ clean. Tức kiến trúc mới + lớn cải thiện THẬT
   trên dữ liệu khó, không chỉ trên benchmark đẹp.
3. **Đánh đổi tốc độ.** 13M/115M nhanh hơn ~3 lần 0,6B trên CPU. Chọn model là cân giữa
   độ khó dữ liệu thật (callbot có nhiễu/giọng vùng) và ngân sách hạ tầng.

## Hạn chế (đọc kèm để không kết luận quá)

- 12 utt/bộ ở cột khó → nhiễu thống kê lớn, vài chỗ đảo bậc do nhiễu; chỉ tin xu hướng tổng.
- Chỉ lấy utt dài nhất mỗi bộ; Nemotron chạy offline (không streaming); chỉ tiếng Anh.

## Hướng tiếp theo (chưa làm)

- **Train/fine-tune tiếng Việt** là việc chính — bàn với team.
- Nếu cần mốc tiếng Việt sẵn có: thử `nemotron-3.5-asr-streaming-0.6b` (có vi-VN) + bộ Việt khó hơn VIVOS
  (VIVOS là giọng đọc sạch, sẽ "dễ" như cột clean).
- Muốn số đáng tin hơn: tăng utt/bộ (~40) + lấy utt độ dài vừa (~6-10s) thay vì utt dài nhất.
