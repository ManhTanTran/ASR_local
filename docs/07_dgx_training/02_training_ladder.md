# 07.02 — Thang huấn luyện easy→hard + trick + nhóm model

Kế hoạch huấn luyện ASR tiếng Việt trên 1× GB10, đi từ dữ liệu dễ đến khó để model hội tụ dần.
Tổng hợp từ research (curriculum learning, fine-tune low-resource, đặc thù Blackwell/NeMo).
Số GPU-hour là **ước lượng bậc độ lớn**, phải đo lại bằng smoke test.

---

## 1. Thang curriculum (easy → hard)

Nguyên tắc: mồi hội tụ trên **đọc sạch, câu ngắn** trước; tăng dần nhiễu/tự nhiên/độ dài; luôn **replay** tập cũ để chống quên; giữ **eval set cố định** (FLEURS-vi + CV test) đo xuyên các nấc.

| Nấc | Thêm data (cộng dồn) | Giờ ~ | Tính chất | Mục tiêu |
| --- | --- | --- | --- | --- |
| **S1** | VIVOS + CV + FLEURS + InfoRE1 | ~55h | đọc sạch, câu ngắn | khóa âm vị + thanh điệu, hội tụ nhanh |
| **S2** | + FOSD + VLSP2020 + LSVSC | ~640h | đọc + tự nhiên nhẹ, nhiều speaker | mở rộng speaker/điều kiện thu |
| **S3** | + VietSuperSpeech + Bud500 + InfoRE2 | ~1.660h | hội thoại + audiobook dài + 3 miền | robust domain callbot + câu dài |
| **S4 (sau)** | + gated: PhoAudiobook / viVoice / VietMed | +2.000h | bulk + y tế | đẩy trần chất lượng, domain hẹp |

Cơ chế curriculum trong NeMo:
- **Trong 1 nấc:** dùng **sorted bucketing** (gom câu cùng độ dài) — vừa tăng throughput vừa cho hiệu ứng "câu ngắn trước" epoch đầu. Rẻ, nên bật mặc định.
- **Giữa các nấc:** resume từ `.nemo` nấc trước, đổi manifest = tập cũ + tập mới (có trọng số), lịch LR mới.
- **Chống quên (replay):** luôn giữ VIVOS + CV trong mix; **upsample tập sạch nhỏ** để không bị Bud500 nuốt.

> Ghi chú tỉnh táo: curriculum giúp **tốc độ hội tụ** rõ, nhưng lợi WER cuối là **khiêm tốn** và phụ thuộc tiêu chí sắp xếp (nghiên cứu gần đây: sắp theo độ-khó-nhận-dạng > sắp theo độ dài thuần).
> Coi length-sort là công cụ hội tụ rẻ, không phải viên đạn bạc WER.

---

## 2. Trick tăng WER (xếp theo tác động)

1. **Learning rate + lịch** — đòn bẩy lớn nhất. LR quá cao phá feature pretrained. Dùng **1e-4 → 1e-5**, thay Noam bằng **CosineAnnealing + warmup ngắn** (~500-1000 step), `min_lr ~1e-6`.
2. **Tokenizer/vocab** — tiếng Việt cần BPE phủ dấu + thanh. `change_vocabulary()` đổi vocab decoder, giữ encoder. BPE **256–1024** hợp 50–500h. Khi đổi vocab cho Transducer: **init encoder-only**, loại decoder+joint (`init_from_nemo_model_exclude`).
3. **Cân bằng corpus + chống quên** — manifest nhiều tập + trọng số/upsample tập sạch; **replay** tập cũ; theo dõi val set nguồn để phát hiện quên.
4. **SpecAugment** — bật khi fine-tune; tăng masking cho low-resource chống overfit (nhưng quá tay làm chậm hội tụ). Khởi điểm: freq_masks 2 (width ~27), time_masks 5-10.
5. **Freeze/unfreeze encoder** — data rất ít: train decoder+joint trước, rồi mở encoder với LR thấp.
6. **Precision** — **`bf16-mixed`** trên Blackwell (ổn hơn fp16, không lo loss-scaling). FP8/MXFP4 chưa cần cho 115M, chưa kiểm chứng cho RNNT → bỏ qua.
7. **Batch + grad-accum** — RNNT ngốn bộ nhớ; fit batch/GPU rồi `accumulate_grad_batches` để đạt effective batch vài trăm.
8. **Best-checkpoint theo val WER + EMA (~0.999)** — lợi nhỏ, rẻ, rủi ro thấp.

---

## 3. Transducer vs CTC vs hybrid

- **Transducer (RNNT):** WER offline tốt nhất, LM ngầm, hợp thanh điệu tiếng Việt; nặng train, khó streaming ổn định.
- **CTC:** nhẹ, nhanh, throughput cao; đủ khi có LM n-gram ngoài hoặc cần long-form/streaming.
- **Hybrid Transducer-CTC (khuyến nghị):** 1 lần train ra 2 đầu — RNNT cho offline, CTC cho streaming/long-form rẻ.

> Kinh nghiệm cũ của lab (streaming transducer **không hội tụ**, offline **được**) khớp với khó khăn đã biết của cache-aware RNNT.
> Định hướng: **train offline trước** cho mạnh, rồi mới adapt streaming ở nấc riêng — không khởi động streaming từ đầu.

---

## 4. Nhóm model nên train (tài sản lab)

| Nhóm | Model | Từ checkpoint | Mục đích | Ưu tiên |
| --- | --- | --- | --- | --- |
| A | FastConformer Transducer 115M (hybrid CTC) | ckpt VIVOS hiện có / `nvidia/parakeet` | model VI offline nền tảng, chạy thang S1→S3 | **1** |
| B | Scale 600M–1B (parakeet/canary VI) | `nvidia/parakeet-ctc-0.6b` hoặc canary | đẩy trần WER khi corpus đã đủ | 2 |
| C | Streaming adapt (cache-aware) từ A | model A đã mạnh | callbot real-time FCI | 3 |
| D | So sánh PhoWhisper / Whisper VI (baseline ngoài) | `vinai/PhoWhisper-large` | mốc đối chiếu, không train | 3 |

Bắt đầu **nhóm A** — rẻ, chắc, tạo tài sản đầu tiên. B/C sau khi A ổn.

---

## 5. Kỳ vọng thực tế trên 1× GB10 (ước lượng — phải đo)

- **115M fine-tune trên 50–500h:** khả thi tốt. ~vài GPU-hour/epoch ở mức cao; tổng single-digit đến vài chục GPU-hour cho 1 fine-tune tốt. Thoải mái bộ nhớ.
- **600M–1B fine-tune:** được nhưng chậm — ngày chứ không phải giờ; canh bộ nhớ với grad-accum + bucketing.
- **Train-from-scratch model lớn (nghìn giờ):** **KHÔNG khả thi** trên 1 GB10 — việc của cụm nhiều GPU.

---

## 6. Bước chạy (khi corpus đã về)

1. **Smoke máy (bắt buộc trước run dài):** 50-step trên VIVOS, xác nhận **RNNT loss + SpecAugment kernel chạy trên sm_121** (GB10 chưa có benchmark NeMo-ASR công khai — phải tự verify).
2. **Setup repo trên DGX:** clone `nvidia_asr_nemo`, `uv sync` với torch cu130 (không dùng index CPU như local — sửa `[tool.uv.sources]` cho môi trường GPU), verify `import nemo` + CUDA.
3. **Build manifest** các bộ đã kéo (xem `01_corpus_pull.md` §Bước tiếp).
4. **Nấc S1:** fine-tune 115M hybrid, bf16-mixed, cosine 1e-4, bucketing, eval FLEURS-vi + CV test.
5. **Nấc S2, S3:** resume + thêm data + replay; ghi mỗi nấc thành experiment (`spec.md`/`RESULT.md`).
6. Cập nhật `_SCOREBOARD.md` sau mỗi nấc.

---

## ✅ Tự kiểm nhanh (đáp án ẩn cuối)

1. Vì sao train easy→hard? 2. Replay chống điều gì? 3. Precision nào trên GB10? 4. Vì sao offline trước streaming? 5. Việc gì KHÔNG khả thi trên 1 GB10?

<details><summary>Đáp án</summary>

1. Hội tụ nhanh, ít phân kỳ (lợi WER cuối khiêm tốn). 2. Catastrophic forgetting khi thêm domain mới. 3. `bf16-mixed`. 4. Cache-aware RNNT streaming khó hội tụ; cần model offline mạnh làm nền rồi mới adapt. 5. Train-from-scratch model lớn nghìn giờ (cần cụm nhiều GPU).
</details>
