# 09.02 — Kinh nghiệm train đã tích luỹ: thứ tự curriculum & chi phí từng lesson

> **Vai trò:**
>
> Gói lại số đo THẬT của 3 nấc S1→S2→S3 đã chạy.
>
> Làm cơ sở ước lượng chi phí + rủi ro cho phase 2 (rebuild vocab).

---

## Glossary

- `nấc / lesson` → **curriculum stage** → một lần fine-tune thêm một nhóm dataset mới.
- `n_train` → **training lines per epoch** → số dòng manifest mỗi epoch, ĐÃ nhân trọng số upsample.
- `epoch` → **epoch** → một lượt quét hết `n_train` dòng.
- `GPU-h` → **GPU-hour (wall-clock)** → giờ đồng hồ thực GPU chạy, GỒM cả lúc chia GPU với job khác.
- `WER` → **Word Error Rate** → tỉ lệ lỗi từ; càng thấp càng tốt.
- `replay` → **replay** → giữ lại tập cũ trong mix nấc sau để chống quên.
- `upsample` → **upsample** → nhân bản (× trọng số) tập sạch nhỏ để không bị tập lớn nuốt.
- `cap` → **subsample cap** → chặn tối đa N clip của tập quá lớn (lấy đều tay).
- **T** → **encoder frames** → số khung thời gian sau encoder (∝ thời lượng audio).
- **U** → **label length** → số token của nhãn.
- **V** → **vocab size** → kích thước từ điển (ở đây 1024).
- **B** → **batch size** → số clip mỗi bước.

---

## Dẫn dắt bối cảnh

- Hình dung một xưởng rèn chỉ có **một lò** (1× GPU GB10):
  - mỗi mẻ rèn (nấc train) tốn nhiều giờ,
  - nên phải biết trước mẻ nào tốn bao nhiêu than để xếp lịch.
- Nghịch lý đã gặp:
  - nấc S3 có **gấp ~3,6 lần** số dòng của S2,
  - nhưng lại chạy **ít epoch hơn** và tốn giờ **tương đương** —
  - vậy "số dòng" KHÔNG phải thước đo chi phí đúng.

> Tài liệu này bóc tách chi phí train về các đại lượng gốc (thời lượng audio, độ dài nhãn, vocab), rồi đối chiếu với số đo thật của ba nấc để rút ra công thức ước lượng cho phase 2.

---

## 1. Thứ tự curriculum đã chạy easy đến hard

Nguyên tắc: mồi hội tụ trên **đọc sạch câu ngắn** trước, tăng dần nhiễu/tự nhiên/độ dài, luôn **replay** tập cũ. Chi tiết thiết kế: [../07_dgx_training/02_training_ladder.md](../07_dgx_training/02_training_ladder.md).

- **S1 — đọc sạch, câu ngắn:**
  - dataset: VIVOS + Common Voice + FLEURS + InfoRE1,
  - tính chất: studio/đọc, clip ngắn, ít nhiễu,
  - mục tiêu: khoá âm vị + thanh điệu, cho encoder hội tụ nhanh.
- **S2 — thêm tự nhiên nhẹ, nhiều speaker:**
  - thêm: FOSD + VLSP2020 + LSVSC (giữ replay S1),
  - tính chất: đọc tin tức + tự nhiên đa speaker/điều kiện thu,
  - mục tiêu: mở rộng speaker + điều kiện thu.
- **S3 — thêm hội thoại + audiobook dài:**
  - thêm chính: VietSuperSpeech + Bud500 (hội thoại),
  - thêm liều nhỏ: viVoice + phoaudiobook + InfoRE2 (audiobook, `cap` nhỏ),
  - mục tiêu: robust domain callbot + câu dài.

### 1.1 Khái niệm replay chống quên

- ⚙️ **Cơ chế:**
  - mỗi nấc GIỮ lại tập cũ (VIVOS/CV/FLEURS) trong mix, `upsample` để không bị tập mới nuốt.
- 🔍 **Nhận diện:**
  - trong `train_sets` của config, tập cũ vẫn có mặt với `weight ≥ 2`.
- 💡 **Ý nghĩa:**
  - giữ WER các test cũ KHÔNG xấu đi khi thêm domain mới (đo thật: cả 9/9 test cải thiện qua các nấc).
- ⚠️ **Bẫy:**
  - thiếu replay → catastrophic forgetting (model quên đọc-sạch khi học hội thoại).

---

## 2. Chi phí train về đại lượng gốc

Chi phí train RNNT KHÔNG tỉ lệ với "số dòng", mà với **thể tích tensor joint** mỗi bước.

$$ C_{epoch} \;\propto\; V \cdot \sum_{i=1}^{N} T_i \cdot U_i $$

Giải nghĩa từng ký hiệu (trái → phải):
- **C_epoch** = chi phí tính toán một epoch (∝ GPU-hour).
- **V** = **vocab size** (ở đây 1024) — joint chiếu ra V+1 lớp đầu ra.
- **N** = số clip trong epoch (sau `cap` + `upsample`).
- **T_i** = **encoder frames** của clip i (∝ thời lượng audio / bước subsample 80 ms).
- **U_i** = **label length** của clip i (số token nhãn).

Hệ quả rút ra:
- Joint của RNNT là tensor **B × T × U × V** → chi phí đội theo CẢ độ dài audio (T), độ dài nhãn (U) VÀ vocab (V).
- Clip **hội thoại dài** (T, U lớn) đắt hơn nhiều clip **đọc ngắn**, dù đếm "1 dòng" như nhau.
- Tăng **V** (rebuild vocab lớn hơn) → tăng chi phí joint tuyến tính theo V → giữ V = 1024 là lựa chọn tiết kiệm.

---

## 3. Chi phí đo thật của ba nấc

Số từ `results.json` mỗi run (`train_sec`, `n_train`, `completed_epochs`).

| Nấc | n_train/epoch | epochs | GPU-h (wall) | h/epoch | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| **S1** | 63.073 | 20 | **7,5h** | 0,38 | clip ngắn → rẻ/epoch, nhiều epoch |
| **S2** | 172.999 | 8 | **19,9h** | 2,49 | +tự nhiên; nửa đầu chia GPU |
| **S3** | 623.311 | 4 | **~21h** (16,7 + 4,35 resume) | ~5,25 | +hội thoại clip dài; nửa đầu chia GPU |
| **Tổng 3 nấc** | | | **~48 GPU-h** | | trên 1× GB10 |

> ⚠️ Số GPU-h là **wall-clock**, GỒM lúc chia GPU với job khác (S2 và nửa đầu S3 bị chậm ~2× do share). GPU độc chiếm sẽ thấp hơn.

### 3.1 Khung đọc bảng chi phí

- **Đề bài:** mỗi nấc tốn bao nhiêu giờ GPU, vì sao lệch nhau.
- **Giả định:** cùng model 114M, cùng eff-batch 256, bf16; khác nhau ở data + epoch + mức chia GPU.
- **Trục đọc:**
  - cột `n_train/epoch` = khối lượng mỗi epoch (đơn vị: dòng),
  - cột `GPU-h` = tổng giờ đồng hồ (đơn vị: giờ),
  - cột `h/epoch` = chi phí một epoch (đơn vị: giờ/epoch).
- **Cách đọc — vì sao hình dạng vậy:**
  - S1 rẻ/epoch (0,38h) dù chạy 20 epoch, vì clip **ngắn** (T, U nhỏ) → tổng T·U nhỏ.
  - S2 đắt/epoch gấp ~6× S1 dù n_train chỉ ~2,7× — vì clip **dài hơn** (tự nhiên) → T·U lớn hơn, đúng công thức §2.
  - S3 đắt/epoch nhất (~5,25h) vì vừa **nhiều clip** vừa **dài** (hội thoại) — nhưng chỉ cần 4 epoch nên tổng ~ngang S2.

### 3.2 Quy tắc ngón tay cái cho phase 2

- **Continue-train từ .nemo có sẵn (giữ vocab):** ~4-6 GPU-h/epoch với mix cỡ S3 → 4-5 epoch ≈ **20-30 GPU-h**.
- **Rebuild vocab rồi train lại từ đầu curriculum:** ~48 GPU-h (như đã đo) — **đắt nhất**, đây là lý do vocab là "issue tốn kém nhất".
- **Nhánh s3rv (rebuild-vocab từ S3, đã thử):** 3 epoch ~12,7h nhưng **thất bại** (WER xấu hơn, xem [00_overview](00_overview.md)).

---

## 4. Quỹ đạo WER từng nấc

Đo trên suite test cố định; mỗi ô là WER sau nấc (%).

| Test | S1 | S2 | S3 | Ghi chú |
| --- | --- | --- | --- | --- |
| cv | 31,3 | 21,4 | **17,2** | mic đời thường |
| fleurs | 25,0 | 20,3 | **16,5** | studio chuẩn |
| vlsp | — | 31,2 | **24,8** | tin tức formal |
| lsvsc | — | 15,9 | **13,1** | tự nhiên đa miền |
| fosd | — | 22,8 | **20,0** | đọc FPT |
| vivos | — | 9,6 | **8,5** | đọc studio |
| vietsuperspeech | — | 40,0 | **22,9** | **callbot** (mở ở S3) |
| bud500 | — | 16,4 | **6,7** | hội thoại 3 miền |

- Mỗi nấc kéo TOÀN BỘ test đang có xuống — replay hoạt động, không quên.
- Nấc mở domain mới (S3 mở hội thoại) cho lợi lớn nhất ở test tương ứng (callbot 40 → 22,9).

---

## 5. Bài học rút ra

- **Bài học vocab (tốn kém nhất):**
  - curriculum GIỮ nguyên tokenizer VIVOS-era qua cả S1→S2→S3 (`change_vocabulary=false`),
  - tokenizer nhỏ đó KHÔNG phủ charset các tập lớn nhiễu phía sau (thiếu f/j/w/z),
  - → kẹt trần loanword; sửa sau bằng rebuild giữa chừng thì reset decoder+joint (đắt + hỏng).
  - Kết luận: **charset phải chốt ĐÚNG từ nấc đầu** — xem [01_incremental_vocab_methods](01_incremental_vocab_methods.md).
- **Bài học chi phí:**
  - đơn vị chi phí là **T·U·V** chứ không phải "số dòng" → ước lượng theo thời lượng audio + độ dài nhãn.
- **Bài học vận hành:**
  - flagship run cần **độc chiếm GPU** (chia GPU cắt ~½ tốc độ), và nâng `max_minutes` khi resume (xem [../07_dgx_training/09_splits_eval_lineage](../07_dgx_training/09_splits_eval_lineage.md)).

---

## ✅ Tự kiểm nhanh

1. Vì sao S3 nhiều dòng gấp ~3,6× S2 mà tổng giờ chỉ ngang nhau?
2. Đơn vị chi phí train RNNT là gì, tăng vocab ảnh hưởng ra sao?
3. Rebuild vocab rồi train lại từ đầu tốn ~bao nhiêu GPU-h, so continue-train?

<details><summary>Đáp án</summary>

1. S3 chỉ chạy 4 epoch (S2 chạy 8); và chi phí tính theo T·U·V — S3 clip dài nhưng ít epoch nên tổng ~ngang.
2. Chi phí ∝ V · Σ(T_i·U_i) (thể tích tensor joint B×T×U×V); tăng V làm đội chi phí joint tuyến tính → giữ V=1024.
3. Rebuild + train lại toàn curriculum ~48 GPU-h (đắt nhất); continue-train giữ vocab ~20-30 GPU-h cho 4-5 epoch.

</details>
