# FastConformer-RNNT 115M — phân tích lỗi output sau fine-tune tiếng Việt + hướng cải thiện

Report soi kỹ output của model `stt_en_fastconformer_transducer_large` (115M) sau khi fine-tune VIVOS
(WER 20,37%), tìm **lỗi chung lặp lại**, truy **root-cause** (gồm 1 giả thuyết đã bị bác bằng thí
nghiệm), và đề xuất hướng giảm WER.

---

## Glossary

- **`<unk>`:** token "không biết" của tokenizer; in ra dạng `⁇`. Xuất hiện khi text chứa ký tự KHÔNG
  có trong vocab của tokenizer.
- **normalize_vi:** chuẩn hoá = hạ thường + bỏ dấu câu, GIỮ chữ có dấu tiếng Việt.
- **Tokenizer/manifest mismatch:** tokenizer học trên text kiểu A, nhưng nhãn train ở kiểu B → ký tự
  lệ thuộc kiểu B không có trong vocab → thành `<unk>`.
- **RNNT greedy:** giải mã từng frame, có thể phát token rỗng (blank) hoặc token thật.

---

## 1. Lỗi chung quan sát được

Đọc cặp ref/hyp (đã normalize), lỗi nổi bật và LẶP LẠI là **mất ký tự đầu từ đầu câu**:

```
REF: tuy nhiên ca ghép ...      HYP: uy nhiên cây ghép ...     (mất 't')
REF: ăn ở có nhân có nghĩa       HYP: n ở có nhân có nghĩa       (mất 'ă')
REF: những loại thực phẩm này    HYP: hững loại thực phẩm này    (mất 'n')
REF: hai mươi sáu ...            HYP: ai mươi sáu ...            (mất 'h')
REF: sự buồn tẻ ...              HYP: ự buồn tệ ...              (mất 's')
```

Phần thân câu khá chuẩn → đây là lỗi **hệ thống ở đầu chuỗi + danh từ riêng**, không phải nhiễu ngẫu nhiên.

## 2. Truy root-cause

### 2.1 Giả thuyết A (BÁC BỎ bằng thí nghiệm): lỗi biên audio

Giả thuyết: audio bắt đầu ngay, frame đầu chưa "ấm" → rớt âm đầu. **Thử**: đệm 0,3s im lặng đầu mỗi
audio rồi đo lại (60 utt).

| | WER | khớp-ký-tự-đầu |
| --- | --- | --- |
| Gốc | 23,44% | 16/60 |
| Đệm 0,3s | 23,83% | 15/60 |

→ **Không đổi.** Lỗi KHÔNG do biên audio. Loại bỏ giả thuyết A.

### 2.2 Root-cause THẬT: tokenizer/manifest mismatch → `<unk>` ở chữ HOA

Soi output THÔ (trước normalize) lộ thủ phạm — `⁇` (`<unk>`) đứng đúng chỗ **chữ HOA**:

```
REF: 'Tuy nhiên, ca ghép ... ông Liêm nói.'
RAW: ' ⁇ uy nhiên ⁇  cây ghép ... ông  ⁇ iêm nói ⁇ '       (T->⁇, L->⁇, dấu câu->⁇)
REF: 'Học bổng ... tại Hàn Quốc.'
RAW: 'học bổng ... sau tiếng sĩ  ⁇ ài  ⁇ àn  ⁇ uốc ⁇ '       (tài/Hàn/Quốc: H,Q,t-hoa->⁇)
```

Cơ chế:
1. **Tokenizer** được train từ text đã `normalize_vi` → vocab CHỈ có chữ thường, không có chữ HOA/dấu câu.
2. **Manifest train** lại để text RAW của VIVOS ("Tuy nhiên, ... ông Liêm nói.") — có HOA + dấu câu.
3. Khi train, NeMo token hoá nhãn bằng tokenizer thường → mọi chữ HOA + dấu câu → **`<unk>`**.
4. Model **học phát `<unk>`** ở: chữ hoa ĐẦU CÂU (luôn có) + chữ hoa danh từ riêng (Liêm, Hàn, Quốc).
5. Lúc eval, `normalize_vi` hạ thường ref + bỏ `⁇` khỏi hyp → `<unk>` biến mất → trông như "rớt ký tự đầu".

→ Đây là lỗi **dữ liệu/cấu hình**, không phải năng lực model. Mỗi câu mất ≥1 từ đầu + sai danh từ
riêng → đẩy WER lên đáng kể.

## 3. Hướng cải thiện (xếp theo impact/effort)

1. **[ĐÃ SỬA CODE — impact cao, effort thấp] Chuẩn hoá nhãn train/val KHỚP tokenizer.**
   `prepare_data` nay `normalize_vi` text train/val trước khi train (hạ thường + bỏ dấu câu) → vocab
   phủ hết ký tự nhãn → KHÔNG còn `<unk>`. Kỳ vọng giảm WER rõ (sửa toàn bộ lỗi đầu-câu + danh từ riêng).
   *Cần 1 lần train lại để xác nhận.*
2. **Giảm overfit / tăng tổng quát** (val plateau ~17% trong khi train-batch WER ~1-3%; test speakers
   TÁCH BIỆT train): bật SpecAugment mạnh hơn, thêm dữ liệu đa giọng (Common Voice VI, FOSD), chọn
   **best-checkpoint theo val** thay vì epoch cuối.
3. **Khử dấu câu/PnC nhất quán** ở cả pipeline (nếu sau này cần dấu câu thì phải đưa dấu câu vào vocab
   tokenizer + giữ ở nhãn, không nửa vời).
4. **Soi train vs test** cho lỗi đầu-câu sau khi sửa (1): nếu vẫn còn, mới xét cơ chế giải mã RNNT.

## 4. Bài học rút ra (tái dùng)

- **Tokenizer và nhãn train PHẢI cùng một chuẩn hoá.** Đây là bẫy im lặng: train chạy "thành công",
  loss giảm, nhưng model học cả thói quen phát `<unk>` — chỉ lộ khi soi output THÔ.
- **Luôn soi output `repr()` (chưa normalize)** khi WER cao bất thường — normalize có thể che lỗi
  (ở đây `<unk>` bị normalize nuốt mất, làm lỗi trông giống "rớt ký tự").
- **Negative result có giá trị**: thử đệm im lặng loại trừ giả thuyết audio, dẫn tới đúng root-cause.

---

## ✅ Tự kiểm nhanh

1. Vì sao output mất "ký tự đầu" mà đệm im lặng không sửa được?

<details><summary>Đáp án</summary>

Vì không phải lỗi audio. Thật ra model phát `<unk>` (⁇) cho chữ HOA (đầu câu + danh từ riêng) do
tokenizer (train trên text thường) không có chữ hoa; eval normalize bỏ `⁇` nên trông như rớt ký tự.
Gốc rễ = tokenizer/manifest mismatch.
</details>

2. Cách sửa gốc rễ + vì sao đúng?

<details><summary>Đáp án</summary>

Chuẩn hoá nhãn train/val GIỐNG cách build tokenizer (cùng `normalize_vi`: hạ thường + bỏ dấu câu).
Khi đó mọi ký tự nhãn đều nằm trong vocab → không sinh `<unk>` → model không học phát ⁇.
</details>
