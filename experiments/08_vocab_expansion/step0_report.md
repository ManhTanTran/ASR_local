# Step 0 — Đo hiện trạng lỗi f/j/w/z trên S3 (2026-07-05)

> **Vai trò:**
>
> Kết quả đo step 0 của task 08 — trả lời 3 câu hỏi trong [00_PLAN.md](00_PLAN.md) TRƯỚC khi tốn GPU cho expansion.
>
> Kết quả làm thay đổi kế hoạch: sinh thêm **step 0.5 (data audit)** trước cửa 1.

**Cách đo:** transcribe S3 (`s3-fc115m-full.nemo`) trên `vietsuperspeech.test` (1000 câu) + `fleurs_vi.test` (844 câu), dump hypothesis từng câu, phân tích alignment mức từ. Hyp lưu tại `dgx:/srv/team-share/datasets/asr_vi/_runs/probe_fjwz/`.

---

## 1. Phân rã lỗi callbot test — bảng chính

Subset "có f/j/w/z" KHÔNG thuần nhất — tách bằng heuristic mật độ từ tiếng Anh (≥40% từ trong câu là từ Anh phổ biến hoặc chứa f/j/w/z):

| Nhóm | Câu | Từ ref | Lỗi | WER | % tổng lỗi |
| --- | --- | --- | --- | --- | --- |
| Toàn bộ test | 1000 | 44.194 | 10.104 | **22,86%** | 100% |
| Không f/j/w/z (nền) | 659 | 33.985 | 4.847 | **14,26%** | 48,0% |
| Câu Việt có loanword | 189 | 7.197 | 2.480 | **34,45%** | 24,5% |
| Câu english-heavy | 152 | 3.012 | 2.777 | **92,2%** | 27,5% |

- **Nhóm "câu Việt có loanword"** (đích thật của expansion):
  - WER 34,5% — gấp 2,4× nền,
  - lỗi tại đúng TỪ chứa f/j/w/z: 932 lần xuất hiện → **932 lỗi, 0 lần đúng** (đúng bản chất: vocab không sinh nổi ký tự này),
  - lỗi tại từ f/j/w/z = **9,2% tổng lỗi** toàn test.
- **Nhóm "english-heavy"** (phát hiện NGOÀI dự kiến):
  - 152 câu (15% test) nhưng chiếm **27,5% tổng lỗi**, WER 92%,
  - trong đó **41 câu có lỗi > số từ ref** (WER câu >100%) — hyp dài gấp nhiều lần ref, nghi nhãn cắt cụt / audio-nhãn lệch,
  - nhìn mẫu: ref là tiếng Anh NHIỄU ("im profile", "the kitline put foo") — chính nhãn cũng không phải tiếng Anh chuẩn,
  - tập TRAIN vietsuperspeech cũng chứa **12,5% dòng english-heavy** (7.469/59.656) cùng loại → model đã và đang HỌC nhiễu này (thấy rõ: hyp chèn chuỗi từ Anh rác rất dài).

**Fleurs test (đối chiếu):** subset f/j/w/z = tên riêng nước ngoài (giza, ferguson, jakarta, ghz) — 113 câu, WER 26,8% vs nền 14,9%; lỗi tại từ f/j/w/z chỉ 3,4% tổng lỗi. Cùng bản chất nhóm "loanword", không có nhóm english-heavy.

---

## 2. Trả lời 3 câu hỏi của step 0

- **Câu 1 — trần do vocab lớn cỡ nào?**
  - trần TRỰC TIẾP (sửa hết lỗi tại từ f/j/w/z): 932/44.194 ≈ **−2,1 điểm** WER callbot,
  - trần GIÁN TIẾP (kéo cả nhóm loanword về nền 14,3% — gồm hết spillover): tiết kiệm ~1.451 lỗi ≈ **−3,3 điểm**,
  - phần còn lại của khoảng cách 22,86% ↔ 14,26% nằm ở nhóm english-heavy — KHÔNG phải bài toán vocab.
- **Câu 2 — lỗi thuộc từ f/j/w/z chiếm bao nhiêu?**
  - 9,2% tổng lỗi (callbot), 3,4% (fleurs) — nhỏ hơn nhiều so với cảm giác "34% câu dính".
- **Câu 3 — map từ điển (giải pháp A) sống được không?** → **Yếu, hạ độ ưu tiên.**
  - 227/932 lỗi là XOÁ HẲN (∅) — không có gì để map,
  - map biến thể top-1 (loại ∅) trần = 395/932 (42,4%) ≈ −0,9 điểm — nhưng là trần LẠC QUAN,
  - biến thể rất bất ổn ("soft" → ∅/t/proble; "power" → cation/pol/cost) và biến thể ngắn ĐỤNG từ Việt thật ("of"→"o", "how"→"ho" — map ngược "o"→"of" sẽ phá câu Việt),
  - chỉ ~10 cặp an toàn kiểu "isky→whisky", "rom→from" → ước thực tế **≤ −0,3 điểm**. Không đáng đầu tư quá nửa ngày.

---

## 3. Ước tác động các hướng (số ước, đánh dấu rõ)

| Hướng | Cơ chế | Callbot WER kỳ vọng | Ghi chú |
| --- | --- | --- | --- |
| A — map từ điển | ~10 cặp an toàn | 22,9 → ~22,6 (ước) | trần đo được −0,9; thực tế thấp hơn nhiều |
| C — expansion + fine-tune | sửa từ f/j/w/z + bớt spillover | 22,9 → ~19,5-21 (ước) | trần đo được −3,3; cần data nhãn sạch |
| Lọc english-heavy khỏi train | bớt học nhiễu | chưa có số | 12,5% train vss là nhiễu english |
| Lọc english-heavy khỏi test | đổi thước đo cho đúng phạm vi | 22,9 → 17,8 (đo lại được) | KHÔNG phải cải thiện model — là sửa thước |

- Số "17,8%" = WER đo lại trên 848 câu không-english-heavy ((4.847+2.480)/(33.985+7.197)) — nói rõ đây là **đổi thước đo**, chỉ hợp lệ nếu chốt phạm vi callbot là tiếng Việt + loanword.

---

## 4. Kết luận + việc sinh ra cho step tiếp

- **C (expansion + surgery) vẫn đúng hướng** — 932 lỗi tại từ f/j/w/z là lỗi "không thể thắng" hôm nay, và train data có content để học (27,4% dòng train vss chứa f/j/w/z).
- **Nhưng sinh thêm step 0.5 TRƯỚC cửa 1 — data audit + chốt phạm vi:**
  - đo kỹ nhóm english-heavy trong train (12,5%): bao nhiêu là nhãn bẩn thật, lọc thế nào,
  - cần Kỳ chốt business: callbot có phải nhận diện đoạn TIẾNG ANH thật không, hay chỉ tiếng Việt + loanword?
    - nếu chỉ Việt + loanword → lọc english-heavy khỏi train (bớt nhiễu) + báo cáo WER trên thước mới 17,8%,
    - nếu phải nhận tiếng Anh → cần nguồn data Anh nhãn SẠCH, bài toán lớn hơn vocab nhiều.
- **A** giữ lại như tiện ích nhỏ (10 cặp an toàn) — không phải một "giải pháp".

---

## 5. Bổ sung sau khi Kỳ chốt phạm vi (2026-07-05)

- **Phạm vi chốt:** callbot giao tiếp ngắn tiếng Việt, CÓ từ tiếng Anh ngắn (tên sản phẩm/tên riêng); KHÔNG làm hội thoại mix tiếng Anh dài (họp, lớp đa ngôn ngữ).
- **Đo lại trong phạm vi (848 câu = nền 659 + loanword 189):**
  - WER in-scope hiện tại: **17,8%** — đây là thước chính thức từ giờ,
  - từ f/j/w/z trong phạm vi: **351 lần / 201 từ khác nhau** (581 lần còn lại nằm ở nhóm english-heavy ngoài phạm vi),
  - trần sửa trực tiếp: 17,8% → **16,9%**; trần gồm hết lan toả (nhóm loanword về nền): → **~14,3%**.
- **Bằng chứng "thiếu chữ cái" là nguyên nhân trực tiếp** — model nghe ĐÚNG âm, chỉ không viết nổi ký tự thiếu:
  - facebook → "acebook" (×4), offline → "opline", first → "irst", full → "ull", follow → "ollo", whisky → "isky".
  - → phần âm học đã học xong; expansion chỉ cần dạy phần ký hiệu → xác suất fine-tune ngắn ăn nhóm từ này cao.
- **Từ đích tần suất cao:** of (42), whisky (22), fpt (9), we/for (9), first/file (5), web/life/west/facebook/fir (4)... — đuôi dài 201 từ; cần xem tay vài câu "of"/"whisky" ở step 0.5 để hiểu domain thật của corpus.

**Kết quả thực thi step 0.5 (cùng ngày, sau khi Kỳ OK lọc data):**
- rule lọc chốt: câu có **≥2 từ chức năng tiếng Anh khác nhau** VÀ tỉ lệ ≥0,3 (điều kiện ≥2 distinct để không oan câu Việt ngắn dính chữ trùng "to/an/no"),
- `vietsuperspeech.train.clean.jsonl`: giữ 54.025, lọc 5.631 (9,4% — xem tay đều là nhãn rác thật); bản lọc lưu `*.train.flagged.jsonl` để truy vết; **bản gốc không đụng**,
- viVoice **KHÔNG lọc**: chỉ 443 dòng dính (0,05%), nhãn sạch, có ca oan kiểu "trong quyển the eye of the storm" — chính là loanword hợp lệ,
- `vietsuperspeech.test.inscope.jsonl`: 842/1000 câu — **thước chính thức, WER S3 = 17,68%** (tính lại từ hyp đã dump, không tốn GPU),
- công cụ: [step0_5_audit.py](step0_5_audit.py) (audit + build + tính WER in-scope).

---

## 6. Tái lập

```bash
# trên DGX
cd /srv/team-share/projects/nvidia_asr_nemo
PYTHONPATH=deploy/asr_vi .venv/bin/python -u experiments/08_vocab_expansion/step0_transcribe.py \
  --nemo /srv/team-share/models/asr_vi/s3-fc115m-full.nemo \
  --manifests vss=/srv/team-share/datasets/asr_vi/_manifests/vietsuperspeech.test.jsonl \
              fleurs=/srv/team-share/datasets/asr_vi/_manifests/fleurs_vi.test.jsonl \
  --outdir /srv/team-share/datasets/asr_vi/_runs/probe_fjwz
python3 experiments/08_vocab_expansion/step0_analyze.py \
  --hyp /srv/team-share/datasets/asr_vi/_runs/probe_fjwz/s3_vss_hyp.jsonl --top 25 --samples 5
```
