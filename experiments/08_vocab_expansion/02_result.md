# 08.02 — Kết quả task vocab expansion

> **Vai trò:** gói kết quả 2 cửa + verdict trung thực + recipe vòng sau.
>
> Chốt: **cơ chế phẫu thuật đúng hoàn toàn (cửa 1) và fine-tune KÍCH HOẠT được token mới (0 → 99 từ đúng)**, nhưng recipe cửa 2 này net-neutral vì đánh đổi replay + train ngắn. S3 vẫn là model production.

---

## 1. Cửa 1 — verify phẫu thuật (FINAL, PASS tuyệt đối)

Model `s3-vocabexp.nemo` (thêm f/j/w/z, V 1024 → 1028), eval NGAY sau phẫu thuật, CHƯA train — khớp S3 chính xác cả 10 metric:

| Test | S3 | s3-vocabexp (chưa train) | Δ |
| --- | --- | --- | --- |
| vivos | 8,47 | 8,47 | 0,00 |
| cv | 17,19 | 17,20 | +0,01 |
| fleurs | 16,46 | 16,45 | −0,01 |
| vlsp | 24,81 | 24,81 | 0,00 |
| lsvsc | 13,12 | 13,12 | 0,00 |
| fosd | 19,96 | 19,95 | −0,01 |
| vss full | 22,87 | 22,86 | −0,01 |
| vss in-scope | 17,68 | 17,68 | 0,00 |
| bud500 | 6,73 | 6,73 | 0,00 |
| vietmed | 26,38 | 26,38 | 0,00 |

- **Kết luận:** expansion + tensor surgery giữ nguyên 100% hành vi model. Cập nhật vocab KHÔNG cần re-train — khác hẳn s3rv (reset toàn phần → 2400%). Rủi ro lớn nhất của task đã khử với 0 GPU-h.

---

## 2. Cửa 2 — fine-tune (2 epoch, freeze encoder, mix gọn 122k, 3,67h)

### 2.1 Bảng 10 test: before (S3) → after

| Test | before | after | Δ | |
| --- | --- | --- | --- | --- |
| **vss in-scope** (thước chính) | 17,68 | **17,43** | **−0,25** | tốt nhẹ |
| vss full | 22,86 | 22,35 | −0,51 | tốt nhẹ |
| bud500 | 6,73 | 6,61 | −0,12 | tốt nhẹ |
| vivos | 8,47 | 8,58 | +0,11 | xấu nhẹ |
| cv | 17,20 | 17,36 | +0,16 | xấu nhẹ |
| fleurs | 16,45 | 16,75 | +0,30 | xấu |
| lsvsc | 13,12 | 13,45 | +0,33 | xấu |
| vlsp | 24,81 | 25,26 | +0,45 | xấu |
| **fosd** | 19,95 | 21,33 | **+1,38** | xấu, vượt ngưỡng 1 điểm |
| vietmed | 26,38 | 26,44 | +0,06 | xấu nhẹ |

- Nhóm hội thoại (vss, bud500) cải thiện nhẹ; nhóm đọc/formal (fosd, vlsp, lsvsc, fleurs, vivos) thoái lui.
- **fosd +1,38 vượt ngưỡng H2 (< 1 điểm)** → cửa 2 KHÔNG pass sạch ở tiêu chí giữ nền.

### 2.2 Mức kích hoạt token f/j/w/z — bằng chứng cơ chế

| Chỉ số (vss test) | S3 | S4 |
| --- | --- | --- |
| từ f/j/w/z đúng nguyên văn | **0 / 932** | **99 / 932** |
| WER subset có f/j/w/z | 51,49% | 49,87% |
| WER subset không f/j/w/z | 14,26% | 14,10% |
| câu Việt sạch bị chèn f/j/w/z bừa | 0% | **3,3%** |

- **Token mới ĐÃ được kích hoạt:** whisky nghe ra "wisky"/"wosky" (trước "isky"), fpt ra "fp"/"ept" (trước "t") — model giờ phát được 'f','w'. Từ điều KHÔNG THỂ (0) thành làm được một phần (99 đúng + nhiều từ phát đúng ký tự).
- fleurs cùng chiều: 0 → 10 đúng / 140.
- **Cái giá:** 3,3% câu Việt sạch bị chèn f/j/w/z bừa (false positive) — token mới còn "quá hăng" sau 2 epoch, góp phần làm nhóm đọc xấu đi.

### 2.3 Vì sao gain nhỏ dù cơ chế chạy — chẩn đoán

- **Đánh đổi replay:** mix gọn cap mạnh nhóm đọc (fosd 23.668 → 4.000, vlsp 52.414 → 6.000) → decoder+joint trôi về giọng hội thoại, quên giọng đọc-sạch → nhóm đọc thoái lui. Đúng rủi ro đã ghi ở [01_surgery_design.md](01_surgery_design.md) §7.
- **Train ngắn + freeze encoder:** 2 epoch chưa đủ để (a) học hết 833 từ f/j/w/z còn sai, (b) calib token mới hết "hăng" (3,3% false positive); 242/833 lỗi là XOÁ HẲN (∅) — encoder đóng băng không sửa được ca model "nuốt" từ.
- Kết quả tổng: −0,25 ở thước chính bị 6 test đọc xấu đi che lấp → net-neutral.

---

## 3. Verdict + recipe vòng sau

**Verdict trung thực:**
- ✅ **Thắng về cơ chế (điều quan trọng nhất):** phẫu thuật đúng tuyệt đối + fine-tune chứng minh token mới học được (0 → 99). Con đường "expansion + surgery" khả thi end-to-end, khác hẳn s3rv chết.
- ⚠️ **Recipe này chưa dùng được:** net-neutral, fosd vượt ngưỡng. **S3 vẫn là model production**; s4-vocabexp-ft.nemo chỉ là mốc chứng minh cơ chế, KHÔNG thay S3, KHÔNG publish.

**Recipe vòng sau (2 điều sửa rõ ràng từ chẩn đoán):**
- **Giữ FULL replay** nhóm đọc (không cap fosd/vlsp/lsvsc) — chặn trôi giọng, giữ 9 test. Chấp nhận epoch nặng hơn (cần GPU rảnh hoặc chạy dài).
- **Train lâu hơn + mở encoder lr nhỏ** (vd 2e-5) để (a) sửa ca ∅ deletion, (b) calib token mới hết false positive; hoặc thêm regularization giữ phân phối token cũ.
- (tuỳ) thêm loanword nguyên từ (fpt, wifi, facebook) làm token atomic để học nhanh hơn 4-char.

---

## 4. Tài sản để lại

- `s3-vocabexp.nemo` — model vocab 1028, hành vi = S3, **dùng làm điểm xuất phát chuẩn cho mọi thử nghiệm vocab sau** (khỏi phẫu thuật lại).
- `s4-vocabexp-ft.nemo` — mốc chứng minh cơ chế (không production).
- Code tái dùng: [surgery.py](surgery.py), [step0_transcribe.py](step0_transcribe.py), [step0_analyze.py](step0_analyze.py), [step0_5_audit.py](step0_5_audit.py).
- Manifest sạch: `vietsuperspeech_clean.{train,val}.jsonl`, thước `vietsuperspeech.test.inscope.jsonl` (bản gốc không đụng).
