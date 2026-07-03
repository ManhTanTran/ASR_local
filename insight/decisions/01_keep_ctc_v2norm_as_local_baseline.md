# decision 01 - giữ `vivos-fc-ctc-v2norm` làm baseline CTC local

**Bối cảnh:** `ASR_local` cần một mốc CTC sạch để so với các hướng FastConformer/RNNT bên `ASR` main và để thử decoder/LM sau này.

**Bằng chứng:** run `vivos-fc-ctc-v2norm` đã hoàn tất 50 epoch trên VIVOS, WER test **14.13%**, CER **7.67%**, best validation WER **13.10%** tại epoch 49. Xem `../../experiments/01_fastconformer_ctc_v2norm_kaggle/RESULT.md`.

**Chốt:** giữ run này làm **baseline CTC local**. Chưa coi là kết quả cuối hoặc tự động promote sang `ASR` main cho tới khi artifact được kéo đủ về local và có thử nghiệm kế tiếp trên decoder/best-checkpoint.

**Đánh đổi:** kết quả đủ tốt để chứng minh pipeline CTC, nhưng error analysis còn substitution cao (**88.3%** tổng lỗi word). Vì vậy việc tối ưu tiếp nên ưu tiên beam/LM/rescoring và nghe lại mẫu lỗi nặng trước khi train thêm lớn.
