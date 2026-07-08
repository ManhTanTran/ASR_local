# ASR main conflict snapshots

Folder này giữ bản copy từ `ASR`/`origin/main` cho các file có cùng path với `ASR_local` nhưng khác nội dung tại thời điểm gộp ngày 2026-07-08.

Quy ước:

- File chỉ có ở `ASR` đã được copy vào đúng path trong `ASR_local`.
- File trùng path khác nội dung không bị overwrite để giữ nguyên kết quả/code local.
- Bản từ `ASR` của các file conflict nằm ở `from_ASR_main_20260708/` để đối chiếu hoặc merge thủ công sau.
- `deploy/kaggle/accounts.json` được bỏ qua để tránh kéo config/credential local.
