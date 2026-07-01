# Git Setup

Ghi chú nhanh cho project ASR.

## Main repo

```powershell
cd C:\Users\Admin\OneDrive\Documents\Documents\ASR
git status --short --branch
git remote -v
```

Repo main đang theo:

```text
https://github.com/qualphachain/nvidia_asr_nemo.git
```

## Local workspace

`ASR_local/` không nhất thiết là git repo. Đây là nơi giữ notebook cuối, code tách riêng cho notebook, và ghi chú cá nhân.

## Nguyên tắc

- Không commit credential Kaggle.
- Không đưa file artifact lớn vào git nếu chưa cần.
- Nếu cần version các work note, tạo repo riêng hoặc đưa `_works/` vào repo có chủ đích.

