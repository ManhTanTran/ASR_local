# Issue Notes - FastConformer Fine-tune

## Vấn đề

Notebook cũ chứa quá nhiều logic inline: tải data, normalize, train, eval, compare, export artifact trong cùng một file.

## Cách xử lý

Tách logic sang `src/asr_local/`:

- `model/fastconformer.py`: config run.
- `deploy/kaggle.py`: build/push/poll/pull.
- `analytics/compare.py`: load results, report, artifact manifest.

Notebook mới chỉ còn:

1. Bootstrap package.
2. Tạo config.
3. Gọi build/push/poll/pull.
4. Hiển thị result.

## Lưu ý an toàn

- `RUN_REMOTE = False` mặc định để tránh tiêu Kaggle quota.
- Chỉ bật `RUN_REMOTE = True` khi đã kiểm tra account, kernel slug, input dataset và `.nemo` resume.

