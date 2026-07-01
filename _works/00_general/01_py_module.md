# Python Module Layout

Notebook nên import code từ `src/`, không chứa toàn bộ logic train/eval inline.

## Layout hiện tại

```text
ASR_local/
  src/
    asr_local/
      common/
      data/
      model/
      train/
      deploy/
      analytics/
  notebooks/
    final/
```

## Bootstrap trong notebook

Notebook cần thêm `ASR_local/src` và `ASR/src` vào `sys.path`, sau đó import:

```python
from asr_local.paths import bootstrap

LOCAL_ROOT, MAIN_ROOT = bootstrap()
```

## Quy ước

- `common/metrics.py`: normalize, WER, extract text.
- `data/`: chuẩn bị manifest/dataset.
- `model/`: config model/run.
- `train/`: train/eval cụ thể.
- `deploy/`: wrapper Kaggle.
- `analytics/`: đọc results, compare, artifact manifest.

