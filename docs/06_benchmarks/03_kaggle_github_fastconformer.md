# Chạy FastConformer trên Kaggle bằng code GitHub

Mục tiêu: tạo Kaggle Notebook, clone code từ GitHub, train FastConformer trên GPU Kaggle và kiểm tra `results.json` có cùng hướng với bản main.

## Repo dùng để chạy

Code Kaggle sẽ clone từ repo public của bạn:

```text
https://github.com/ManhTanTran/ASR_local.git
```

Notebook local của bạn nằm ở:

```text
notebooks/final/fastconformer/02_fastconformer_main.ipynb
```

Notebook này không dùng `asr_lab.deploy.kaggle build/push`; nó chạy trực tiếp trên Kaggle bằng `git clone`.
Package train `asr_lab` đã được đưa vào `ASR_local/src/asr_lab` để Kaggle không cần clone repo main private.

## Step by step trên Kaggle

1. Mở Kaggle và tạo Notebook mới.
2. Bật `Accelerator = GPU`.
3. Bật `Internet = On`.
4. Upload hoặc copy nội dung notebook `02_fastconformer_main.ipynb` vào Kaggle.
5. Chạy từng cell từ trên xuống:
   - clone repo từ GitHub;
   - cài `torch==2.7.1` CUDA 11.8 và `nemo_toolkit[asr]==2.7.3`;
   - pin lại `numpy==1.26.4`, `numba==0.60.0`, `llvmlite==0.43.0` và bỏ `numba-cuda` để tránh lỗi import NeMo ASR trên image Kaggle mới;
   - verify GPU + NeMo;
   - chạy FastConformer CTC để tránh lỗi CUDA kernel của RNNT `warprnnt_numba` trên Kaggle;
   - chạy `python -u -m asr_lab.train.finetune_vivos`;
   - ghi log đầy đủ các command chính vào `run.log`, còn output notebook chỉ lọc dòng quan trọng;
   - ghi checkpoint cuối mỗi epoch vào `checkpoints/*.ckpt`;
   - đọc `results.json`;
   - liệt kê artifact.
6. Khi train xong, bấm `Save Version` hoặc `Commit` trên Kaggle để lưu output.
7. Tải output trong thư mục:

```text
/kaggle/working/runs/vivos-fc-ctc-v2norm/
```

## Xem log khi chạy run/command

Bản GitHub main hiện có helper dùng chung ở:

```text
src/asr_lab/common/run_logging.py
```

Notebook import `run_logged(...)` từ file này sau khi clone repo. Từ đó, mọi command chính như cài package, verify runtime, train model, eval hoặc command khác đều có thể ghi stdout/stderr vào cùng một file log:

```text
/kaggle/working/runs/vivos-fc-ctc-v2norm/run.log
```

Riêng các script train Lightning có metric callback dùng chung. Mặc định notebook dùng:

```text
--console-log-steps 0
```

Nghĩa là không spam từng step; sau mỗi epoch chỉ hiện một dòng gọn dạng:

```text
$ python -u -m asr_lab.train.finetune_vivos ...
[epoch] epoch=0 step=695 train_loss=... val_loss=... val_wer=...
```

Nếu output của Kaggle không tự cuộn xuống dòng mới, chạy cell `Xem log run mới nhất` trong notebook để xem đoạn cuối `run.log`.

Nếu command lỗi, notebook chỉ hiện `Command failed with exit code ... Full log: ...`; không bung traceback `CalledProcessError` dài. Full traceback vẫn nằm trong `run.log` để debug khi cần.

Với notebook/script khác, chỉ cần import và gọi:

```python
from asr_lab.common.run_logging import run_logged

run_logged(cmd, cwd=repo_dir, env=env, log_path=run_dir / "run.log")
```

## Checkpoint và resume khi hết GPU

Bản GitHub main hiện có helper dùng chung ở:

```text
src/asr_lab/common/checkpointing.py
```

Notebook/script sẽ lưu checkpoint Lightning trong lúc train:

```text
/kaggle/working/runs/vivos-fc-ctc-v2norm/checkpoints/*.ckpt
/kaggle/working/runs/vivos-fc-ctc-v2norm/checkpoints/checkpoint_manifest.json
```

Mặc định notebook dùng:

```text
checkpoint_steps: 0
checkpoint_keep: 2
auto_resume: true
```

Nếu Kaggle hết GPU trước khi sinh `.nemo`, bấm `Save Version` để giữ output. Lần chạy sau, add output version cũ làm input cho notebook. Script sẽ tự tìm checkpoint mới nhất của cùng `run_id` trong:

```text
/kaggle/input/**/runs/vivos-fc-ctc-v2norm/checkpoints/*.ckpt
```

Nếu muốn ép resume từ một file cụ thể, đặt `RESUME_FROM_CHECKPOINT` trong notebook hoặc truyền:

```text
--resume-from-checkpoint "/kaggle/input/.../*.ckpt"
```

## Config bản main

```text
pretrained: nvidia/stt_en_fastconformer_ctc_large
run_id: vivos-fc-ctc-v2norm
epochs: 50
batch: 16
vocab_size: 1024
lr: 2e-4
precision: 32
max_minutes: 660
console_log_steps: 0
checkpoint_steps: 0
checkpoint_keep: 2
fused_batch_size: 0
```

Ghi chú runtime: bản RNNT `nvidia/stt_en_fastconformer_transducer_large` đã fail trên Kaggle ở train step đầu với `TypeError: Signature mismatch: 2 argument types given, but function takes 1 arguments` trong `warprnnt_numba`, kể cả khi giảm batch và tắt fused mode. Bản main vì vậy dùng CTC FastConformer 115M để vẫn fine-tune FastConformer nhưng không phụ thuộc RNNT CUDA kernel.

Nếu Kaggle image kéo `numpy 2.x + numba-cuda`, NeMo ASR có thể lỗi khi import. Notebook main đã có bước fix dependency trước khi verify:

```text
pip uninstall -y numba-cuda
pip install --force-reinstall numpy==1.26.4 numba==0.60.0 llvmlite==0.43.0
```

## File kết quả cần kiểm tra

```text
run.log
results.json
status.json
checkpoints/checkpoint_manifest.json
checkpoints/*.ckpt
metrics.csv
fastconformer_vivos_ft.nemo
```

Nếu `results.json` được sinh ra và WER giảm mạnh so với baseline English-only, nghĩa là luồng GitHub -> Kaggle -> train -> artifact đã chạy đúng. Nếu muốn so với bản main local, tải thư mục run về `ASR/artifacts/runs/vivos-fc-ctc-v2norm/` rồi chạy report/compare trong repo main.
