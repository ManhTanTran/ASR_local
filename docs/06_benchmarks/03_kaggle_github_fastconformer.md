# Chạy FastConformer trên Kaggle bằng code GitHub

Mục tiêu: tạo Kaggle Notebook, clone code main từ GitHub, train FastConformer trên GPU Kaggle và kiểm tra `results.json` có cùng hướng với bản main.

## Repo dùng để chạy

Code train thật nằm trong repo main:

```text
https://github.com/qualphachain/nvidia_asr_nemo.git
```

Notebook local của bạn nằm ở:

```text
notebooks/final/fastconformer/02_fastconformer_main.ipynb
```

Notebook này không dùng `asr_lab.deploy.kaggle build/push`; nó chạy trực tiếp trên Kaggle bằng `git clone`.

## Step by step trên Kaggle

1. Mở Kaggle và tạo Notebook mới.
2. Bật `Accelerator = GPU`.
3. Bật `Internet = On`.
4. Upload hoặc copy nội dung notebook `02_fastconformer_main.ipynb` vào Kaggle.
5. Chạy từng cell từ trên xuống:
   - clone repo main từ GitHub;
   - cài `torch==2.7.1` CUDA 11.8 và `nemo_toolkit[asr]==2.7.3`;
   - verify GPU + NeMo;
   - chạy `python -m asr_lab.train.finetune_vivos`;
   - đọc `results.json`;
   - liệt kê artifact.
6. Khi train xong, bấm `Save Version` hoặc `Commit` trên Kaggle để lưu output.
7. Tải output trong thư mục:

```text
/kaggle/working/runs/vivos-fc115m-v2norm/
```

## Config bản main

```text
pretrained: nvidia/stt_en_fastconformer_transducer_large
run_id: vivos-fc115m-v2norm
epochs: 50
batch: 16
vocab_size: 1024
lr: 2e-4
precision: 32
max_minutes: 480
```

## File kết quả cần kiểm tra

```text
results.json
status.json
metrics.csv
nemotron_vivos_ft.nemo
```

Nếu `results.json` được sinh ra và WER giảm mạnh so với baseline English-only, nghĩa là luồng GitHub -> Kaggle -> train -> artifact đã chạy đúng. Nếu muốn so với bản main local, tải thư mục run về `ASR/artifacts/runs/vivos-fc115m-v2norm/` rồi chạy report/compare trong repo main.

