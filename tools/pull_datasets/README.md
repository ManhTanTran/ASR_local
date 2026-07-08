# tools/pull_datasets — kéo dataset ASR tiếng Việt về DGX

Kéo các public dataset trong [`datasets.yaml`](datasets.yaml) về `/srv/team-share/datasets/asr_vi/`
trên DGX, làm **tài sản data dùng chung** cho lab QASI.

## Cách chạy (trên DGX)

```bash
# env login-shell DGX đã set HF_HOME=/srv/team-share/cache/hf
cd /srv/team-share/datasets/asr_vi/_tool   # nơi copy tool

# xem kế hoạch (không tải)
uv run --no-project --with huggingface_hub --with pyyaml python pull.py --dry-run

# kéo toàn bộ ungated stage 1..3 (auto, chịu lỗi, resume)
uv run --no-project --with huggingface_hub --with pyyaml python pull.py --stages 1,2,3

# chạy nền qua đêm (sống sau khi ngắt ssh)
setsid bash -c 'uv run --no-project --with huggingface_hub --with pyyaml \
    python pull.py --stages 1,2,3 > pull.log 2>&1' &
tail -f pull.log
```

## Đặc tính an toàn cho chạy-qua-đêm
- **Resume**: file dở tự nối; dataset đã xong có cờ `<name>/.done` → lần sau bỏ qua.
- **Chịu lỗi**: một dataset fail chỉ ghi log rồi chạy tiếp cái sau.
- **Theo dõi**: `_pull_status.json` (state/gb/seconds/error mỗi dataset) + `pull.log`.

## Dataset gated (KHÔNG auto)
`phoaudiobook`, `vivoice`, `vietmed` cần **chấp nhận điều khoản trên HF + token**:
```bash
uv run ... python pull.py --only vivoice --include-gated --token <HF_TOKEN>
```

## Sau khi kéo xong
Data thô (parquet/audio) nằm ở `asr_vi/<name>/`. Bước tiếp = build manifest NeMo
(`{audio_filepath,duration,text}`) trỏ vào parquet local — xem `docs/07_dgx_training/`.
