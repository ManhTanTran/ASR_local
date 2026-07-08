"""Nạp/khởi tạo model NeMo ASR theo cfg.model: from_pretrained (NGC/HF) hoặc restore_from (.nemo nấc trước),
đổi vocab (change_vocabulary) khi cần, freeze encoder tuỳ chọn.
"""
from __future__ import annotations

from pathlib import Path


def load_model(cfg, tokenizer_dir: str | None, cuda: bool):
    import nemo.collections.asr as nemo_asr
    src = str(cfg.model.init_from)
    loc = "cuda" if cuda else "cpu"

    # .nemo nấc trước -> restore (giữ tokenizer/decoder); tên NGC/HF -> from_pretrained
    if src.endswith(".nemo") or Path(src).is_file():
        model = nemo_asr.models.ASRModel.restore_from(src, map_location=loc)
        print(f"[model] restore_from {src}", flush=True)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(src, map_location=loc)
        print(f"[model] from_pretrained {src}", flush=True)

    if cfg.model.get("change_vocabulary"):
        if not tokenizer_dir:
            raise SystemExit("change_vocabulary=true nhưng không có tokenizer_dir")
        model.change_vocabulary(new_tokenizer_dir=str(tokenizer_dir),
                                new_tokenizer_type=str(cfg.model.tokenizer.type))
        print(f"[model] change_vocabulary -> {tokenizer_dir} ({model.tokenizer.vocab_size} token)", flush=True)

    if cfg.model.get("freeze_encoder"):
        model.encoder.freeze()
        print("[model] freeze encoder — chỉ train decoder+joint", flush=True)

    # change_vocabulary dựng decoder+joint MỚI trên CPU -> phải đẩy lại cả model về cuda,
    # nếu không encoder(cuda) vs joint(cpu) lệch device khi eval/transcribe (đã gặp ở GPU smoke).
    if cuda:
        model = model.cuda()
    return model
