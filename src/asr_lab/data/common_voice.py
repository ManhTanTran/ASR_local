"""Tải Common Voice tiếng Việt từ HuggingFace -> wav 16kHz mono + manifest NeMo.

Dùng để GỘP vào fine-tune (mở rộng data ngoài VIVOS) — xem experiments/04_add_commonvoice.

Mirror parquet `tsdocode/common_voice_13_0_vi_pseudo_labelled` (datasets v5 OK, KHÔNG gated, KHÔNG
loading-script). Cột transcript THẬT = `sentence`; cột `whisper_transcript` là nhãn-giả -> BỎ.
Audio mp3 nhúng dạng bytes (tránh torchcodec bằng Audio(decode=False), decode thẳng bằng soundfile).

Điểm khác VIVOS: CV có loanword (facebook/firefox/zoom) chứa f/j/w/z mà tokenizer VIVOS không phủ.
Khi DÙNG CHUNG tokenizer VIVOS phải lọc clip có ký tự ngoài vocab (charset) -> bảo đảm 0 <unk>.

Chạy: uv run python -m asr_lab.data.common_voice --split test [--n 0]
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from asr_lab.common.metrics import normalize_vi  # chuẩn hoá DUY NHẤT, khớp tokenizer
from asr_lab.data.vivos import to_16k_mono  # dùng chung resample, tránh lặp code

HF_ID = "tsdocode/common_voice_13_0_vi_pseudo_labelled"
TARGET_SR = 16000


def decode_bytes(b: bytes) -> tuple[np.ndarray, int]:
    """Decode bytes audio CV (mp3 nhúng) bằng soundfile.

    libsndfile >= 1.1 (gói sẵn trong wheel `soundfile`) đọc thẳng mp3 -> KHÔNG cần librosa/ffmpeg.
    Resample 48kHz->16k để cho `to_16k_mono` (librosa, thuần Python). sr trả nguyên gốc."""
    return sf.read(io.BytesIO(b))


def dump_split(split: str, out_wav: Path, manifest: Path, charset: set[str] | None = None,
               n: int = 0, hf_id: str = HF_ID) -> Path:
    """Tải 1 split CV -> wav 16kHz mono + manifest NeMo (text đã normalize_vi). Trả về manifest.

    charset != None (DÙNG cho train): drop clip có ký tự normalize ngoài tập charset (tokenizer VIVOS)
    -> 0 <unk>. charset None (DÙNG cho test): giữ hết, đo WER thật kể cả loanword model spell sai.
    """
    out_wav = Path(out_wav); manifest = Path(manifest)
    out_wav.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    sel = f"{split}[:{n}]" if n > 0 else split
    ds = load_dataset(hf_id, split=sel)
    ds = ds.cast_column("audio", Audio(decode=False))  # bytes thô, không cần torchcodec

    kept = dropped = empty = 0
    with manifest.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            raw = row.get("sentence") or ""           # transcript THẬT (không phải whisper_transcript)
            text = normalize_vi(raw)
            if not text:
                empty += 1
                continue                              # bỏ câu rỗng -> tránh nhiễu
            if charset is not None and any(c != " " and c not in charset for c in text):
                dropped += 1
                continue                              # clip có ký tự ngoài vocab VIVOS -> drop (train)
            arr, sr = decode_bytes(row["audio"]["bytes"])
            arr = to_16k_mono(arr, sr)
            wav = out_wav / f"{i:05d}.wav"
            sf.write(str(wav), arr, TARGET_SR)
            fout.write(json.dumps({
                "audio_filepath": str(wav.resolve()),
                "duration": round(len(arr) / TARGET_SR, 3),
                "text": text,
            }, ensure_ascii=False) + "\n")
            kept += 1
    total = sum(json.loads(l)["duration"] for l in manifest.open(encoding="utf-8"))
    print(f"CV {split}: giữ {kept} (drop ngoài-vocab {dropped}, rỗng {empty}) | "
          f"{total/60:.1f} phút audio -> {manifest}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=0, help="số mẫu (0 = lấy hết split)")
    args = ap.parse_args()
    dump_split(args.split, Path(f"data/raw/common_voice_vi/{args.split}"),
               Path(f"data/manifests/common_voice_{args.split}.jsonl"), n=args.n)


if __name__ == "__main__":
    main()
