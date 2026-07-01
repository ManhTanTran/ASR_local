"""Tải split VIVOS từ HuggingFace -> wav 16kHz mono + manifest NeMo.

VIVOS = corpus đọc tiếng Việt (ĐHQG TP.HCM). Bản gốc `AILAB-VNUHCM/vivos` dùng loading-script
(bị bỏ ở datasets v5) -> dùng mirror PARQUET `ademax/vivos-vie-speech2text` (train 11.420 + test
1.000, cột `audio` + `transcription`). Đủ cho đo WER baseline + dữ liệu fine-tune.

Tránh phụ thuộc torchcodec (datasets v5): tắt auto-decode (Audio(decode=False)) rồi tự đọc
bytes bằng soundfile. Cột text là `transcription` (fallback sentence/text cho chắc).

Chạy: uv run python src/load_vivos.py --split test [--n 0 (0 = lấy hết)]
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

HF_ID = "ademax/vivos-vie-speech2text"  # mirror parquet (bản gốc dùng script, bỏ ở datasets v5)
TARGET_SR = 16000
TEXT_KEYS = ("transcription", "sentence", "text")  # mirror dùng 'transcription'; thử lần lượt cho bền


def to_16k_mono(arr: np.ndarray, sr: int) -> np.ndarray:
    if arr.ndim > 1:  # stereo -> mono
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    return arr.astype("float32")


def pick_text(row: dict) -> str:
    for k in TEXT_KEYS:
        if k in row and row[k]:
            return str(row[k]).strip()
    raise KeyError(f"Không thấy cột text trong {list(row.keys())}")


def dump_split(split: str, out_wav: Path, manifest: Path, n: int = 0, hf_id: str = HF_ID) -> Path:
    """Tải 1 split VIVOS -> ghi wav 16kHz mono + manifest NeMo. Trả về đường dẫn manifest.

    Hàm dùng chung cho cả CLI (đo WER) lẫn fine-tune (chuẩn bị train/val/test) — tránh lặp code.
    """
    out_wav = Path(out_wav)
    manifest = Path(manifest)
    out_wav.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    sel = f"{split}[:{n}]" if n > 0 else split
    ds = load_dataset(hf_id, split=sel)
    ds = ds.cast_column("audio", Audio(decode=False))  # bytes thô, không cần torchcodec

    cnt = 0
    with manifest.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            arr = to_16k_mono(arr, sr)
            wav = out_wav / f"{i:05d}.wav"
            sf.write(str(wav), arr, TARGET_SR)
            fout.write(json.dumps({
                "audio_filepath": str(wav.resolve()),
                "duration": round(len(arr) / TARGET_SR, 3),
                "text": pick_text(row),
            }, ensure_ascii=False) + "\n")
            cnt += 1
    total = sum(json.loads(l)["duration"] for l in manifest.open(encoding="utf-8"))
    print(f"VIVOS {split}: {cnt} mẫu, {total/60:.1f} phút audio -> {manifest}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--n", type=int, default=0, help="số mẫu (0 = lấy hết split)")
    args = ap.parse_args()
    dump_split(args.split, Path(f"data/raw/vivos/{args.split}"),
               Path(f"data/manifests/vivos_{args.split}.jsonl"), n=args.n)


if __name__ == "__main__":
    main()
