"""Load TEST SPLIT một subset của `hf-audio/esb-datasets-test-only-sorted` từ HuggingFace
-> dump file wav 16kHz mono + manifest NeMo (.jsonl), để chạy bench WER trên CPU.

Tránh phụ thuộc torchcodec (datasets v5): tắt auto-decode (Audio(decode=False)) rồi
tự đọc bytes bằng soundfile. Resample về 16kHz nếu cần.

Chạy: uv run python src/load_hf_testset.py <subset> [--n 50]
  subset: ami | earnings22 | voxpopuli | librispeech | tedlium ...
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

HF_ID = "hf-audio/esb-datasets-test-only-sorted"
TARGET_SR = 16000


def to_16k_mono(arr: np.ndarray, sr: int) -> np.ndarray:
    if arr.ndim > 1:  # stereo -> mono
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    return arr.astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subset")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--slice", default=None,
                    help="biểu thức split tuỳ chọn, vd 'test[-40:]' (utt ngắn nhất vì bộ sort dài->ngắn)")
    args = ap.parse_args()

    out_wav = Path(f"data/raw/hf_{args.subset}")
    out_wav.mkdir(parents=True, exist_ok=True)
    manifest = Path(f"data/manifests/hf_{args.subset}.jsonl")

    split = args.slice or f"test[:{args.n}]"
    ds = load_dataset(HF_ID, args.subset, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))  # lấy bytes thô, không cần torchcodec

    n = 0
    with manifest.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            raw = row["audio"]["bytes"]
            arr, sr = sf.read(io.BytesIO(raw))
            arr = to_16k_mono(arr, sr)
            wav = out_wav / f"{i:04d}.wav"
            sf.write(str(wav), arr, TARGET_SR)
            fout.write(json.dumps({
                "audio_filepath": str(wav.resolve()),
                "duration": round(len(arr) / TARGET_SR, 3),
                "text": row["text"],
            }, ensure_ascii=False) + "\n")
            n += 1
    total = sum(json.loads(l)["duration"] for l in manifest.open())
    print(f"{args.subset}: {n} mẫu, {total/60:.1f} phút audio -> {manifest}")


if __name__ == "__main__":
    main()
