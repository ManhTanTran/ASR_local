"""VIVOS data helpers for NeMo ASR manifests."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from asr_lab.common.metrics import normalize_vi

HF_ID = "ademax/vivos-vie-speech2text"
TARGET_SR = 16000
TEXT_KEYS = ("transcription", "sentence", "text")


@dataclass(frozen=True)
class VivosManifests:
    train: Path
    val: Path
    test: Path
    train_rows: int
    val_rows: int
    test_rows: int

    def as_dict(self) -> dict[str, str]:
        return {
            "train": str(self.train),
            "val": str(self.val),
            "test": str(self.test),
        }


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def to_16k_mono(arr: np.ndarray, sr: int) -> np.ndarray:
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        arr = librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    return arr.astype("float32")


def pick_text(row: dict) -> str:
    for key in TEXT_KEYS:
        if key in row and row[key]:
            return str(row[key]).strip()
    raise KeyError(f"No text column found in {list(row.keys())}")


def dump_split(split: str, out_wav: Path, manifest: Path, n: int = 0, hf_id: str = HF_ID) -> Path:
    """Download one VIVOS split, write 16 kHz mono wav files and a NeMo manifest."""
    out_wav = Path(out_wav)
    manifest = Path(manifest)
    out_wav.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    selected_split = f"{split}[:{n}]" if n > 0 else split
    dataset = load_dataset(hf_id, split=selected_split)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    count = 0
    with manifest.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(dataset):
            arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            arr = to_16k_mono(arr, sr)
            wav = out_wav / f"{idx:05d}.wav"
            sf.write(str(wav), arr, TARGET_SR)
            fout.write(
                json.dumps(
                    {
                        "audio_filepath": str(wav.resolve()),
                        "duration": round(len(arr) / TARGET_SR, 3),
                        "text": pick_text(row),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    total = sum(json.loads(line)["duration"] for line in manifest.open(encoding="utf-8"))
    print(f"VIVOS {split}: {count} samples, {total / 60:.1f} audio minutes -> {manifest}")
    return manifest


def prepare_vivos_manifests(
    data_dir: Path,
    train_n: int = 0,
    val_n: int = 300,
    test_n: int = 0,
) -> VivosManifests:
    """Prepare train/val/test VIVOS manifests and normalize train/val text."""
    data_dir = Path(data_dir)
    full_train = dump_split(
        "train",
        data_dir / "raw" / "train",
        data_dir / "manifests" / "train_full.jsonl",
        n=train_n,
    )
    test_manifest = dump_split(
        "test",
        data_dir / "raw" / "test",
        data_dir / "manifests" / "test.jsonl",
        n=test_n,
    )

    rows = read_manifest(full_train)
    for row in rows:
        row["text"] = normalize_vi(row["text"])

    holdout = min(val_n, max(1, len(rows) // 10))
    train_rows = rows[:-holdout]
    val_rows = rows[-holdout:]
    train_manifest = data_dir / "manifests" / "train.jsonl"
    val_manifest = data_dir / "manifests" / "val.jsonl"
    _write_jsonl(train_manifest, train_rows)
    _write_jsonl(val_manifest, val_rows)

    return VivosManifests(
        train=train_manifest,
        val=val_manifest,
        test=test_manifest,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        test_rows=len(read_manifest(test_manifest)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--n", type=int, default=0, help="number of samples; 0 means the full split")
    args = parser.parse_args()
    dump_split(
        args.split,
        Path(f"data/raw/vivos/{args.split}"),
        Path(f"data/manifests/vivos_{args.split}.jsonl"),
        n=args.n,
    )


if __name__ == "__main__":
    main()
