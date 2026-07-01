"""Quét cây LibriSpeech (.flac + .trans.txt) -> manifest NeMo (.jsonl).

Mỗi dòng manifest: {"audio_filepath", "duration", "text"}.
Dùng soundfile.info để lấy duration mà không phải decode toàn bộ file (nhanh).

Chạy: uv run python src/build_librispeech_manifest.py <thu_muc_LibriSpeech> <out.jsonl>
"""

import json
import sys
from pathlib import Path

import soundfile as sf


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n = 0
    with out.open("w", encoding="utf-8") as fout:
        # Mỗi file .trans.txt chứa transcript cho mọi utterance trong cùng chapter
        for trans in sorted(root.rglob("*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                utt_id, _, text = line.partition(" ")
                flac = trans.parent / f"{utt_id}.flac"
                if not flac.exists():
                    continue
                info = sf.info(str(flac))
                duration = info.frames / info.samplerate
                fout.write(json.dumps({
                    "audio_filepath": str(flac.resolve()),
                    "duration": round(duration, 3),
                    # LibriSpeech transcript viết HOA; hạ về thường để khớp output model
                    "text": text.lower(),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"Đã ghi {n} dòng vào {out}")


if __name__ == "__main__":
    main()
