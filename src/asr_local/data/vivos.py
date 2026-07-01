from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from asr_local.common.metrics import normalize_vi


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


def prepare_vivos_manifests(
    data_dir: Path,
    train_n: int = 0,
    val_n: int = 300,
    test_n: int = 0,
) -> VivosManifests:
    """Download/prepare VIVOS using main repo loader, then normalize train/val text."""
    from asr_lab.data.vivos import dump_split

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

