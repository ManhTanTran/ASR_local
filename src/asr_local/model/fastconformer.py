from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastConformerRunConfig:
    account: str = "trnmnhtn"
    kernel: str = "asr-ft-fc115m-v2norm"
    run_id: str = "vivos-fc115m-v2norm"
    model: str = "nvidia/stt_en_fastconformer_transducer_large"
    epochs: int = 50
    batch: int = 16
    vocab_size: int = 1024
    lr: str = "2e-4"
    precision: str = "32"
    max_minutes: int = 480

    def script_args(self) -> str:
        return " ".join(
            [
                "--pretrained",
                self.model,
                "--run-id",
                self.run_id,
                "--epochs",
                str(self.epochs),
                "--batch",
                str(self.batch),
                "--vocab-size",
                str(self.vocab_size),
                "--lr",
                self.lr,
                "--precision",
                self.precision,
                "--max-minutes",
                str(self.max_minutes),
            ]
        )


@dataclass(frozen=True)
class CommonVoiceRunConfig:
    account: str = "trnmnhtn"
    kernel: str = "asr-cv-fc115m"
    run_id: str = "vivos-cv"
    resume_dataset: str = "asr-v2norm-nemo"
    epochs: int = 25
    batch: int = 16
    precision: str = "32"

    def resume_nemo(self, main_root: Path, base_run_id: str) -> Path:
        return main_root / "artifacts" / "runs" / base_run_id / "nemotron_vivos_ft.nemo"

    def script_args(self) -> str:
        return " ".join(
            [
                "--resume-from",
                "/kaggle/input",
                "--run-id",
                self.run_id,
                "--epochs",
                str(self.epochs),
                "--batch",
                str(self.batch),
                "--precision",
                self.precision,
            ]
        )
