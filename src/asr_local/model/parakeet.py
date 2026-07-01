from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParakeetConfig:
    model_repo: str = "nvidia/parakeet-ctc-0.6b-Vietnamese"
    model_file: str = "parakeet-ctc-0.6b-vi.nemo"
    run_id: str = "parakeet-vivos-decoder-only"
    train_n: int = 0
    val_n: int = 300
    test_n: int = 0
    max_epochs: int = 10
    train_batch: int = 1
    accumulate_grad_batches: int = 32
    eval_batch: int = 8
    lr: float = 1e-5


def resolve_device() -> str:
    import torch

    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    return "cuda" if torch.cuda.is_available() else "cpu"


def download_parakeet(config: ParakeetConfig) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=config.model_repo, filename=config.model_file))


def run_paths(work_root: Path, config: ParakeetConfig) -> dict[str, Path]:
    run_dir = Path(work_root) / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "finetuned_model": run_dir / "parakeet_vivos_decoder_only.nemo",
        "results_json": run_dir / "results.json",
        "error_csv": run_dir / "vivos_test_error_analysis.csv",
    }

