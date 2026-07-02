from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from asr_lab.model.fastconformer import CommonVoiceRunConfig, FastConformerRunConfig


def uv_module(module: str, *args: str) -> list[str]:
    return ["uv", "run", "python", "-m", module, *args]


def run_cmd(args: list[str], main_root: Path, run_remote: bool = False, check: bool = True):
    print("$", " ".join(shlex.quote(str(arg)) for arg in args))
    if not run_remote:
        print("CHẠY THỬ: truyền run_remote=True để thực thi lệnh.")
        return None
    return subprocess.run(args, cwd=main_root, text=True, check=check)


def build_code_dataset(config: FastConformerRunConfig | CommonVoiceRunConfig, main_root: Path, run_remote: bool):
    return run_cmd(
        uv_module("asr_lab.deploy.kaggle", "build", "--account", config.account),
        main_root,
        run_remote,
    )


def push_vivos_finetune(config: FastConformerRunConfig, main_root: Path, run_remote: bool):
    return run_cmd(
        uv_module(
            "asr_lab.deploy.kaggle",
            "push",
            "--account",
            config.account,
            "--gpu",
            "--as",
            config.kernel,
            "--module",
            "asr_lab.train.finetune_vivos",
            "--script-args",
            config.script_args(),
        ),
        main_root,
        run_remote,
    )


def poll_kernel(account: str, kernel: str, main_root: Path, run_remote: bool):
    return run_cmd(
        uv_module("asr_lab.deploy.kaggle", "poll", "--account", account, "--kernel", kernel),
        main_root,
        run_remote,
        check=False,
    )


def pull_kernel(account: str, kernel: str, main_root: Path, run_remote: bool):
    return run_cmd(
        uv_module("asr_lab.deploy.kaggle", "pull", "--account", account, "--kernel", kernel),
        main_root,
        run_remote,
    )


def upload_resume_dataset(config: CommonVoiceRunConfig, nemo_path: Path, main_root: Path, run_remote: bool):
    if not Path(nemo_path).exists():
        print("Missing resume .nemo:", nemo_path)
        return None
    return run_cmd(
        uv_module(
            "asr_lab.deploy.kaggle",
            "upload-data",
            "--account",
            config.account,
            "--file",
            str(nemo_path),
            "--as",
            config.resume_dataset,
        ),
        main_root,
        run_remote,
    )


def push_common_voice_resume(config: CommonVoiceRunConfig, main_root: Path, run_remote: bool):
    return run_cmd(
        uv_module(
            "asr_lab.deploy.kaggle",
            "push",
            "--account",
            config.account,
            "--gpu",
            "--as",
            config.kernel,
            "--module",
            "asr_lab.train.continue_vi",
            "--script-args",
            config.script_args(),
            "--input-dataset",
            config.resume_dataset,
        ),
        main_root,
        run_remote,
    )
