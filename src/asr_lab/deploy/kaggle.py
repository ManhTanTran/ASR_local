"""Adapter deploy lab ASR lên Kaggle — Kaggle TRAIN (free GPU), local điều phối + kéo artifact.

Clone cơ chế từ numerai lab_v2 (deployment/kaggle.py), chế lại cho ASR:
  - Kernel bật GPU (fine-tune NeMo 0.6B cần GPU; máy local chỉ CPU).
  - Code đẩy lên CODE-DATASET (src/ vài chục KB) — kernel đọc từ /kaggle/input, KHÔNG cần GitHub.
    (Đổi từ GitHub-pull sang code-dataset để không phụ thuộc git push.)
  - KHÔNG `pip install -e` repo (pyproject pin torch-CPU sẽ phá CUDA của Kaggle) — chỉ thêm thư mục
    chứa `asr_lab` vào PYTHONPATH + cài `nemo_toolkit[asr]` trên kernel (giữ torch+CUDA sẵn của Kaggle).
  - Data: script tự tải VIVOS từ HuggingFace trên mạng Kaggle (internet bật).
  - Artifact: run-dir (/kaggle/working/runs/<id>) có status.json + results.json + .nemo. pull -> artifacts/runs/.

Lệnh (account qua deploy/kaggle/accounts.json + ~/.kaggle/<slug>/kaggle.json):
    uv run python -m asr_lab.deploy.kaggle accounts
    uv run python -m asr_lab.deploy.kaggle build  --account kyhoolee      # đẩy src/ -> code-dataset
    uv run python -m asr_lab.deploy.kaggle smoke  --account kyhoolee --gpu
    uv run python -m asr_lab.deploy.kaggle push   --account kyhoolee --gpu --as asr-ft-vivos \
        --module asr_lab.train.finetune_vivos --script-args "--epochs 40"
    uv run python -m asr_lab.deploy.kaggle poll   --account kyhoolee --kernel asr-ft-vivos
    uv run python -m asr_lab.deploy.kaggle pull   --account kyhoolee --kernel asr-ft-vivos
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# .../src/asr_lab/deploy/kaggle.py → repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = REPO_ROOT / "deploy" / "kaggle"
ACCOUNTS_JSON = DEPLOY_DIR / "accounts.json"
CODE_DATASET = "nvidia-asr-nemo-src"   # code-dataset tí hon (src/) — kernel đọc thay vì clone GitHub
NEMO_PIP = "nemo_toolkit[asr]==2.7.3"   # khớp pyproject; cài trên kernel, giữ torch+CUDA của Kaggle


# ------------------------------ accounts + kaggle CLI ------------------------------

def kaggle_bin() -> str:
    # Ưu tiên kaggle trong .venv của repo này; fallback venv numerai (đã có sẵn) rồi PATH.
    cands = [
        REPO_ROOT / ".venv/bin/kaggle",
        Path.home() / "work/startup/_1_dagfl_/numerai-ml-explore/numerai/lab_v2/.venv/bin/kaggle",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return "kaggle"


KAGGLE = kaggle_bin()


def guard_access_token():
    """Bẫy: ~/.kaggle/access_token được CLI đọc bằng đường-dẫn-cứng → hijack mọi account."""
    for name in ("access_token", "access_token.txt"):
        p = Path(os.path.expanduser(f"~/.kaggle/{name}"))
        if p.exists():
            sys.exit(f"LỖI: tồn tại {p} → hijack mọi account. Sửa: mv {p} {p}.bak")


def load_accounts() -> dict:
    if not ACCOUNTS_JSON.exists():
        sys.exit(f"Chưa có {ACCOUNTS_JSON}. Tạo từ accounts.example.json.")
    raw = json.loads(ACCOUNTS_JSON.read_text())
    return {a["slug"]: Path(os.path.expanduser(a["config_dir"])) for a in raw["accounts"]}


def account_env(config_dir: Path) -> dict:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    return env


def username_of(config_dir: Path) -> str:
    kj = config_dir / "kaggle.json"
    if not kj.exists():
        sys.exit(f"Không thấy {kj}")
    return json.loads(kj.read_text())["username"]


def run(cmd: list, env: dict, **kw) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], env=env, **kw)


# ------------------------------ sinh notebook (test được offline) ------------------------------

def _nb(cells: list[str]) -> dict:
    """Bọc danh sách source-code-cell thành ipynb tối thiểu hợp lệ."""
    return {
        "cells": [
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": src.splitlines(keepends=True)}
            for src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


# Định vị package asr_lab trong code-dataset (/kaggle/input/...). KHÔNG pip install -e (tránh phá
# torch-CUDA của Kaggle) — chỉ thêm thư mục CHỨA asr_lab vào path để `import asr_lab` chạy.
_LOCATE = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "_ft = next(Path('/kaggle/input').rglob('finetune_vivos.py'))  # .../src/asr_lab/train/finetune_vivos.py\n"
    "PKG_PARENT = _ft.parents[2]  # train->asr_lab->src : thư mục chứa package asr_lab\n"
    "sys.path.insert(0, str(PKG_PARENT))\n"
    "print('PKG_PARENT =', PKG_PARENT, '| has asr_lab:', (PKG_PARENT / 'asr_lab').is_dir())\n"
)

# Cài torch cu118 (phủ sm_60 P100 + sm_75 T4) TRƯỚC — Kaggle mặc định torch cu128 ĐÃ BỎ kernel
# sm_60 -> RNNT LSTM trên P100 chết 'no kernel image'. Rồi cài nemo (chỉ cần torch>=2.6, không đổi).
# KHÔNG import/reload torch trong kernel (reload -> double-register TORCH_LIBRARY triton -> crash):
# verify bằng SUBPROCESS, chạy thử LSTM-trên-cuda để bắt lỗi sm NGAY (không để chết giữa training).
_PIP_NEMO = (
    "import subprocess, sys\n"
    "print('vá torch+vision+audio 2.7.1/0.22.1 cu118 (P100 sm_60 + T4; khớp ABI)...', flush=True)\n"
    "subprocess.run([sys.executable,'-m','pip','-q','install','torch==2.7.1','torchvision==0.22.1',"
    "'torchaudio==2.7.1','--index-url','https://download.pytorch.org/whl/cu118'], check=True)\n"
    "print('cài " + NEMO_PIP + " ...', flush=True)\n"
    "subprocess.run([sys.executable,'-m','pip','-q','install','" + NEMO_PIP + "'], check=True)\n"
    "_chk = ('import torch,nemo;'\n"
    "        'print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"nemo\",nemo.__version__);'\n"
    "        'l=torch.nn.LSTM(8,8).cuda();x=torch.randn(2,3,8).cuda();_=l(x);print(\"LSTM-cuda-OK\")')\n"
    "_v = subprocess.run([sys.executable,'-c',_chk], capture_output=True, text=True)\n"
    "print('VERIFY:', _v.stdout.strip(), (_v.stderr[-700:] if _v.returncode else ''), flush=True)\n"
    "assert _v.returncode == 0, 'torch/nemo/LSTM-cuda lỗi sau khi cài'\n"
)


# Report: liệt kê run-dir (có status.json) trong /kaggle/working — đã là output kernel nên KHÔNG
# copy (tránh nhân đôi .nemo 2.4GB); `pull` tải cả cây /kaggle/working rồi tìm status.json.
_REPORT = (
    "runs = sorted({s.parent for s in Path('/kaggle/working').rglob('status.json')})\n"
    "for r in runs:\n"
    "    mb = sum(f.stat().st_size for f in r.rglob('*') if f.is_file())/1e6\n"
    "    print('run-dir', r.relative_to('/kaggle/working'), 'size(MB)=%.1f' % mb)\n"
    "print('TONG', len(runs), 'run-dir trong output (pull se keo ve)')\n"
)


def build_script_notebook(module: str, script_args: list[str]) -> dict:
    """Notebook chạy MỘT module asr_lab: định vị code → cài nemo → python -m <module> → liệt kê run-dir.

    Script tự lo tải data (VIVOS từ HF) + ghi runs/<run_id>/status.json (+ results.json, + .nemo)
    vào /kaggle/working (đã là output kernel) để `pull` kéo về.
    """
    args_repr = repr([str(a) for a in script_args])
    run_cmd = (
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = str(PKG_PARENT)\n"
        "env['ASR_ARTIFACTS_DIR'] = '/kaggle/working'  # run-dir nằm trong output kernel\n"
        "env['ASR_DATA_DIR'] = '/tmp/vivos_data'  # data NGOÀI output (11k wav không lẫn vào pull)\n"
        "env['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'\n"
        f"cmd = [sys.executable, '-m', '{module}'] + {args_repr}\n"
        "print('RUN', ' '.join(cmd), flush=True)\n"
        "p = subprocess.run(cmd, env=env, capture_output=True, text=True)\n"
        "print(p.stdout[-8000:]); print('--- stderr ---'); print(p.stderr[-4000:])\n"
        "assert p.returncode == 0, f'script exit {p.returncode}'\n"
    )
    return _nb([_LOCATE, _PIP_NEMO, run_cmd, _REPORT])


# Smoke check chạy trong SUBPROCESS (không đụng torch/nemo trong kernel notebook).
_SMOKE_CHECK_SCRIPT = (
    "import nemo.collections.asr as nemo_asr, torch, json, time\n"
    "from pathlib import Path\n"
    "print('GPU' if torch.cuda.is_available() else 'CPU', '| nap model nho...', flush=True)\n"
    "m = nemo_asr.models.ASRModel.from_pretrained('nvidia/stt_en_conformer_ctc_small',\n"
    "    map_location='cuda' if torch.cuda.is_available() else 'cpu')\n"
    "rid = 'smoke-' + str(int(time.time()))\n"
    "rd = Path('/kaggle/working/runs') / rid; rd.mkdir(parents=True, exist_ok=True)\n"
    "(rd / 'status.json').write_text(json.dumps({'state':'ok','run_id':rid,\n"
    "    'cuda': torch.cuda.is_available(), 'note':'smoke round-trip'}))\n"
    "print('THONG LUONG OK — clone+nemo+model+run-dir. run_id=', rid)\n"
)


def build_smoke_notebook() -> dict:
    """Smoke: clone code + cài nemo + (subprocess) nạp model nhỏ + ghi run-dir → kiểm round-trip pull."""
    check = (
        "import subprocess, sys\n"
        "_chk = " + repr(_SMOKE_CHECK_SCRIPT) + "\n"
        "p = subprocess.run([sys.executable, '-c', _chk], text=True)\n"
        "assert p.returncode == 0, 'smoke check (subprocess) lỗi'\n"
    )
    return _nb([_LOCATE, _PIP_NEMO, check, _REPORT])


# ------------------------------ lệnh ------------------------------

def _push_kernel(env: dict, user: str, kslug: str, notebook: dict, gpu: bool,
                 kernel_inputs: list[str] | None = None,
                 dataset_inputs: list[str] | None = None) -> None:
    stage = Path(tempfile.mkdtemp(prefix="kgasr-knl-"))
    (stage / "notebook.ipynb").write_text(json.dumps(notebook))
    # dataset_sources = code-dataset (src/) + mọi dataset input thêm (vd .nemo resume). slug ngắn -> '<user>/'.
    ds_sources = [f"{user}/{CODE_DATASET}"] + [d if "/" in d else f"{user}/{d}" for d in (dataset_inputs or [])]
    meta = {
        "id": f"{user}/{kslug}", "title": kslug, "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true" if gpu else "false", "enable_internet": "true",
        "dataset_sources": ds_sources,
    }
    # kernel_sources = output kernel khác làm INPUT (vd .nemo của run trước để resume) -> /kaggle/input.
    # Cho phép truyền slug ngắn ('asr-ft-fc115m-v2norm') -> tự thêm '<user>/'.
    if kernel_inputs:
        meta["kernel_sources"] = [k if "/" in k else f"{user}/{k}" for k in kernel_inputs]
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    run([KAGGLE, "kernels", "push", "-p", stage], env)
    shutil.rmtree(stage, ignore_errors=True)


def stage_code(user: str) -> Path:
    """Đóng gói src/ thành code-dataset (vài chục KB) — upload tức thì kể cả mạng VN."""
    stage = Path(tempfile.mkdtemp(prefix="kgasr-src-"))
    shutil.copytree(REPO_ROOT / "src", stage / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": CODE_DATASET, "id": f"{user}/{CODE_DATASET}",
         "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    return stage


def cmd_build(args):
    """Tạo/cập nhật code-dataset từ src/ local. Chạy lại mỗi khi sửa code trước khi push kernel."""
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    stage = stage_code(user)
    exists = run([KAGGLE, "datasets", "status", f"{user}/{CODE_DATASET}"],
                 env, capture_output=True, text=True).returncode == 0
    print(f"== Build code-dataset {user}/{CODE_DATASET} (exists={exists}) ==")
    if exists:
        run([KAGGLE, "datasets", "version", "-p", stage, "--dir-mode", "zip", "-m", "update src"], env)
    else:
        run([KAGGLE, "datasets", "create", "-p", stage, "--dir-mode", "zip"], env)
    shutil.rmtree(stage, ignore_errors=True)


def cmd_upload_data(args):
    """Đẩy 1 FILE (vd .nemo resume) thành dataset Kaggle riêng của account -> dùng làm input cho kernel.

    Cần khi resume từ ckpt mà kernel gốc (output kernel khác) PRIVATE/khác-account: thay vì kernel_sources
    cross-account (bị 403), gói .nemo thành dataset CÙNG account đang push -> đưa vào /kaggle/input."""
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"Không thấy file {src}")
    slug = args.as_slug
    stage = Path(tempfile.mkdtemp(prefix="kgasr-data-"))
    shutil.copy2(src, stage / src.name)  # Kaggle cần file thật trong thư mục stage
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = run([KAGGLE, "datasets", "status", f"{user}/{slug}"],
                 env, capture_output=True, text=True).returncode == 0
    print(f"== Upload data {user}/{slug} ({src.name}, {src.stat().st_size/1e6:.0f}MB, exists={exists}) ==")
    if exists:
        run([KAGGLE, "datasets", "version", "-p", stage, "--dir-mode", "zip", "-m", f"update {src.name}"], env)
    else:
        run([KAGGLE, "datasets", "create", "-p", stage, "--dir-mode", "zip"], env)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"-> dùng: push ... --input-dataset {slug}")


def cmd_accounts(args):
    for slug, cfg in load_accounts().items():
        ok = (cfg / "kaggle.json").exists()
        user = username_of(cfg) if ok else "??"
        p = run([KAGGLE, "config", "view"], account_env(cfg), capture_output=True, text=True)
        alive = "OK" if p.returncode == 0 else "LỖI"
        print(f"  - {slug:12s} user={user:14s} [{alive}]")


def cmd_smoke(args):
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    kslug = args.as_slug or "asr-smoke"
    gpu = getattr(args, "gpu", False)
    print(f"== Smoke push {user}/{kslug} (gpu={gpu}) ==")
    _push_kernel(env, user, kslug, build_smoke_notebook(), gpu=gpu)
    print(f"-> poll: ... poll --account {args.account} --kernel {kslug}")


def cmd_push(args):
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    if not args.as_slug:
        sys.exit("push cần --as <slug> (đặt tên kernel rõ).")
    kslug = args.as_slug[:50]
    script_args = args.script_args.split() if args.script_args else []
    gpu = getattr(args, "gpu", False)
    nb = build_script_notebook(args.module, script_args)
    kin = args.kernel_input or []
    din = args.input_dataset or []
    print(f"== Push {user}/{kslug} (module={args.module} args={script_args} gpu={gpu} "
          f"kernel_in={kin} dataset_in={din}) ==")
    _push_kernel(env, user, kslug, nb, gpu=gpu, kernel_inputs=kin, dataset_inputs=din)
    print(f"-> poll: ... poll --account {args.account} --kernel {kslug}")


def cmd_poll(args):
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    run([KAGGLE, "kernels", "status", f"{user}/{args.kernel}"], env)


def _write_provenance(dst: Path, **meta) -> None:
    (dst / "provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_pull(args):
    """Tải output kernel về scratch rồi đẩy mỗi run-dir (có status.json) về artifacts/runs/<id>/."""
    cfg = load_accounts()[args.account]
    user, env = username_of(cfg), account_env(cfg)
    scratch = (Path(args.dest) if args.dest
               else REPO_ROOT / "artifacts" / "_pull" / f"{args.account}-{args.kernel}")
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"== Pull {user}/{args.kernel} -> {scratch} (scratch) ==")
    run([KAGGLE, "kernels", "output", f"{user}/{args.kernel}", "-p", scratch, "-o"], env)

    runs = sorted({p.parent for p in scratch.rglob("status.json")})
    if not runs:
        print("[pull] không thấy run-dir (status.json); kernel chạy xong chưa?")
        return
    dst_root = REPO_ROOT / "artifacts" / "runs"
    dst_root.mkdir(parents=True, exist_ok=True)
    promoted, skipped = [], []
    for src in runs:
        run_id = src.name
        dst = dst_root / run_id
        if dst.exists():
            skipped.append(run_id)  # artifact bất biến, không ghi đè
            continue
        shutil.copytree(src, dst)
        _write_provenance(dst, method="kaggle", account=args.account,
                          kernel=args.kernel, source=f"{user}/{args.kernel}")
        promoted.append(run_id)
        print(f"[pull] đẩy {run_id} -> {dst}")
    if skipped:
        print(f"[pull] bỏ qua (đã có): {', '.join(sorted(set(skipped)))}")
    print(f"[pull] xong: {len(set(promoted))} run về artifacts/runs/.")


def main():
    ap = argparse.ArgumentParser(description="Deploy lab ASR lên Kaggle (chạy script fine-tune/eval).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("accounts").set_defaults(fn=cmd_accounts)
    p = sub.add_parser("build"); p.add_argument("--account", required=True); p.set_defaults(fn=cmd_build)
    p = sub.add_parser("smoke"); p.add_argument("--account", required=True)
    p.add_argument("--gpu", action="store_true"); p.add_argument("--as", dest="as_slug", default=None)
    p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("push"); p.add_argument("--account", required=True)
    p.add_argument("--module", required=True, help="module chạy bằng python -m, vd asr_lab.train.finetune_vivos.")
    p.add_argument("--script-args", default="", help="chuỗi tham số truyền cho module.")
    p.add_argument("--gpu", action="store_true", help="bật GPU (mặc định CPU).")
    p.add_argument("--kernel-input", action="append", default=None,
                   help="output kernel khác làm input (vd asr-ft-fc115m-v2norm để resume .nemo); lặp nhiều lần.")
    p.add_argument("--input-dataset", action="append", default=None,
                   help="dataset làm input (vd asr-v2norm-nemo chứa .nemo resume); slug ngắn tự thêm <user>/; lặp.")
    p.add_argument("--as", dest="as_slug", default=None); p.set_defaults(fn=cmd_push)
    p = sub.add_parser("upload-data"); p.add_argument("--account", required=True)
    p.add_argument("--file", required=True, help="đường dẫn file local (vd .nemo) để gói thành dataset.")
    p.add_argument("--as", dest="as_slug", required=True, help="slug dataset (vd asr-v2norm-nemo).")
    p.set_defaults(fn=cmd_upload_data)
    p = sub.add_parser("poll"); p.add_argument("--account", required=True)
    p.add_argument("--kernel", required=True); p.set_defaults(fn=cmd_poll)
    p = sub.add_parser("pull"); p.add_argument("--account", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--dest", default=None, help="Thư mục tải thô; mặc định artifacts/_pull/<account>-<kernel>.")
    p.set_defaults(fn=cmd_pull)
    args = ap.parse_args()
    guard_access_token()
    args.fn(args)


if __name__ == "__main__":
    main()
