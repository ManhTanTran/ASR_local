"""Upload / cập nhật model ASR VI lên HuggingFace Hub — mặc định PRIVATE (dùng nội bộ team).

Chạy (từ máy CÓ file .nemo + đã đăng nhập HF hoặc set HF_TOKEN):
  python deploy/hf/publish.py \
      --nemo /duong/dan/s3-fc115m-full.nemo \
      --repo kyle/vi-asr-fastconformer-114m \
      --card deploy/hf/model_card.md \
      --version v1

Cập nhật weight mới về sau: chạy lại với `--nemo <bản mới> --version v2`
  -> repo GIỮ nguyên, thêm 1 commit + tag mới; bản cũ vẫn truy được qua git history/tag.

Token: đọc từ HF login cache hoặc env HF_TOKEN. KHÔNG truyền token qua CLI (tránh lộ vào log/history).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True, help="đường dẫn file .nemo cần up")
    ap.add_argument("--repo", required=True, help="vd kyle/vi-asr-fastconformer-114m")
    ap.add_argument("--card", default=None, help="model_card.md -> upload thành README.md")
    ap.add_argument("--version", default=None, help="tag phiên bản, vd v1 (bỏ qua nếu trùng)")
    ap.add_argument("--public", action="store_true", help="MẶC ĐỊNH private; cờ này mới public")
    ap.add_argument("--filename", default=None, help="tên .nemo trên repo (mặc định = basename)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in việc sẽ làm, KHÔNG upload")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    nemo = Path(args.nemo)
    if not nemo.is_file():
        sys.exit(f"[hf] không thấy .nemo: {nemo}")
    fname = args.filename or nemo.name
    private = not args.public
    size_mb = nemo.stat().st_size / 1e6

    api = HfApi()
    who = api.whoami()  # xác nhận token hợp lệ trước khi làm gì
    print(f"[hf] user={who.get('name')} repo={args.repo} private={private} "
          f"file={fname} ({size_mb:.0f}MB) version={args.version}", flush=True)

    if args.dry_run:
        print("[hf] --dry-run: DỪNG, không tạo repo / không upload.", flush=True)
        return

    api.create_repo(args.repo, repo_type="model", private=private, exist_ok=True)
    print(f"[hf] repo sẵn sàng (private={private})", flush=True)

    if args.card:
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model",
                        commit_message=f"card: {args.version or 'update'}")
        print("[hf] upload README.md (model card)", flush=True)

    print(f"[hf] upload {fname} ({size_mb:.0f}MB) — LFS, có thể lâu...", flush=True)
    api.upload_file(path_or_fileobj=str(nemo), path_in_repo=fname,
                    repo_id=args.repo, repo_type="model",
                    commit_message=f"weight: {fname} {args.version or ''}".strip())
    print(f"[hf] upload {fname} xong", flush=True)

    if args.version:
        try:
            api.create_tag(args.repo, tag=args.version, repo_type="model")
            print(f"[hf] tag {args.version}", flush=True)
        except Exception as e:  # noqa: BLE001 — tag trùng không nên chặn
            print(f"[hf] tag {args.version} bỏ qua ({type(e).__name__}: {str(e)[:80]})", flush=True)

    print(f"[hf] XONG -> https://huggingface.co/{args.repo} (private={private})", flush=True)


if __name__ == "__main__":
    main()
