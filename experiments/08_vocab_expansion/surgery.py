"""Cửa 1 — phẫu thuật vocab expansion: S3 .nemo -> s3-vocabexp.nemo (thêm f/j/w/z, GIỮ trọng số cũ).

Thiết kế: [01_surgery_design.md]. Không train. Sau chạy phải eval 9 test để verify WER ≈ S3.

Chạy trên DGX:
  cd /srv/team-share/projects/nvidia_asr_nemo
  .venv/bin/python experiments/08_vocab_expansion/surgery.py \
      --in-nemo  /srv/team-share/models/asr_vi/s3-fc115m-full.nemo \
      --out-nemo /srv/team-share/models/asr_vi/s3-vocabexp.nemo \
      --new-pieces f,j,w,z
"""
from __future__ import annotations

import argparse
import copy
import os
import tarfile
import tempfile


def build_expanded_tokenizer(src_nemo: str, out_dir: str, new_pieces: list[str]) -> tuple[int, int]:
    """Rút tokenizer.model từ .nemo, append pieces mới (giữ ID cũ), ghi ra out_dir.
    Trả (old_vocab, new_vocab)."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as smp

    tmp = tempfile.mkdtemp()
    with tarfile.open(src_nemo, "r") as tar:
        model_files = [n for n in tar.getnames() if n.endswith("tokenizer.model")]
        if not model_files:
            raise SystemExit("không tìm thấy tokenizer.model trong .nemo")
        tar.extract(model_files[0], tmp)
    src_model = os.path.join(tmp, model_files[0])

    mp = smp.ModelProto()
    with open(src_model, "rb") as f:
        mp.ParseFromString(f.read())
    old_vocab = len(mp.pieces)
    scores = [p.score for p in mp.pieces if p.type == 1]
    min_score = min(scores) if scores else -20.0

    existing = {p.piece for p in mp.pieces}
    added = []
    for ch in new_pieces:
        if ch in existing:
            print(f"[tokenizer] '{ch}' ĐÃ có trong vocab -> bỏ qua")
            continue
        p = mp.pieces.add()
        p.piece = ch
        p.score = min_score - 1.0     # điểm thấp: không phá segmentation câu cũ
        p.type = 1                    # NORMAL
        added.append(ch)
    new_vocab = len(mp.pieces)

    os.makedirs(out_dir, exist_ok=True)
    out_model = os.path.join(out_dir, "tokenizer.model")
    with open(out_model, "wb") as f:
        f.write(mp.SerializeToString())
    # vocab phụ (soi nhanh + NeMo có thể đọc)
    with open(os.path.join(out_dir, "tokenizer.vocab"), "w", encoding="utf-8") as fv, \
         open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as ft:
        for p in mp.pieces:
            fv.write(f"{p.piece}\t{p.score}\n")
            ft.write(p.piece + "\n")

    # verify round-trip
    sp = spm.SentencePieceProcessor()
    sp.Load(out_model)
    print(f"[tokenizer] {old_vocab} -> {new_vocab} (thêm {added})")
    for s in ["facebook", "wifi", "zalo", "fpt", "whisky", "xin chào việt nam"]:
        ids = sp.EncodeAsIds(s)
        print(f"  {s:20} ids={ids} '{sp.DecodeIds(ids)}'")
    return old_vocab, new_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-nemo", required=True)
    ap.add_argument("--out-nemo", required=True)
    ap.add_argument("--new-pieces", default="f,j,w,z")
    ap.add_argument("--bias-suppress", type=float, default=10.0,
                    help="đè bias token mới xuống min-θ (giữ cửa 1 ≈ S3)")
    args = ap.parse_args()
    new_pieces = [c for c in args.new_pieces.split(",") if c]

    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.utils import logging as _nlog
    _nlog.set_verbosity(_nlog.ERROR)

    # 1. restore S3, lưu state_dict cũ
    model = nemo_asr.models.ASRModel.restore_from(args.in_nemo, map_location="cpu")
    old_sd = copy.deepcopy(model.state_dict())
    old_V = model.tokenizer.vocab_size
    print(f"[surgery] S3 vocab={old_V}, blank_idx={old_V}")

    # 2. build tokenizer mở rộng
    exp_dir = tempfile.mkdtemp(prefix="exp_tok_")
    old_vocab, new_vocab = build_expanded_tokenizer(args.in_nemo, exp_dir, new_pieces)
    assert old_vocab == old_V, f"lệch vocab proto({old_vocab}) vs model({old_V})"
    k = new_vocab - old_V
    if k == 0:
        raise SystemExit("không thêm token nào (đã có sẵn?) — dừng")

    # 3. change_vocabulary: NeMo resize + gắn tokenizer mới (dựng decoder+joint NGẪU NHIÊN)
    model.change_vocabulary(new_tokenizer_dir=exp_dir, new_tokenizer_type="bpe")
    new_V = model.tokenizer.vocab_size
    print(f"[surgery] sau change_vocabulary: vocab={new_V}, k={k}")
    assert new_V == new_vocab, f"lệch vocab sau change ({new_V} vs {new_vocab})"

    # 4. khôi phục trọng số: cùng shape -> copy cũ; đổi shape (3 tensor V) -> ánh xạ hàng
    new_sd = model.state_dict()
    restored, surgered, copied = {}, [], 0
    with torch.no_grad():
        for name, new_t in new_sd.items():
            old_t = old_sd.get(name)
            if old_t is not None and old_t.shape == new_t.shape:
                restored[name] = old_t.clone()
                copied += 1
                continue
            if old_t is None:
                restored[name] = new_t.clone()
                print(f"[surgery] CẢNH BÁO tensor mới không có bản cũ: {name} {tuple(new_t.shape)}")
                continue
            # đổi shape -> phẫu thuật theo dim 0 (vocab+blank)
            t = new_t.clone()
            t[0:old_V] = old_t[0:old_V]                    # token cũ 0..old_V-1
            if "embed" in name:
                t[old_V:old_V + k] = old_t[0:old_V].mean(dim=0, keepdim=True)   # FVT mean cho input
            elif name.endswith(".weight"):
                t[old_V:old_V + k] = 0.0                    # joint out weight: ép im
            elif name.endswith(".bias"):
                t[old_V:old_V + k] = old_t[0:old_V].min() - args.bias_suppress   # bias thấp
            t[new_V] = old_t[old_V]                         # blank cũ (old_V) -> blank mới (new_V)
            restored[name] = t
            surgered.append((name, tuple(old_t.shape), tuple(t.shape)))

    print(f"[surgery] copy cùng-shape: {copied} tensor | phẫu thuật: {len(surgered)} tensor")
    for name, os_, ns_ in surgered:
        print(f"    {name:52} {os_} -> {ns_}")
    assert len(surgered) == 3, f"kỳ vọng 3 tensor phụ thuộc V, gặp {len(surgered)}"

    model.load_state_dict(restored, strict=True)
    print("[surgery] load_state_dict strict OK")

    # 5. save
    os.makedirs(os.path.dirname(args.out_nemo), exist_ok=True)
    model.save_to(args.out_nemo)
    print(f"[surgery] LƯU -> {args.out_nemo}")

    # verify nhanh: transcribe 1 câu callbot có loanword (nếu có audio)
    print("SURGERY_DONE")


if __name__ == "__main__":
    main()
