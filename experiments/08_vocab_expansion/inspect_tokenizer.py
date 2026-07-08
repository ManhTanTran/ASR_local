"""Mổ SentencePiece bên trong .nemo: model_type, unk/bos/eos id, byte_fallback,
và thử proto append 4 char f/j/w/z rồi encode lại (dry-run, KHÔNG ghi đè gì).

Chạy trên DGX (venv có sentencepiece):
  .venv/bin/python experiments/08_vocab_expansion/inspect_tokenizer.py \
      --nemo /srv/team-share/models/asr_vi/s3-fc115m-full.nemo
"""
from __future__ import annotations

import argparse
import os
import tarfile
import tempfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    args = ap.parse_args()

    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as smp

    # .nemo = tar; tìm file tokenizer .model
    tmp = tempfile.mkdtemp()
    with tarfile.open(args.nemo, "r") as tar:
        names = tar.getnames()
        model_files = [n for n in names if n.endswith(".model")]
        print("=== file trong .nemo ===")
        for n in names:
            print(f"  {n}")
        print(f"\ntokenizer .model tìm thấy: {model_files}")
        for n in model_files:
            tar.extract(n, tmp)
    mp_path = os.path.join(tmp, model_files[0])

    sp = spm.SentencePieceProcessor()
    sp.Load(mp_path)
    print("\n=== SentencePieceProcessor ===")
    print(f"GetPieceSize     : {sp.GetPieceSize()}")
    print(f"unk_id / bos/eos/pad : {sp.unk_id()} / {sp.bos_id()} / {sp.eos_id()} / {sp.pad_id()}")
    for i in range(min(6, sp.GetPieceSize())):
        print(f"  piece[{i}] = '{sp.id_to_piece(i)}' (byte={sp.IsByte(i)}, ctrl={sp.IsControl(i)}, unk={sp.IsUnknown(i)})")

    # proto để đọc trainer_spec (model_type, byte_fallback)
    mp = smp.ModelProto()
    with open(mp_path, "rb") as f:
        mp.ParseFromString(f.read())
    ts = mp.trainer_spec
    print("\n=== trainer_spec ===")
    print(f"model_type   : {ts.model_type}  (1=UNIGRAM 2=BPE 3=WORD 4=CHAR)")
    print(f"vocab_size   : {ts.vocab_size}")
    print(f"byte_fallback: {ts.byte_fallback}")
    print(f"character_coverage: {ts.character_coverage}")
    print(f"num pieces trong proto: {len(mp.pieces)}")
    scores = [p.score for p in mp.pieces if p.type == 1]  # NORMAL
    if scores:
        print(f"score NORMAL: min={min(scores):.3f} max={max(scores):.3f} (dùng cho init char mới)")

    # DRY-RUN: append f/j/w/z, lưu ra file tạm, load lại, encode thử
    print("\n=== DRY-RUN append f/j/w/z rồi encode ===")
    mp2 = smp.ModelProto()
    mp2.ParseFromString(mp.SerializeToString())
    min_score = min(scores) if scores else -20.0
    for ch in ["f", "j", "w", "z"]:
        p = mp2.pieces.add()
        p.piece = ch
        p.score = min_score - 1.0     # điểm thấp: chỉ dùng khi buộc phải (không phá segmentation cũ)
        p.type = 1                    # NORMAL
    out = os.path.join(tmp, "expanded.model")
    with open(out, "wb") as f:
        f.write(mp2.SerializeToString())
    sp2 = spm.SentencePieceProcessor()
    sp2.Load(out)
    print(f"vocab sau expand: {sp2.GetPieceSize()} (từ {sp.GetPieceSize()})")
    for s in ["facebook", "wifi", "zalo", "fpt", "whisky", "west", "xin chào việt nam"]:
        ids_old = sp.EncodeAsIds(s)
        ids_new = sp2.EncodeAsIds(s)
        print(f"  {s:20} OLD ids={ids_old} '{sp.DecodeIds(ids_old)}'")
        print(f"  {'':20} NEW ids={ids_new} '{sp2.DecodeIds(ids_new)}'")
    print(f"\ntmp dir giữ lại để dùng tiếp: {tmp}")


if __name__ == "__main__":
    main()
