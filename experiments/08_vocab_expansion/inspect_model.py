"""Mổ cấu trúc RNNT .nemo + tokenizer để phẫu thuật vocab expansion đúng (read-only).

In ra: kiểu tokenizer, vocab_size, blank index, shape 2 tensor phụ thuộc V
(decoder.prediction.embed, joint.joint_net cuối) + bias, và thử encode f/j/w/z.

Chạy trên DGX:
  cd /srv/team-share/projects/nvidia_asr_nemo
  .venv/bin/python experiments/08_vocab_expansion/inspect_model.py \
      --nemo /srv/team-share/models/asr_vi/s3-fc115m-full.nemo
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.utils import logging as _nlog
    _nlog.set_verbosity(_nlog.ERROR)

    m = nemo_asr.models.ASRModel.restore_from(args.nemo, map_location="cpu")
    print(f"=== MODEL: {type(m).__name__} ===")
    tok = m.tokenizer
    V = tok.vocab_size
    print(f"tokenizer type : {type(tok).__name__}")
    print(f"vocab_size (V) : {V}")

    # blank index trong RNNT = V (num_classes); decoder/joint output = V+1
    print("\n=== các tensor phụ thuộc V ===")
    sd = m.state_dict()
    depends = []
    for name, t in sd.items():
        if any(d == V + 1 for d in t.shape):
            depends.append((name, tuple(t.shape)))
    for name, shape in depends:
        print(f"  {name:60} {shape}")

    print("\n=== module trọng yếu ===")
    dec = m.decoder
    joint = m.joint
    print(f"decoder class : {type(dec).__name__}")
    print(f"joint class   : {type(joint).__name__}")
    # embed của prediction network
    try:
        emb = dec.prediction.embed
        print(f"decoder.prediction.embed : {type(emb).__name__} weight={tuple(emb.weight.shape)} "
              f"padding_idx={emb.padding_idx}")
    except AttributeError as e:
        print(f"  (không tìm thấy dec.prediction.embed: {e})")
    # joint output layer cuối
    try:
        jnet = joint.joint_net
        last = jnet[-1]
        print(f"joint.joint_net[-1] : {type(last).__name__} weight={tuple(last.weight.shape)} "
              f"bias={tuple(last.bias.shape) if last.bias is not None else None}")
        print(f"joint.joint_net len={len(jnet)} : {[type(x).__name__ for x in jnet]}")
    except (AttributeError, TypeError) as e:
        print(f"  (không đọc được joint.joint_net: {e})")

    print(f"\njoint num_classes_with_blank = {getattr(joint, 'num_classes_with_blank', 'n/a')}")
    print(f"decoder blank_idx = {getattr(dec, 'blank_idx', 'n/a')}")

    print("\n=== encode thử f/j/w/z + loanword ===")
    for s in ["facebook", "wifi", "zalo", "fpt", "west", "whisky", "xin chào việt nam"]:
        try:
            ids = tok.text_to_ids(s)
            back = tok.ids_to_text(ids)
            print(f"  {s:20} -> ids(len {len(ids)}) -> '{back}'")
        except Exception as e:  # noqa: BLE001
            print(f"  {s:20} -> LỖI {e}")

    # kiểm tra có char f/j/w/z trong vocab không
    print("\n=== charset check ===")
    vocab = [tok.ids_to_tokens([i])[0] if hasattr(tok, "ids_to_tokens") else "" for i in range(V)]
    joined = "".join(vocab).lower()
    for ch in "fjwz":
        print(f"  '{ch}' xuất hiện trong pieces: {ch in joined}")

    # tokenizer model file path bên trong
    print("\n=== tokenizer internals ===")
    for attr in ["model_path", "vocab_path", "tokenizer"]:
        print(f"  tok.{attr} = {getattr(tok, attr, 'n/a')}")
    inner = getattr(tok, "tokenizer", None)
    if inner is not None:
        print(f"  inner type = {type(inner).__name__}")
        # SentencePieceProcessor?
        for meth in ["GetPieceSize", "IsByte", "id_to_piece"]:
            print(f"    has {meth}: {hasattr(inner, meth)}")


if __name__ == "__main__":
    main()
