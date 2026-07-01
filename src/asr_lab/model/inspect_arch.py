"""Soi kiến trúc + đếm tham số một model NeMo ASR (chạy CPU, không cần audio).

Mục đích: thông luồng tải model pretrained và lấy số liệu kiến trúc thật
(danh sách layer, số tham số từng module) — phục vụ đọc hiểu, không huấn luyện.

Chạy:  uv run python src/inspect_arch.py [--model <ten_model_NGC_hoac_HF>]
"""

import argparse

import torch
import nemo.collections.asr as nemo_asr


def count_params(module: torch.nn.Module) -> int:
    """Đếm tổng số tham số của một module (gồm cả phần không train)."""
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Soi kiến trúc model NeMo ASR trên CPU")
    # Mặc định dùng model nhỏ để tải nhanh + chạy CPU nhẹ; đổi sang parakeet-tdt-0.6b-v2
    # khi muốn so với Fast-Conformer cũ (nặng hơn, tải lâu hơn).
    parser.add_argument("--model", default="nvidia/stt_en_conformer_ctc_small")
    args = parser.parse_args()

    # map_location="cpu": ép tải trọng số về CPU, tránh đòi CUDA trên máy không GPU.
    print(f"Tải model: {args.model} (CPU)...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.eval()

    print("\n==================== CÂY LAYER ====================")
    print(model)  # in toàn bộ cây module, tương tự bản dump VPB cũ

    print("\n==================== BẢNG THAM SỐ ====================")
    model.summarize()  # bảng tham số từng module (PyTorch Lightning)

    print("\n==================== TỔNG HỢP THAM SỐ ====================")
    print(f"Tổng tham số toàn model: {count_params(model):,}")
    # Tách theo các khối chính của ASR để đối chiếu với tài liệu deep-dive.
    for name in ("preprocessor", "encoder", "decoder", "joint"):
        sub = getattr(model, name, None)
        if sub is not None:
            print(f"  {name:14s}: {count_params(sub):,}")


if __name__ == "__main__":
    main()
