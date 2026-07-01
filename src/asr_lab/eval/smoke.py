"""Smoke test: phiên âm một file wav bằng model NeMo ASR trên CPU.

Mục đích: xác nhận luồng tải model + tiền xử lý + suy luận chạy thông trên máy,
không nhằm đo tốc độ hay chất lượng.

Chạy:  uv run python src/smoke_transcribe.py <duong_dan.wav> [--model <ten_model>]
Yêu cầu file wav: mono, 16kHz (resample trước nếu khác, vì đặc trưng phụ thuộc sample_rate).
"""

import argparse

import nemo.collections.asr as nemo_asr


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test phiên âm wav bằng NeMo ASR (CPU)")
    parser.add_argument("audio", help="Đường dẫn file wav (mono 16kHz)")
    parser.add_argument("--model", default="nvidia/stt_en_conformer_ctc_small")
    args = parser.parse_args()

    print(f"Tải model: {args.model} (CPU)...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.eval()

    print(f"Phiên âm: {args.audio}")
    result = model.transcribe([args.audio])
    print("\n==================== KẾT QUẢ ====================")
    print(result)


if __name__ == "__main__":
    main()
