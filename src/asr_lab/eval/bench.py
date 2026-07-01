"""Đo nhanh tốc độ suy luận (RTF) + WER của nhiều model NeMo ASR trên CPU.

Mục đích: so sánh BƯỚC ĐẦU để thông luồng, không phải benchmark chuẩn.
Nguyên tắc an toàn (đã từng treo máy vì CPU quá tải):
- Cap số thread (đặt OMP_NUM_THREADS=4 trước khi chạy + torch.set_num_threads).
- Nạp TỪNG model một, chạy xong giải phóng (del + gc) rồi mới sang model sau.

RTF = thời gian xử lý / thời lượng audio. RTF < 1 nghĩa là nhanh hơn thời gian thực.
WER = tỉ lệ lỗi từ sau khi chuẩn hoá (hạ thường + bỏ dấu câu).

Chạy: uv run python src/bench_asr.py <manifest.jsonl> [--limit 30] [--batch 8]
"""

import argparse
import gc
import json
import time

import torch

torch.set_num_threads(4)

import nemo.collections.asr as nemo_asr  # noqa: E402

from asr_lab.common.metrics import normalize_en as normalize, wer, extract_text  # noqa: E402
from asr_lab.common.models import MODELS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")][: args.limit]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]
    audio_sec = sum(r["duration"] for r in rows)
    print(f"Tập test: {len(rows)} utt, tổng {audio_sec/60:.1f} phút audio, batch={args.batch}\n")

    results = []
    for name in MODELS:
        print(f"--- Nạp {name} ...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=name, map_location="cpu")
        model.eval()
        t0 = time.perf_counter()
        out = model.transcribe(paths, batch_size=args.batch)
        infer_sec = time.perf_counter() - t0
        hyps = [normalize(extract_text(x)) for x in out]
        score = wer(refs, hyps)
        rtf = infer_sec / audio_sec
        results.append((name, infer_sec, rtf, score))
        print(f"    infer {infer_sec:.1f}s | RTF {rtf:.3f} | WER {score*100:.2f}%\n")
        del model
        gc.collect()

    print("\n================ BẢNG SO SÁNH (CPU, cap 4 thread) ================")
    print(f"{'Model':<46} {'infer(s)':>9} {'RTF':>7} {'WER%':>7}")
    for name, infer_sec, rtf, score in results:
        print(f"{name:<46} {infer_sec:>9.1f} {rtf:>7.3f} {score*100:>6.2f}")


if __name__ == "__main__":
    main()
