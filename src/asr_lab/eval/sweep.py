"""Quét nhiều model x nhiều bộ test -> ma trận WER + RTF (CPU).

Nạp MỖI model một lần, chạy hết các bộ test rồi mới giải phóng (tiết kiệm load + RAM).
Cap thread chống treo. Dùng lại normalize/wer/extract_text từ asr_lab.common.

Chạy: uv run python -m asr_lab.eval.sweep
"""

import gc
import json
import time

import torch

torch.set_num_threads(4)

import nemo.collections.asr as nemo_asr  # noqa: E402

from asr_lab.common.metrics import normalize_en as normalize, wer, extract_text  # noqa: E402
from asr_lab.common.models import MODELS  # noqa: E402

# (nhãn ngắn, đường dẫn manifest) — xếp từ dễ -> khó
DATASETS = [
    ("clean", "data/manifests/hf_clean12.jsonl"),       # LibriSpeech dev-clean (dễ)
    ("voxpop", "data/manifests/hf_voxpopuli.jsonl"),    # nghị viện EU, giọng đa dạng
    ("earn22", "data/manifests/hf_earnings22.jsonl"),   # họp tài chính, giọng đa quốc
    ("ami", "data/manifests/hf_ami.jsonl"),             # họp nhiều người + nhiễu (khó nhất)
]


def load_manifest(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]
    audio_sec = sum(r["duration"] for r in rows)
    return paths, refs, audio_sec


def main() -> None:
    data = {label: load_manifest(p) for label, p in DATASETS}
    wer_mat, rtf_list = {}, {}

    for name in MODELS:
        short = name.split("/")[-1]
        print(f"\n=== Nạp {short} ===", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=name, map_location="cpu")
        model.eval()
        wer_mat[short], tot_infer, tot_audio = {}, 0.0, 0.0
        for label, _ in DATASETS:
            paths, refs, audio_sec = data[label]
            t0 = time.perf_counter()
            out = model.transcribe(paths, batch_size=8)
            dt = time.perf_counter() - t0
            score = wer([r for r in refs], [normalize(extract_text(x)) for x in out])
            wer_mat[short][label] = score
            tot_infer += dt
            tot_audio += audio_sec
            print(f"    {label:8s} WER {score*100:5.2f}%  ({dt:.0f}s)", flush=True)
        rtf_list[short] = tot_infer / tot_audio
        del model
        gc.collect()

    # In ma trận tổng
    labels = [l for l, _ in DATASETS]
    print("\n\n================ MA TRẬN WER% (CPU, 12 utt/bộ) ================")
    header = f"{'Model':<40}" + "".join(f"{l:>9}" for l in labels) + f"{'RTF':>8}"
    print(header)
    for name in MODELS:
        short = name.split("/")[-1]
        row = f"{short:<40}" + "".join(f"{wer_mat[short][l]*100:>8.2f} " for l in labels)
        row += f"{rtf_list[short]:>7.3f}"
        print(row)


if __name__ == "__main__":
    main()
