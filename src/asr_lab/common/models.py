"""Danh sách model ASR dùng cho benchmark tiếng Anh (đều English-only)."""

# Bốn model so sánh ở docs/06_benchmarks (CTC nhỏ · FastConformer-RNNT · TDT 0.6B · streaming 0.6B).
MODELS = [
    "nvidia/stt_en_conformer_ctc_small",            # 13M, Conformer cũ + CTC
    "nvidia/stt_en_fastconformer_transducer_large",  # ~115M, FastConformer + RNNT (cỡ VPB)
    "nvidia/parakeet-tdt-0.6b-v2",                  # 618M, FastConformer + TDT
    "nvidia/nemotron-speech-streaming-en-0.6b",     # 618M, FastConformer + RNNT streaming
]
