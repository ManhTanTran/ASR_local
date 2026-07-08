# concepts — luận điểm nền ASR

Kiến thức nền **ổn định** đã viết kỹ ở `docs/02_asr_components/` — không lặp lại ở đây, chỉ trỏ:

| Khái niệm | Vị trí |
| --- | --- |
| Tokenizer / BPE / mel | `docs/02_asr_components/` |
| Encoder Conformer / FastConformer | `docs/02_asr_components/05_encoder_conformer.md` |
| Giải mã RNNT / CTC / cache-aware streaming | `docs/02_asr_components/07_decode_rnnt.md` |
| WER / CER / RTF | `docs/06_benchmarks/` + `experiments/_PROTOCOL.md` |
| Chuẩn hoá text (bài học `<unk>`) | `.agent/skills/04_data_infra/data_prep/normalize-text-data.md` |

> `concepts/` chỉ thêm file mới khi có luận điểm nền CHƯA có trong `docs/` (vd "vì sao data > model
> với ASR low-resource"). Mục tiêu tránh rải mảnh.
