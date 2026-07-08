"""Helper soi cấu hình model NeMo ASR — dùng trong notebook explore (chạy CPU).

Mục tiêu: in cấu hình model gọn, mạch lạc (params, encoder, decoder/joint, tokenizer)
mà không phải nhớ API NeMo. Mọi hàm phòng thủ (try/except) để chạy được trên nhiều loại model
(CTC không có joint; RNNT/TDT có decoder + joint).
"""

import gc

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr


def set_cpu_threads(n: int = 4) -> None:
    """Giới hạn số nhân CPU torch dùng, tránh pin 100% mọi nhân làm treo máy.
    Gọi ngay đầu notebook. Nên đặt thêm OMP_NUM_THREADS trước khi import torch để chắc."""
    torch.set_num_threads(n)


def free() -> None:
    """Dọn bộ nhớ sau khi xong một model. Dùng kèm `del model` ở phía gọi:
    `del m; ih.free()` — tránh giữ nhiều model 0.6B cùng lúc trong RAM."""
    gc.collect()


# ---- Load ----

def load(model_name: str, device: str = "cpu"):
    """Tải model pretrained về CPU và đặt ở chế độ eval."""
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=device)
    model.eval()
    return model


# ---- Đếm tham số ----

def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def params_by_module(model):
    """Trả về list (tên_khối, số_params) cho các khối chính của ASR."""
    rows = []
    for name in ("preprocessor", "encoder", "decoder", "joint"):
        sub = getattr(model, name, None)
        if sub is not None:
            rows.append((name, count_params(sub)))
    return rows


# ---- Đọc cấu hình (an toàn) ----

def _get(cfg, path, default="—"):
    """Lấy cfg theo đường dẫn 'a.b.c', thiếu thì trả default."""
    cur = cfg
    for key in path.split("."):
        try:
            cur = cur[key]
        except Exception:
            return default
    return cur


def encoder_info(model) -> dict:
    return {
        "class": str(_get(model.cfg, "encoder._target_")).split(".")[-1],
        "d_model": _get(model.cfg, "encoder.d_model"),
        "n_layers": _get(model.cfg, "encoder.n_layers"),
        "n_heads": _get(model.cfg, "encoder.n_heads"),
        "ff_expansion_factor": _get(model.cfg, "encoder.ff_expansion_factor"),
        "subsampling": _get(model.cfg, "encoder.subsampling"),
        "subsampling_factor": _get(model.cfg, "encoder.subsampling_factor"),
        "self_attention_model": _get(model.cfg, "encoder.self_attention_model"),
        "conv_kernel_size": _get(model.cfg, "encoder.conv_kernel_size"),
    }


def vocab_size(model):
    """Số token của tokenizer (BPE)."""
    tok = getattr(model, "tokenizer", None)
    for attr in ("vocab_size",):
        v = getattr(tok, attr, None)
        if v:
            return v
    try:
        return len(tok.vocab)
    except Exception:
        return None


def _last_linear_out(module):
    """out_features của lớp Linear cuối trong một module (vd joint_net)."""
    lins = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return lins[-1].out_features if lins else None


def decode_info(model) -> dict:
    """Nhận diện kiểu giải mã (CTC / RNNT / TDT) + số chiều joint."""
    info = {
        "decoder_class": str(_get(model.cfg, "decoder._target_")).split(".")[-1],
        "joint_class": None,
        "joint_out": None,
        "vocab": vocab_size(model),
        "kind": "CTC",
        "num_durations": 0,
    }
    joint = getattr(model, "joint", None)
    if joint is not None:
        info["joint_class"] = type(joint).__name__
        jn = getattr(joint, "joint_net", None)
        if jn is not None:
            info["joint_out"] = _last_linear_out(jn)
        info["kind"] = "RNNT"
        # TDT: joint_out > vocab + 1 (blank) -> phần dư là số duration
        if info["joint_out"] and info["vocab"]:
            extra = info["joint_out"] - (info["vocab"] + 1)
            if extra > 0:
                info["kind"] = "TDT"
                info["num_durations"] = extra
    return info


# ---- In gọn ----

def print_overview(model, name: str) -> None:
    """In tổng quan một model: params, encoder, giải mã, tokenizer."""
    line = "=" * 64
    print(line)
    print(f"MODEL: {name}")
    print(f"  class: {type(model).__name__}")
    print(line)

    total = count_params(model)
    print(f"Tổng tham số: {total:,}")
    for n, p in params_by_module(model):
        print(f"    {n:13s}: {p:>13,}  ({p / total * 100:5.1f}%)")

    print("\nEncoder:")
    for k, v in encoder_info(model).items():
        print(f"    {k:22s}: {v}")

    d = decode_info(model)
    print("\nGiải mã:")
    print(f"    kiểu                  : {d['kind']}")
    print(f"    decoder class         : {d['decoder_class']}")
    print(f"    joint class           : {d['joint_class']}")
    print(f"    joint out / vocab     : {d['joint_out']} / {d['vocab']}")
    if d["kind"] == "TDT":
        print(f"    -> TDT: joint_out {d['joint_out']} = {d['vocab']} token + 1 blank + {d['num_durations']} duration")
    print(line + "\n")


def summary_row(model, name: str) -> dict:
    """Một dòng dữ liệu để gộp thành bảng so sánh (pandas)."""
    e = encoder_info(model)
    d = decode_info(model)
    return {
        "model": name,
        "params(M)": round(count_params(model) / 1e6, 1),
        "encoder": e["class"],
        "d_model": e["d_model"],
        "n_layers": e["n_layers"],
        "subsamp": e["subsampling_factor"],
        "decode": d["kind"],
        "joint_out": d["joint_out"],
        "vocab": d["vocab"],
        "durations": d["num_durations"],
    }


def config_yaml(model, section: str) -> str:
    """Trả về YAML của một phần config (vd 'encoder', 'decoder', 'joint')."""
    node = _get(model.cfg, section, None)
    if node is None:
        return f"(không có section '{section}')"
    return OmegaConf.to_yaml(node)
