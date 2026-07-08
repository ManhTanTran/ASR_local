# 07.07 — Quản lý vòng đời training: save / load / resume

Chốt cách lưu–nạp–tiếp tục cho training DÀI trên máy chung (khác hẳn Kaggle: 1 phiên ngắn, lưu 1 lần cuối).
Đây là gap lớn nhất của code hiện tại và điều kiện để job **preemptible** (nhường máy không mất tiến độ).

---

## Glossary
- **`.ckpt` (Lightning):** ảnh chụp giữa chừng (trọng số + optimizer + scheduler + global_step) → **resume đúng chỗ**.
- **`.nemo`:** gói model hoàn chỉnh (weights + cfg + tokenizer) để **nạp lại/inference/bàn giao**. Không có optimizer state.
- **top-k / save_last:** giữ k checkpoint tốt nhất theo val_wer + luôn giữ cái mới nhất.
- **EMA:** trung bình trượt trọng số → thường giảm WER nhẹ, rẻ.

---

## Hiện trạng (đọc từ code)

`finetune_vivos` + `continue_vi` đều: `enable_checkpointing=False`, chỉ `model.save_to(.nemo)` **1 lần cuối**.
- Hợp Kaggle (1 phiên ≤12h). **Không hợp DGX**: mất điện/bị kill giữa chừng = mất TOÀN BỘ tiến độ.
- Không resume được giữa run; không giữ best-by-val (chỉ có epoch cuối).

---

## Quyết định

### QĐ-1 — Bật `ModelCheckpoint` (top-k theo val_wer + save_last)
```python
from lightning.pytorch.callbacks import ModelCheckpoint
ckpt_cb = ModelCheckpoint(
    dirpath=run_dir/"checkpoints", monitor="val_wer", mode="min",
    save_top_k=3, save_last=True,           # 3 tốt nhất + cái mới nhất (để resume)
    filename="{epoch}-{step}-{val_wer:.4f}",
    every_n_train_steps=2000,                # ~mỗi 30–60 phút -> preemptible
)
trainer = pl.Trainer(..., enable_checkpointing=True, callbacks=[ckpt_cb], ...)
```
- `val_check_interval` đủ dày để `val_wer` cập nhật kịp cho monitor (hiện `1.0`/epoch — nấc lớn epoch dài, nên hạ xuống theo step, vd `val_check_interval=2000`).

### QĐ-2 — Hai kiểu "tiếp tục", dùng đúng chỗ
- **Resume GIỮA một run** (bị kill/hết max_time): `pl.Trainer(..., ).fit(model, ckpt_path=run_dir/"checkpoints/last.ckpt")` → khôi phục **cả optimizer + scheduler + step**. Dùng khi tiếp tục CÙNG cấu hình.
- **Sang NẤC MỚI** (S1→S2, đổi data/LR): `restore_from(.nemo nấc trước)` (như `continue_vi`) → **lịch LR mới**, KHÔNG mang optimizer cũ. Đây là ranh giới experiment mới.

### QĐ-3 — Export `.nemo` ở cuối mỗi nấc + backup team-share
- Cuối nấc: `model.save_to(run_dir/"<stage>.nemo")` (đã có) → **copy sang `/srv/team-share/models/asr_vi/`** làm tài sản chung + điểm khởi cho nấc sau.
- Đặt tên rõ: `asr_vi_fc115m_s1_YYYYMMDD.nemo` (model_stage_ngày).

### QĐ-4 — EMA (tùy chọn, rẻ)
Thêm `nemo` EMA callback (decay ~0.999) → nhặt ~0.5–1% WER. Bật ở nấc ổn định, tắt nếu smoke thấy lệch.

### QĐ-5 — Registry + provenance (dùng lại đồ có sẵn)
- Đã có `artifacts/runs/<id>/{results.json,status.json}` + `asr_lab.registry.build_scoreboard` + convention `experiments/<NN>/{spec.md,RESULT.md}`. **Giữ nguyên**, mỗi nấc = 1 experiment.
- Thêm vào `results.json`: `stage`, `resume_from`, `data_hours`, `eff_batch`, `ckpt_best_val_wer`, `gpu="GB10"`, `precision`. Để scoreboard so được across nấc/model.

---

## Vòng đời 1 nấc (chuẩn)

```
1. Chọn ckpt nền (.nemo nấc trước hoặc base) -> restore_from
2. change_vocabulary nếu đổi tokenizer (chỉ nấc đầu) [03]
3. build manifest stage + tarred [04]
4. cấu hình optim/sched/spec_aug/bucketing [05]  + ModelCheckpoint [QĐ-1]
5. trainer.fit(...)  (max_time nhường máy [06]; nếu bị kill -> fit(ckpt_path=last.ckpt) [QĐ-2])
6. eval trên eval-set cố định (FLEURS-vi + CV test) [04]
7. save_to .nemo + copy team-share [QĐ-3]; ghi results.json + RESULT.md; build_scoreboard
```

---

## Việc code
1. Thêm tham số `--resume-ckpt` (đường dẫn `last.ckpt`) vào script train → gọi `fit(ckpt_path=...)`.
2. Bật `ModelCheckpoint` (QĐ-1) + hạ `val_check_interval` theo step ở nấc lớn.
3. Hàm `export_and_backup(.nemo)` copy sang `/srv/team-share/models/asr_vi/` + đặt tên chuẩn.
4. Mở rộng `results.json` (QĐ-5) + cập nhật `build_scoreboard` nếu cần cột mới.
5. (Tùy) EMA callback.

## ✅ Tự kiểm nhanh
1. Khác nhau `.ckpt` vs `.nemo`? 2. Khi bị kill giữa run thì resume kiểu gì? 3. Sang nấc mới thì nạp kiểu gì, vì sao khác?

<details><summary>Đáp án</summary>
1. `.ckpt` = ảnh chụp giữa chừng có optimizer/step (resume đúng chỗ); `.nemo` = model hoàn chỉnh để nạp/inference/bàn giao, không có optimizer. 2. `fit(ckpt_path=last.ckpt)` — khôi phục cả optimizer+scheduler+step. 3. `restore_from(.nemo)` — lịch LR mới, bỏ optimizer cũ, vì đổi data/LR là ranh giới experiment mới.
</details>
