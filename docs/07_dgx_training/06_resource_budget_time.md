# 07.06 — Ước lượng thời gian train + ngân sách resource + chia máy

- DGX là **1 GPU dùng chung ~10 người**.
- Phần này chốt:
  - ước lượng giờ train,
  - dung lượng đĩa,
  - và **giao ước dùng máy** để không giành tài nguyên của người khác/tác vụ khác.

---

## Glossary

- **GPU-hour:** 1 giờ chiếm trọn GPU. Trên máy 1-GPU: 1 job train = chiếm 100% GPU (không co-train được).
- **max_time:** trần thời gian Lightning tự dừng (code đã có `--max-minutes`) → luôn nhường máy đúng hẹn.
- **Preemptible:** job có thể bị dừng giữa chừng mà không mất tiến độ (nhờ checkpoint đều).

---

## Ước lượng thời gian (PHẢI đo lại bằng smoke)

Chưa có benchmark NeMo-ASR trên GB10 → ước lượng bậc độ lớn, dựa số giây/step đo từ smoke.

Công thức: `giờ ≈ total_steps × (giây/step) / 3600`.
Giả định thận trọng **giây/step ∈ [0.4, 1.0]** cho 115M @ batch 32 bf16 (chốt sau smoke):

| Nấc          | total_steps | @0.4s/step | @1.0s/step |
| ------------ | ----------- | ---------- | ---------- |
| S1 (~65k)    | 65.000      | ~7h        | ~18h       |
| S2 (~180k)   | 180.000     | ~20h       | ~50h       |
| S3 (~240k)   | 240.000     | ~27h       | ~67h       |
| **Cả 3 nấc** | ~485k       | **~54h**   | **~135h**  |

→ Bậc độ lớn: **vài chục đến ~trăm giờ GPU** cho pipeline 3 nấc model 115M.

Không phải việc 1 đêm — chia nhiều phiên, checkpoint đều, chạy chủ yếu **giờ thấp điểm**.

**Rút ngắn khả thi:** bucketing + tarred (throughput ↑), giảm epoch nấc lớn, freeze encoder ở nấc đầu.

---

## Ngân sách đĩa (`/srv` còn 1.6TB)

| Hạng mục                      | Dung lượng ~ | Ghi chú                           |
| ----------------------------- | ------------ | --------------------------------- |
| Raw đã kéo (ungated)          | ~182 GB      | `asr_vi/<name>/`                  |
| WAV 16k giải nén (nếu để lẻ)  | ~180 GB      | → chuyển tarred để giảm file lẻ   |
| Tarred                        | ~120–150 GB  | thay wav lẻ cho bộ lớn            |
| Checkpoint (top-k + last)/run | ~2–5 GB      | .ckpt ~ vài trăm MB; .nemo ~0.5GB |
| Gated (nếu kéo sau)           | +~210 GB     | viVoice/pho/vietmed               |

→ Đủ cho ungated + tarred + nhiều run. Nếu kéo cả gated: cân nhắc **dọn 1.1TB rác** (htt210/hieutb) trước.

---

## 🤝 Giao ước dùng máy (bắt buộc — máy chung)

1. **Kiểm tra trước khi phóng:** `ssh dgx 'nvidia-smi'` hoặc `dgx-gpu`. GPU đang bận (người khác train) → **KHÔNG chen**, xếp lịch sau. 1-GPU không co-train.
2. **Chạy trong tmux** (không dùng `setsid` rời như job kéo data): `tmux new -s asr_train` → job sống, người khác/bản thân theo dõi được. Đặt tên session rõ.
3. **Luôn đặt `--max-minutes`** (trần thời gian) → tự nhường máy đúng hẹn, không chiếm vô thời hạn.
4. **Checkpoint đều** (mỗi ~30–60 phút, xem [07](07_training_lifecycle.md)) → job **preemptible**: ai cần gấp thì dừng được, resume không mất tiến độ.
5. **Ưu tiên giờ thấp điểm** cho run dài (đêm/cuối tuần). Run dài ban ngày phải báo team trước.
6. **Báo team** (kênh chung) khi bắt đầu run nhiều giờ: bộ nào, dự kiến bao lâu, tmux nào.
7. **Không nhồi batch cực lớn** chiếm sạch memory — chừa cho người khác chạy inference/VLM song song (inference nhẹ có thể chen cùng lúc nếu training không nuốt hết memory; canh `nvidia-smi`).
8. Dọn checkpoint/run cũ sau khi export `.nemo` → tránh phình `/srv` chung.

> Có sẵn hạ tầng: `dgx-gpu`, `dgx-scan` (cron 07:00 → `_health/latest.txt`). Nên thêm dòng "ai đang chiếm GPU cho việc gì" vào bảng health để cả team thấy — đề xuất ở [00_README](00_README.md) backlog.

---

## Việc / đo

1. Smoke 50-step → giây/step thật → thay vào bảng trên, chốt lịch chạy từng nấc.
2. Quyết định lịch: nấc S1 chạy 1 phiên đêm; S2/S3 chia nhiều phiên preemptible.
3. (Đề xuất) thêm cột "GPU owner hiện tại" vào `_health` để điều phối.

## ✅ Tự kiểm nhanh

1. Vì sao không chen khi GPU đang bận? 2. Ba cơ chế nhường máy? 3. Vì sao checkpoint đều = nhường máy dễ?

<details><summary>Đáp án</summary>
1. 1-GPU không co-train, chen làm cả hai chậm/OOM. 2. `--max-minutes` (tự dừng), tmux (theo dõi/kill được), ưu tiên giờ thấp điểm + báo team. 3. Preemptible — dừng giữa chừng vẫn resume từ checkpoint gần nhất, không mất tiến độ.
</details>
