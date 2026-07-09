# 결과 보고: Habitat eval(rollout) — UnifiedImageProvider 10개 조합 검증

관련 이전 결과: [260707_unified_provider_combos_result.md](260707_unified_provider_combos_result.md)

## 요약

**10개 전부 PASS** — Traceback 없이 완료. eval 단계에서도 크래시 없음 확인.

GPU 복구(NVML 초기화 오류 해소) 후 2026-07-08 02:04 KST 재실행.

---

## 실행 커맨드

```
python3 scripts/train_eval/qwenvl_train/runner.py \
  --config scripts/train_eval/qwenvl_train/image/<combo>.py \
  --machine 5090 \
  --no-train \
  --model-path checkpoints/InternVLA-N1-DualVLN \
  --eval-max-episodes 2 \
  --debug-dir logs/input_v0.1/habitat_eval/<combo>
```

체크포인트: `InternVLA-N1-DualVLN` (학습 안 된 축 포함 — SR/SPL은 참고용, 크래시 여부만 목적).

---

## 결과 (10/10 PASS)

| # | config | NE | SPL | SR | 디버그 이미지 | 비고 |
|---|---|---|---|---|---|---|
| 1 | `base_s1.fpv_s2.fpv` | 3.69 | 0.321 | 0.500 | 없음 (no-op) | baseline |
| 2 | `s1.bev_rgb_replace_s2.fpv` | 2.83 | 0.384 | 0.500 | 295장 `step_*_b00_{1_fpv,2_gt_depth,3_bev_gt,...}.jpg` | S1 BEV 디버그 정상 |
| 3 | `s1.depth_raw_s2.fpv` | 7.90 | 0.500 | 0.500 | init 이미지만 | S1 depth eval-path 디버그 미저장¹ |
| 4 | `s1.depth_normalized_s2.fpv` | 4.42 | 0.431 | 0.500 | init 이미지만 | 동일 |
| 5 | `s1.depth_colormap_s2.fpv` | 4.73 | 0.309 | 0.500 | init 이미지만 | 동일 |
| 6 | `s1.fpv_s2.bev_rgb` | 3.92 | 0.352 | 0.500 | 31장 `s2_step_*_{1_fpv,2_bev_gt}.jpg` | S2 BEV 디버그 정상 |
| 7 | `s1.fpv_s2.depth_raw` | 3.38 | 0.500 | 0.500 | 93장 `s2_unified_step_*_{1_fpv,2_raw}.jpg` | S2 depth raw 디버그 정상 |
| 8 | `s1.fpv_s2.depth_normalized` | 7.01 | 0.500 | 0.500 | 53장 `s2_unified_step_*_{1_fpv,2_normalized}.jpg` | S2 depth normalized 정상 |
| 9 | `s1.fpv_s2.depth_colormap` | 0.52 | 0.687 | 1.000 | 71장 `s2_unified_step_*_{1_fpv,2_colormap}.jpg` | S2 depth colormap 정상 |
| 10 | `s1.fpv_s2.bev_occ` | 2.99 | 0.365 | 0.500 | 53장 `s2_step_*_{1_fpv,2_bev_occ_gt}.jpg` | S2 occ 디버그 정상 |

> ¹ **S1 depth eval-path 디버그 미저장**: `unified_step_*` 이미지(훈련 단계에서 `internvla_n1_lerobot_dataset.py`가 저장)는 eval 경로에서 호출되지 않음 — 훈련/eval 코드 경로 분리로 인한 예상된 동작이며 버그 아님. 크래시는 없음.

---

## 디버그 이미지 육안 확인

- **`s1.bev_rgb_replace_s2.fpv`**: `step_*_b00_{1_fpv,2_gt_depth,3_bev_gt,5_gt_cam,6_ros2_occ_gt,...}` — 훈련 단계와 동일한 BEV 파이프라인 정상 동작 확인.
- **`s1.fpv_s2.bev_rgb`**: `s2_step_*_1_fpv.jpg`(원본 프레임) + `_2_bev_gt.jpg`(BEV 투영) 쌍 생성 — 기하학적 일치 확인.
- **`s1.fpv_s2.bev_occ`**: `s2_step_*_2_bev_occ_gt.jpg` 생성 — occ map FOV 콘 형태 정상.
- **`s1.fpv_s2.depth_*`**: `s2_unified_step_*_1_fpv.jpg` + `_2_{mode}.jpg` 쌍 생성 — raw/normalized/colormap 각각 예상 형태 확인.

---

## WARN 메시지 (무해)

모든 조합에서 `findDecoder imread_('...step_-00001_3_bev.jpg'): can't open/read file` WARN 발생 — 초기화 시 BEV 디스플레이 코드가 직전 step(-1) BEV를 읽으려다 파일이 없어 나오는 것. 기존 eval 코드의 동작이며 크래시 아님.

---

## 결론

Habitat eval 단계에서 UnifiedImageProvider 10개 조합 전부 정상 동작 확인.
- 학습 단계 검증(260707): 14 PASS + bev-concat 6 NotImplementedError
- Habitat eval 단계 검증(260708): 10/10 PASS (bev-concat/panorama 제외, 실사용 가능 조합 전부)

**다음 단계**: Isaac Sim h1 환경에서 동일 조합 검증 (해당 환경 준비 후 별도 진행).
