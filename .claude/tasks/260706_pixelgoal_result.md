# pixel_goal: MLP 없이 diffusion에 goal 직접 전달 — 결과 보고

## 1. 이전과 비교해서 달라진 점

**이전 (기존 MLP 방식)**
- S2가 예측한 pixel 좌표를 `[0,1]` 정규화값 그대로 `pixel_cond_projector`(Linear(2→768)+GELU+Linear) MLP에 태워 `z_latents`(cross-attention 조건)에 conditioning token으로 추가.
- pixel 좌표와 diffusion이 예측하는 (dx,dy,dyaw) 사이에 좌표계/단위 관계가 없었음.
- `_extract_pixel_coords`가 `resize_w/resize_h`(384, VLM 채팅 이미지 리사이즈 크기)로 정규화 — **버그**: 실측해보니 라벨은 native 카메라 해상도 640×480 기준이었음.

**이번 변경**
- pixel → **metric 절대 위치 (x, y)** 로 변환(`_pixel_norm_to_metric`): depth(`traj_depths[:,0]`) + 카메라 intrinsics(`_BASE_FX/FY/CX/CY`, bev_provider와 동일) + pitch(`traj_cam_pitch_2`)로 실제 미터 단위 좌표 계산. `depth_rgb_to_bev_torch.py`의 world-frame 공식과 동일.
- `_extract_pixel_coords` 정규화를 640×480(native)로 수정 (`internvla_n1_trainer.py`).
- **`pixel_goal_mode` config로 두 가지 방식 모두 지원** (기본값 `"prepend"`):
  - `prepend` (MLP 없음): goal의 metric (x,y,yaw=0)를 diffusion이 denoise하는 궤적 시퀀스(`x`)의 0번째 스텝으로 직접 prepend. 노이즈를 씌우지 않고(clean) loss에서도 제외. 기존 `action_encoder`(Linear(3→384))를 그대로 재사용 — 새 파라미터 없음.
  - `mlp_cond`: 같은 metric (x,y)를 `pixel_cond_projector`로 z_latents 토큰으로 투사 (기존 방식과 동일한 자리, 입력만 metric으로 교체).
- `pixel_goal_scale`(기본 4.0, config) 로 metric goal에 곱하는 스케일을 노출 — 하드코딩 금지.
- `internvla_n1.py`(core, main에도 있는 파일) 변경은 guard clause 3곳 삽입만 (코드 이동 없음). `goal_prefix=None`이면 기존 forward()와 100% 동일 경로.
- 학습 시 goal 위치를 FPV/BEV 프레임에 시각화해서 저장하는 디버그 훅 추가 (`_debug_visualize_goal_train`, `config.debug_dir` 설정 시에만 동작).

## 2. 커맨드 및 argument 의미

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/s1.bev_s2.fpv_rgb_gt.pixel_goal.py --machine 5090 --no-eval --debug-dir logs/pixel_goal_bev
```
- `--config`: 실험 설정 파일. `bev_s1_mode="bev"`(S1이 BEV 이미지 사용) + `use_pixel_goal=True`.
- `--machine 5090`: RTX 5090 1GPU 인프라 오버라이드 (data_root/checkpoint 경로, batch_size=2, grad_accum=4, max_pixels 등).
- `--no-eval`: 학습만 실행(habitat eval 생략).
- `--debug-dir`: BEV/pixel_goal 디버그 이미지 저장 경로 (`config.debug_dir`로 전달됨).
- (`--debugpy` 사용 안 함 — guideline 반영)

동일 커맨드에서 `--config` 만 `base_s1.fpv_s2.fpv_rgb_gt.pixel_goal.py`(S1 FPV 유지)로 바꾸면 두 번째 조합.

**신규 config field** (config 파일에 직접 안 적으면 default 사용):
- `pixel_goal_mode`: `"prepend"`(기본, MLP 없음) | `"mlp_cond"`(MLP 사용).
- `pixel_goal_scale`: `4.0`(기본) — metric goal (x,y)에 곱하는 스케일, traj_poses 학습 스케일과 매칭.

## 3. 검증 (실행 결과)

> 이 sandbox엔 `habitat_sim`이 없어 `runner.py`가 config import 시점에 죽어서, 동일 `Params.train_argv()` 커맨드를 `torchrun`으로 직접 실행(3 step, eval/save 생략)해서 검증함. 실제 GPU 학습 서버에서는 위 커맨드 그대로 사용.

| 조합 | exit | loss (step1→3) |
|---|---|---|
| `s1.bev_s2...pixel_goal` (`bev_s1_mode=bev`, `pixel_goal_mode=prepend`) | 0 | 1.13 → 1.13 → 1.04 |
| `base_s1...pixel_goal` (`bev_s1_mode=fpv`, `pixel_goal_mode=prepend`) | 0 | 1.14 → 1.21 → 1.69 |
| `s1.bev_s2...pixel_goal` (`pixel_goal_mode=mlp_cond`) | 0 | 1.06 → 1.09 → 1.32 |

- `mlp_cond` 모드에서 `pixel_cond_projector`(Linear(2,768)+GELU+Linear) 새로 초기화 + trainable 확인됨.
- 로그 중 `rclpy` ModuleNotFoundError는 BEV 자체 디버그 시각화(`visual_input_provider.py`, ROS2 occupancy-grid)에서 나는 기존 이슈로, 이번 변경과 무관하고 학습 진행에 영향 없음.

**시각화 결과** (`config.debug_dir` 하위 `pixel_goal_train/`):
- `logs/pixel_goal_bev/pixel_goal_train/pixel_goal_bev_metric.jpg`: BEV 이미지 위에 goal 마커. goal_m=[3.03, -2.03] → BEV 공식으로 역변환한 픽셀[157,44]이 실제 표시된 픽셀과 일치 → 좌표 변환 self-consistent 확인.
- `logs/pixel_goal_fpv/pixel_goal_train/pixel_goal_fpv_metric.jpg`: FPV(침실) 이미지 위에 goal 마커 정상 표시, pixel/norm/metric 값 모두 오버레이.

## 미해결 / 후속 확인 필요
- 추론(`s1_step_pixel`/`s2_step`)이 VLM 입력 이미지를 `resize_w/resize_h`(384)로 리사이즈하는데, 학습 라벨은 native(640×480) 기준이라 학습/추론 간 스케일 불일치 가능성 있음 — 이번 범위 밖, 별도 확인 필요.
- `pixel_goal_scale=4.0`이 실제 goal 거리 분포에 적절한지는 실제 학습(step 수 늘려서) 후 loss로 검증 필요.
