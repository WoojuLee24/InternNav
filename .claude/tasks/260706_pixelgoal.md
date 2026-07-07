# 작업 메모리 — 2026-07-06

## 현재 완료된 작업

### pixel_goal 서브클래스 리팩토링 (branch: feature/bev_v0.1)

기존 core 파일에 pixel_goal 코드가 직접 삽입되어 있던 것을 subclass 구조로 분리.

**Revert된 파일 (원복):**
- `internnav/agent/internvla_n1_agent.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`

**최소 변경된 기존 파일:**
- `internnav/model/basemodel/internvla_n1/internvla_n1.py`: `_build_cond_token()` hook 3줄 추가 + forward() 2줄 교체
- `internnav/trainer/internvla_n1_argument.py`: `use_pixel_goal` 1줄
- `internnav/trainer/internvla_n1_trainer.py`: `pixel_cond_projector` trainable 등록 + `setup_pixel_goal_decoder` 호출
- `internnav/model/__init__.py`: `InternVLAN1PixelGoal_Policy` 등록
- `scripts/train_eval/qwenvl_train/default_config.py`: `trainer` property에 `use_pixel_goal` 분기

**새로 생성된 파일:**
- `internnav/model/basemodel/internvla_n1/internvla_n1_pixel_goal.py`
  - `InternVLAN1BEVProviderPixelGoalForCausalLM(InternVLAN1BEVProviderForCausalLM)`
    - `pixel_cond_projector`: `Linear(2→768) + GELU + Linear(768→768)`
    - `_build_cond_token()` override: pixel coord → conditioning token
    - `generate_traj_pixel()`: inference용 diffusion 경로
  - `InternVLAN1PixelGoalNet(InternVLAN1Net)`
    - `s2_step()` override: `generate_latents` 호출 스킵, `output_pixel`만 설정
    - `s1_step_pixel()`: pixel coord로 S1 추론
- `internnav/trainer/internvla_n1_pixel_goal_trainer.py`
  - `parse_bev_cli_args` 재사용, `InternVLAN1BEVProviderPixelGoalForCausalLM`으로 monkey-patch
- `internnav/agent/internvla_n1_pixel_goal_agent.py`
  - `InternVLAN1PixelGoalAgent(InternVLAN1Agent)`: pixel S1 dispatch path

**검증된 config 라우팅:**
- `s1.bev_s2.fpv_rgb_gt.pixel_goal.py`: `bev=True, bev_s1_mode=bev, use_pixel_goal=True` → `pixel_goal_trainer` ✓
- `base_s1.fpv_s2.fpv_rgb_gt.pixel_goal.py`: `bev=True, bev_s1_mode=fpv, use_pixel_goal=True` → `pixel_goal_trainer` ✓

**현재 확인된 동작:**
`python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/s1.bev_s2.fpv_rgb_gt.pixel_goal.py --machine 5090 --debugpy trainer --no-eval`
→ pixel_coordinate를 `pixel_cond_projector` (MLP)로 conditioning한 결과만 확인됨.

---

## 앞으로 할 작업

### 목표: pixel_coordinate를 MLP 없이 diffusion 입력에 직접 넣기
### 조건: Diffusion의 pixel_coordinate입력과 출력 waypoint가 같은 좌표계, 단위를 가져야 됨.
### 참고: depth 사용 가능
### 확인: 명령어 실행 및 정상 작동 확인

**Diffusion 출력 형태:**
```
generate_traj() 출력: (N_trajs=32, T_steps=32, 3)
                                               └─ (dx, dy, d_yaw)
```
- 좌표계: 로봇 local frame, 현재 위치 기준 relative delta
- 단위: 미터/라디안, 학습 시 ×4 스케일 → inference에서 `/= 4.0` (vln_utils.py line 129)
- `dx`: 전진, `dy`: 측방, `d_yaw`: heading 변화

**pixel_coord ↔ diffusion 좌표계 관계:**

| Config | S2 출력 | S1 입력 | 좌표계 호환 |
|--------|---------|---------|------------|
| `s1.bev_s2.fpv_rgb_gt.pixel_goal` | FPV pixel | BEV image | **호환 가능** |
| `base_s1.fpv_s2.fpv_rgb_gt.pixel_goal` | FPV pixel | FPV image | 불일치 (depth 필요) |

**BEV pixel → robot metric 변환 (s1.bev config):**
```
bev_range=5m, bev_size=224px → scale = 224/(2×5) = 22.4 px/m
X_forward = (bev_size/2 - row_px) / scale   [m]
Y_lateral  = (bev_size/2 - col_px) / scale  [m]
```
이 `(X_forward, Y_lateral)`은 diffusion의 `(dx, dy)`와 동일한 좌표계.

**검토할 구현 방향:**
현재: `(2,) pixel_norm` → MLP → `(1, 768)` conditioning token → z_latents에 append

추가 방법: pixel → metric 변환 후 goal waypoint를 diffusion sequence에 직접 prepend
```python
# scale ×4 적용 (학습 시 스케일 맞춤)
goal_step = [X_forward * 4, Y_lateral * 4, 0.0]  # (dx, dy, dyaw=0)
# diffusion latent sequence (T, 3) 앞에 goal step prepend
```
→ MLP projector 없이 goal 정보를 diffusion에 직접 전달

**디버깅: 아래 두 코드 정상 실행 확인 및 pixel goal을 fpv와 bev에 시각화해서 디버깅**
`/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/s1.bev_s2.fpv_rgb_gt.pixel_goal.py --machine 5090 --no-eval --debug-dir logs/pixel_goal_bev`
`/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/base_s1.fpv_s2.fpv_rgb_gt.pixel_goal.py --machine 5090 --no-eval --debug-dir logs/pixel_goal_fpv`
(`--debugpy` 사용 금지 — guideline.md 업데이트에 따름)

---

## 2026-07-07 업데이트 — 구현 완료 + 설계 수정

**발견한 버그:** `_extract_pixel_coords`가 `data_args.resize_w/resize_h`(384, VLM chat 이미지 리사이즈 크기)로 정규화하고 있었는데, parquet `goal.{setting}` 라벨을 실측(최대 col=546, row=452, 640×480 이미지)해보니 실제로는 **native 카메라 해상도 640×480**(= `internvla_n1_bev_provider.py`의 `_BASE_W/_BASE_H`) 기준이었음. `internvla_n1_trainer.py`의 `setup_pixel_goal_decoder` 호출부를 `_BASE_W, _BASE_H`로 수정.

**최종 설계 (`internvla_n1_pixel_goal.py`):**
- `_pixel_norm_to_metric()`: pixel_norm + depth(traj_depths[:,0]) + intrinsics(_BASE_FX/FY/CX/CY, `_BASE_W/H`로 스케일) + cam_pitch(traj_cam_pitch_2, bev_provider.py에서 unconditional stash) → **(X_forward, Y_lateral)**, 현재 프레임 기준 goal의 **절대 위치**(미터). `depth_rgb_to_bev_torch.py`의 world-frame 공식과 동일.
- **네이밍 정정**: 이 값은 궤적의 각 스텝(`dx,dy,dyaw` — 이전 스텝 대비 증분)과 다르게, **절대 (x, y) 위치**(yaw는 미정=0)임. 처음에 "goal_step = (dx,dy,dyaw)"로 잘못 명명했던 것을 `goal_pose`/"ABSOLUTE position" 용어로 정정.
- **스케일**: `× 4.0` (traj_poses 학습 스케일과 맞춤)을 하드코딩하지 않고 `pixel_goal_scale`(config, 기본 4.0)로 노출 — goal까지의 거리가 한 스텝 델타보다 훨씬 클 수 있어 조정 가능해야 함.
- **`pixel_goal_mode` config로 두 가지 방식 모두 지원**:
  - `"prepend"` (기본, MLP 없음): `_build_goal_prefix()`가 goal_pose를 diffusion trajectory 시퀀스 맨 앞(0번째 스텝)에 prepend. 노이즈 안 씌우고(clean), loss도 제외. `action_encoder`(기존 Linear(3,384))를 그대로 재사용.
  - `"mlp_cond"`: `_build_cond_token()`이 동일한 goal_xy를 `pixel_cond_projector`(Linear(2,768)+GELU+Linear, `__init__`에서 항상 생성)로 z_latents 컨디셔닝 토큰으로 투사. 기존 VLM-latent conditioning과 같은 자리.
  - 둘 다 `_decode_goal_xy()` 공유 헬퍼로 동일한 goal 값을 사용 → 모드만 바꿔서 비교 가능.
- `internvla_n1.py` core 변경은 **guard clause만 추가**(코드 이동/추출 없음) — `_build_goal_prefix` 훅 + `relative_poses`/`noisy_trajectory`/loss 부분에 `if goal_prefix is not None:` 3곳 삽입, `goal_prefix=None`이면 기존 코드와 100% 동일 경로.
- config 필드 추가 위치: `internvla_n1_argument.py`(`pixel_goal_mode`, `pixel_goal_scale`) → `internvla_n1_trainer.py`(model.config로 전달) → `default_config.py` Params + `train_argv()`(커맨드라인으로 노출).

**미해결/후속 과제:**
- 추론측(`s1_step_pixel`/`s2_step`)의 `resize_w/resize_h`(384) 정규화는 이번 수정 범위 밖 — 학습 라벨(640×480 native)과 실제 일치하는지 별도 확인 필요.
- `pixel_goal_scale`(기본 4.0)이 실제로 적절한지는 학습 후 loss/시각화로 검증 필요.
