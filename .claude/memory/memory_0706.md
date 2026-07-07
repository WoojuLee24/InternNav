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
### 조건: Diffusion의 pixel_coordinate입력과 
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

대안: pixel → metric 변환 후 goal waypoint를 diffusion sequence에 직접 prepend
```python
# scale ×4 적용 (학습 시 스케일 맞춤)
goal_step = [X_forward * 4, Y_lateral * 4, 0.0]  # (dx, dy, dyaw=0)
# diffusion latent sequence (T, 3) 앞에 goal step prepend
```
→ MLP projector 없이 goal 정보를 diffusion에 직접 전달
