---
name: BEV Injection 관련 핵심 파일
description: BEV 주입 구현 시 참조해야 할 파일 경로 및 핵심 함수
type: reference
---

# BEV Injection 구현 관련 핵심 파일

## 기존 코드 (변경 금지)

| 파일 | 핵심 함수/클래스 |
|---|---|
| `internnav/agent/internvla_n1_agent.py` | `InternVLAN1Agent`, `step()` (L247), S1 입력 구성 (L319-337) |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` | `InternVLAN1Net`, `s2_step()` (L110), `s1_step_latent()` (L200) |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py` | `generate_traj()` (L349) — navdp_async vs nextdit_async 분기 |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | `HabitatVLNEvaluator`, `_run_eval_dual_system()` (L468), LOOKDOWN 분기 (L550-585) |
| `internnav/model/utils/depth_rgb_to_bev_torch.py` | `depth_rgb_to_bev(depth[B,H,W], rgb[B,H,W,3], ...) → [B,3,224,224]` |
| `internnav/model/utils/vln_utils.py` | `S2Input`, `S2Output`, `S1Input`, `S1Output` (L140) |

## 참고 문서

| 파일 | 내용 |
|---|---|
| `/ws/src/InternNav/.claude/understanding/dual_system_flow.md` | H1 기준 S1/S2 전체 흐름 (한국어) |
| `/ws/src/InternNav/.claude/understanding/habitat_dual_system_flow.md` | Habitat 기준 S1/S2 전체 흐름 |

## Habitat LOOKDOWN 흐름 핵심

- `action==LOOKDOWN` 분기 (L550): 카메라가 이미 60° 아래 → 추가 LOOKDOWN 없이 바로 LLM 추론
- `action!=LOOKDOWN` 분기 (L566-567): `env.step(LOOKDOWN)×2`로 60° 내린 후 LLM 추론
- L663-664: pixel_goal 예측 후 `env.step(LOOKUP)×2` — `action==LOOKDOWN` 케이스에서만 유효
- S1 입력: `pix_goal_image`(rgb_memory=look_down 이미지) + 현재 look_down_image (L674-682)

## 체크포인트 설정

| 체크포인트 | `system1` | depth 사용 여부 |
|---|---|---|
| `InternVLA-N1-w-NavDP` | `navdp_async` | **사용** (DAT_RGBD_Patch_Backbone) |
| `InternVLA-N1-DualVLN` | `nextdit_async` | **미사용** (DepthAnythingV2 내부 사용) |
