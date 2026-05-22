---
name: VLN-PE GT Follower Agent Plan
description: GT episode oracle agent 구현 계획 — VLN-PE reference_path를 따라 H1 로봇을 flash 모드로 움직이며 SR/SPL/NDTW 상한 메트릭 측정
type: project
---

VLN-PE GT episode의 reference_path를 따라 로봇을 움직이는 oracle agent 구현을 계획했으나 아직 구현하지 않음.

**Why:** 사용자가 GT 경로 추적 시 얻을 수 있는 upper-bound 메트릭(SR≈1.0, SPL≈1.0, NDTW≈1.0)을 측정하고자 함.

**플랜 파일 위치:** `/root/.claude/plans/vln-pe-gt-episode-misty-moon.md`

**How to apply:** 다음 세션에서 이 작업을 재개할 때 위 플랜 파일을 읽고 구현 시작.

### 구현해야 할 내용 (미완료)

수정/생성 순서:
1. **NEW** `internnav/agent/gt_follower_agent.py` — `@Agent.register('gt_follower')` GTFollowerAgent
2. **MODIFY** `internnav/configs/evaluator/vln_default_config.py` — line 317에 `else: model_settings = {}` 추가 (NameError 방지)
3. **MODIFY** `internnav/evaluator/vln_distributed_evaluator.py` — reference_path를 agent에 주입 (+8 lines, 2곳)
4. **MODIFY** `internnav/agent/__init__.py` — GTFollowerAgent import 추가
5. **NEW** `scripts/eval/configs/h1_gt_follower_cfg.py` — 평가 설정 파일

### 핵심 기술 사항

- Flash 모드 액션: `{'action': [int], 'ideal_flag': True}`, int: 0=stop, 1=forward 0.25m, 2=turn_left 15°, 3=turn_right 15°, -1=stand_still
- `reference_path`는 `reset_info.data['reference_path']` — evaluator에는 있지만 agent에는 전달 안 됨
- agent에 reference_path 전달: evaluator에 `hasattr(self.agent, 'set_reference_paths')` 체크 후 호출하는 방식 (backward compatible)
- 주입 위치 A: `eval()` warm_up 완료 후 (초기 에피소드)
- 주입 위치 B: `terminate_ops()` `self.agent.reset(env_ids)` 직후 (후속 에피소드)
- yaw 계산: `quat_to_euler_angles(obs['globalrotation'])[-1]` from `internnav.utils.geometry_utils`
- WAYPOINT_THRESHOLD = 0.3m, TURN_THRESHOLD = radians(10)
