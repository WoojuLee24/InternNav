# s1.bev.rgb.concat_s2.fpv (Isaac/H1) concat 조사 + BEV world-align 회전 버그 수정 — 결과

## 배경

`s1.bev.rgb.concat_s2.fpv`(isaac/h1)에서 "concat이 안 되고 replace만 되는 것 같다"는 리포트에서 시작. 코드를 추적한 결과 concat/replace 라우팅 자체는 정상(`unified_image_provider.py::_get_s1_bev_concat` → `internvla_n1.py::generate_traj`의 `torch.cat`)이고, habitat도 provider 내부에서 동일하게 concat을 처리하므로 post-processing concat은 애초에 존재하지 않았음. 실제 문제는 디버그 이미지 비교 방식에 있었음.

## 1) world-align 디버그 parity 추가 (isaac에 없던 것)

Habitat evaluator(`habitat_vln_evaluator_unified.py::_save_eval_tdmap_debug`)는 world-frame 정렬된 BEV(`_w3_bev_world_*.jpg`)를 저장하지만, isaac(h1) 쪽 `internvla_n1_policy_unified.py`엔 이 단계가 없어서 `_3_bev_gt_cur.jpg`(agent-frame)와 `_5_gt_cam.jpg`(world-frame, 미회전)를 직접 비교하고 있었음 — 애초에 정렬 안 된 프레임끼리 비교.

`internnav/model/utils/isaac_topdown_debug.py`에 `save_isaac_bev_world_debug()`를 추가해서 이 gap을 메움.

## 2) 회전 버그 발견 및 수정

처음 만든 `save_isaac_bev_world_debug`는 `_3_bev_gt.jpg`를 `_5_gt_cam.jpg`와 "동일한" `yaw` 회전으로 돌렸음 (habitat의 `_save_eval_tdmap_debug` BEV-rotation pairs 루프를 그대로 본뜸). 실제 h1 eval로 재검증해보니 여전히 안 맞았고, 조사 결과 **구조적 버그**를 발견:

- `depth_rgb_to_bev_torch.py`(BEV 투영)는 **yaw를 입력으로 받지 않음** — `cam_pitch_deg`만 받고, `X_w`(forward, 로봇 로컬 프레임)가 항상 이미지 위쪽에 매핑됨 (모듈 상단 주석: `World: X=forward, Y=left, Z=up`). 즉 BEV 이미지는 **이미 "robot forward = 위쪽"으로 고정**돼 있고 world yaw와 무관함.
- `topdown_camera_500`(`internnav/env/utils/internutopia_extension/robots/h1.py` L24-30)은 반대로 **world-fixed orientation**(고정 quaternion, 위치만 로봇을 따라감) — 그래서 `topdown_rgb`는 world 고정 프레임이고, 로봇이 도는 만큼 yaw로 돌려줘야 "forward=위쪽"이 됨.
- 첫 버전은 **이미 forward=up인 BEV를 또 yaw로 회전**시켜서, yaw≠0인 스텝마다 BEV를 오히려 안 맞게 만들었음.

참고로 `scripts/visualization/validate_depth_to_bev.py`(원래 "검증된 회전 컨벤션"으로 문서화돼 있었음)도 실제 실행되는 `run_validation → save_comparison_pair` 경로가 BEV를 yaw로 돌리는 동일 패턴이라, 이 스크립트 자체도 미검증 상태였을 가능성이 높음. `save_panel`(반대로 GT를 돌리는, 올바른 방향)은 `run_validation`에서 호출되지 않는 dead code.

**수정**: `save_isaac_bev_world_debug`에서 `yaw_rad`/`rot_offset_deg` 파라미터를 완전히 제거하고, 회전 없이 BEV jpg를 그대로 `_w3_bev_world_{depth_src}.jpg`로 복사만 하도록 변경. `internvla_n1_policy_unified.py` 호출부도 인자 정리.

## 바뀐 파일

| 파일 | 변경 |
|---|---|
| `internnav/model/utils/isaac_topdown_debug.py` | `save_isaac_bev_world_debug` 신규 추가 → 이후 회전 로직 제거(복사만); `_rotate_by_yaw` 헬퍼 추출(리팩터, 동작 불변) |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy_unified.py` | `_save_isaac_topdown_debug`에서 S1 BEV(`s1_view='bev'`, `s1_combine in ('replace','concat')`)일 때 `save_isaac_bev_world_debug` 호출 |
| `scripts/debugging/unit_test/test_isaac_topdown_debug.py` | 잘못된 기대치("BEV와 topdown이 동일 회전")를 검증하던 테스트 삭제, yaw-불변성 검증 테스트로 교체 + 실제 h1 캡처 이미지 fixture로 회귀 테스트 추가 |
| `scripts/debugging/unit_test/fixtures/isaac_h1_step000003_bev_gt.jpg` | 신규 — 실제 h1 eval에서 캡처된 BEV jpg (3KB), Isaac Sim 재실행 없이 회귀 검증용 |

## 검증

```
/workspace/isaaclab/_isaac_sim/python.sh -m pytest scripts/debugging/unit_test/test_isaac_topdown_debug.py scripts/debugging/unit_test/test_unified_image_provider.py -q -p no:launch_testing -p no:launch_ros
```
- `-p no:launch_testing -p no:launch_ros`: 이 환경의 ROS2 `launch_testing` pytest 플러그인이 무관한 `scripts/debugging/test_prompt_dump.py`(깨진 import)까지 강제로 collect하려다 실패하는 기존 문제를 우회 (내 변경과 무관, pre-existing).
- 23개 전부 통과.

## Isaac Sim 실행 로그 (h1 eval 재실행 시도)

방향 확인을 위해 실제로 h1 eval을 1회 재실행(`--no-train --model-path checkpoints/image_base/s1.bev.rgb.concat_s2.fpv/checkpoint-1 --max-steps 2 --debug-dir logs/input_v0.1/image_base/s1.bev.rgb.concat_s2.fpv_worldalign`)했고, 중간에 이전 세션이 `--debugpy eval`로 GPU를 물고 멈춰있던 걸 발견해서 정리 후 재실행 — 진행 중이던 걸 사용자 요청으로 중단(kill)함. 이 실행에서 저장된 실제 이미지가 현재 unit-test의 fixture(`isaac_h1_step000003_bev_gt.jpg`)로 재사용됨.

## 범위 밖으로 플래그만 한 것

1. **Habitat `_save_eval_tdmap_debug`의 BEV-rotation `pairs` 루프**(`habitat_vln_evaluator_unified.py` L250-253)도 구조적으로 동일한 버그일 가능성 높음 — 이번 세션에서 만든 코드가 아니라 손대지 않음. 확인 필요.
2. **`rotate_topdown_world`(topdown/gt_cam 쪽)의 `rot_offset_deg=0.0`이 실제로 맞는 값인지**는 미검증 — `topdown_camera_500`의 고정 quaternion과 Isaac 카메라→이미지 축 관례를 분석하거나 실제 sim으로 눈으로 확인해야 함. 이번 범위 제외.
3. **RGB/depth desync 버그**(`.claude/tasks/260709_bev_gt_color_orientation_bug_result.md`)는 여전히 미해결 — 이번 회전 수정과 무관하게, `bev_gt`의 색/구조 자체가 `fpv`와 다른 프레임을 반영하고 있어 content 자체는 계속 안 맞을 수 있음. 회전 컨벤션 버그와는 별개 원인.
