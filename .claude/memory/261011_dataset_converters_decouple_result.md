# gs_vlnpe ⇄ 3dloader 양방향 의존 분리 — 2026-10-11

## 목적
`gs_vlnpe`(데이터 생성)와 `2d/3dloader_vlnce`(embodiment augmentation)가 서로를 `sys.path`로
끌어다 써서 **양방향 의존**이었다. 한 방향(`loader → gs_vlnpe`)으로 고정해 gs_vlnpe만 떼어내도
vln_ce 생성이 돌게 만든다.

## 무엇이 달라졌나

가른 기준: **`r_b`(로봇 반경)를 인자로 받으면 augmentation, 아니면 vln_ce 포맷 기하.**

| 전 | 후 | 이유 |
|---|---|---|
| `3dloader_vlnce/vlnce_align.py` | **`gs_vlnpe/vlnce_align.py`** | `r_b` 안 받음 — vln_ce pose를 mesh 좌표로 푸는 포맷 코드 |
| `3dloader_vlnce/pixel_goal_utils.py` 의 투영·가시성부 | **`gs_vlnpe/vlnce_rig.py`** (신규 130줄) | 〃 (`intrinsics_for_rig`, `rig_height_m`, `goal_world_from_poses`, `project_to_pixel`, `check_visible`) |
| 〃 의 `clearance_at`/`_heading_at_end`/`adjust_goal` | **`3dloader_vlnce/goal_adjust.py`** | `r_b`를 받음 — aug 전용 |

`gs_vlnpe`의 `_3D = .../3dloader_vlnce` sys.path 삽입 3곳을 전부 제거했다.
`git mv`로 옮겨 두 파일 모두 **이력이 보존**된다(`git log --follow`).

수정한 import 지점 **15개 파일 / 41 심볼**. 함수 본문은 한 줄도 안 고쳤다.

## 검증

```
/usr/bin/python3 scripts/dataset_converters/3dloader_vlnce/goal_adjust.py
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnce_rig.py
```

| 게이트 | 결과 |
|---|---|
| `vlnce_rig` self-check (V1 투영식, 실데이터) | **PASS** — N=1701, MAE u=1.22 v=0.94 px, p95 2.31, >5px 0.00% — **분리 전 문서값과 동일** |
| `goal_adjust` self-check (retreat/shift/nearest/unchanged) | **PASS** — shift lateral 0.05 m, 진행방향 성분 0.0 |
| AST로 모든 `from vlnce_rig/goal_adjust/vlnce_align import X` 해결 확인 | 41/41, 미해결 0 |
| py3.12 구문 검사 전 파일 | 오류 0 |
| 모듈 19개 실제 import | 6 OK / 13 FAIL — **변경 전 baseline과 완전히 동일** (skimage 미설치, py3.11 f-string 한계) |

⚠️ `gs_vlnpe/apply_real/05_export_vlnce.py:59`에 `_3D` 참조가 한 곳 더 있어 삭제 시 NameError가 날
뻔했다. 정규식이 `(str(_3D), ...)` 형태(쉼표 앞)를 못 잡아서다 — grep 재확인으로 발견했다.

## 남은 것
- 리포트 재생성 검증은 못 했다 — 이 컨테이너에 `skimage`가 없어 `3dloader` 검증 스크립트가
  변경 전에도 안 돈다. habitat 환경에서 `07_validate_pixel_goal.py`를 한 번 돌려 확인할 것.
