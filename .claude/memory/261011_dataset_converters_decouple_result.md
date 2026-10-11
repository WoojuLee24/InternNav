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

## 리포트 재생성 대조 (2026-10-11 추가 — 분리 검증의 결정적 근거)

막혀 있던 건 habitat이 아니라 **`scikit-image` 하나**였다. 체인:
`07_validate_pixel_goal.py` → `embodiment_augment.py` → `gs_vlnpe/04_render_obs.py` → `import skimage`.
3dloader 15개 중 habitat이 실제로 필요한 건 4개(`03`·`04`·`05`·`05b`)뿐이고 `07`은 안 쓴다.

```
/workspace/isaaclab/_isaac_sim/python.sh -m pip install scikit-image   # 0.26.0
timeout 1800 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/3dloader_vlnce/07_validate_pixel_goal.py --scene 17DRP5sb8fy --episode 0 --rig 125cm_30deg --r_bs 0.15,0.25,0.35,0.50 --goal_adjust shift --out_dir logs/embodiment_augment/pixelgoal_shift_verify
```

| 인자 | 뜻 |
|---|---|
| `--rig 125cm_30deg` | **기본값 `125cm_0deg`는 샘플 0개가 나온다**(V0 0/0, V1 nan). pixel goal은 룩다운(pitch_2) rig에 정의돼 있다. 분리 전에도 동일 — 기존 조건이다 |
| `--goal_adjust shift` | 채택된 모드(#07a) |
| `--r_bs` | 로봇 반경 사다리 |

**결과 — `git worktree`로 분리 직전 커밋(`d6329fa^`)을 꺼내 같은 명령을 돌려 대조:**

| | 분리 전 | 분리 후 |
|---|---|---|
| V0 identity @ r_b=0.1 | unchanged **26/26**, 0.00 px | **동일** |
| V1 max | **2.25 px** | **동일** |
| V2 monotonic | True(12) | **동일** |
| V3 reach max | 3.9 cm | **동일** |
| 프레임별 r_b 사다리 6행 | unchanged→adjusted→adjusted_retreat→rejected | **동일** |
| `report.html` | — | **차이 0줄**(경로·시각 제외) |
| 생성 이미지 12장 | — | **12/12 바이트 동일** |

V0 26/26 · V1 2.25 px는 `3dloader_vlnce/reports.md`의 공통 게이트 기록(**V0 0.00 px 26/26 · V1 2.25 px**)과도 일치한다.
V3만 문서의 2.6 cm와 다른 3.9 cm인데, **분리 전 코드도 3.9 cm**라 이번 변경과 무관하다(문서 수치는 다른 인자 조합에서 나온 것).

## 알아둘 것
- `scikit-image`는 **repo 의존성에 선언돼 있지 않다**(`pyproject.toml`에 없음). `gs_vlnpe/04_render_obs*.py`,
  `04d/04e/05b/05c`가 SSIM 때문에 쓰는데 설치는 수동이다. 새 컨테이너에서 같은 데서 막힌다.
- habitat을 이 컨테이너에 넣는 건 권하지 않는다 — repo가 habitat 전용 Dockerfile을 따로 두고 있고
  (`docker/Dockerfile-torch2.9.1-cuda13.0-habitat-jazzy-5090`), `install_habitat_5090.sh`는
  `INTERNNAV_DIR=/ws/src/InternNav/InternNav`·`DATA_SRC=/home/irteam/data-vol1`을 가정해 이 머신에선 1단계에서 죽는다.
  게다가 57행이 `pip uninstall llvmlite -y`라 Isaac 쪽을 건드린다.

## 남은 것
- habitat이 필요한 `03`·`04`·`05`·`05b`는 habitat 컨테이너에서 확인할 것. 이 4개는 `vlnce_align`/`vlnce_rig`
  import 한 줄씩만 바뀌었고 AST로 심볼 해결은 확인했다.
