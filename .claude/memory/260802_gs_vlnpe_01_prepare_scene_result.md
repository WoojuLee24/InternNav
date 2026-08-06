# `01_prepare_scene.py` (M1.1) — 완료

**씬 정합 게이트.** 이 씬의 mesh가 GT와 같은 world 프레임인지 판정하고, 통과하면 다음 스크립트가
읽을 `scene_meta/<scene>.json`을 남긴다.

> 기하·좌표 규약과 과거 오진 이력은 `understanding_gs_vlnpe_fpv_bev_geometry.md` 참고.

## 원안에서 범위가 줄어든 이유

원안은 "mesh를 Z-up·1m로 정규화 + `scene.usd` 생성"이었는데 실측 결과 **둘 다 불필요**했다:

- mesh가 이미 Z-up · 1.0=1m · GT와 동일 원점
- `fixed.usd`가 65개 씬 전부에 이미 존재 (`generate_episode.py:15`가 로드하는 그 파일, `scene_scale=(1,1,1)`)
- `bounds`는 `trimesh.load(...).bounds` 한 줄이라 02가 직접 구해도 됨

남은 가치는 **게이트 하나** — 02(집 전체 0.05m voxelize, 무거움) 전에 수십 초로 씬을 거른다.

## 실행

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py --scene 17DRP5sb8fy
```

| 인자 | 기본값 |
|---|---|
| `--data_root` / `--mesh_root` / `--usd_root` | GT / `data/scene_data/mp3d_n1` / `data/scene_data/mp3d_pe` |
| `--scene` / `--num_episodes` / `--frames_per_episode` | `17DRP5sb8fy` / `3` / `4` |
| `--pose_convention` | `cam2world_gl` — **negative test 전용**으로만 바꾼다 |
| `--out_dir` / `--log_dir` | `scripts/dataset_converters/gs_vlnpe` / `logs/gs-vlnpe` |

## 검증 4가지 (하나라도 실패하면 non-zero exit)

| # | 검증 |
|---|---|
| ① | obj bounds ↔ `fixed.usd` bounds 일치 (tol 1e-3, `pxr.UsdGeom.BBoxCache`) |
| ② | up-axis Z + 씬 extent 자릿수(1~100 m) — 100배/0.01배 스케일 오류만 잡는다 |
| ③ | **GT depth→world→mesh 표면거리 median < 1 mm** (`check_against_scene_mesh`) — **핵심** |
| ④ | GT 장애물·궤적이 bounds 안 + 카메라 높이(= `h_b`)가 0.2~2.5 m |

GT 장애물은 **원본 `meta/pointcloud.ply`**에서 색 필터로 뽑는다(`geometry_utils.load_gt_obstacle_points`).
필터는 데이터로더의 `NavDP_Base_Datset.process_obstacle_points`와 동일. `pointcloud_obstacle.npy`는
그 후처리의 로컬 캐시(릴리스 자산 아님)라 읽지 않는다.

기하 계산은 전부 `geometry_utils.py` 재사용. 고유 로직은 USD bounds 비교·GT 포함 여부·floorplan 시각화.

## 출력

| 경로 | 내용 |
|---|---|
| `scene_meta/<scene>.json` (canonical) | `scene_id`, `mesh_path`, `usd_path`, `bounds_min/max`, `up_axis`, `scale`, `usd_stage`, `floor_z`, `gt_camera_z`, `gt_robot_params`, `pose_convention`, `frame_alignment`, `gt_extent_check`, `bounds_check`, `passed` |
| `logs/gs-vlnpe/01_prepare_scene/<scene>/report.html` + `floorplan_*.jpg` | 검증 요약표 + floorplan blink 4-state |

**씬별 파일**로 둔 이유: 단일 파일이면 씬을 바꿔 돌릴 때 덮어쓴다(00에서 실제로 겪은 마찰).
다음 스크립트가 읽는 값 — 02: `bounds`·`mesh_path` / 03: `floor_z` / 04: `usd_path`·`h_b`.

## 시각화

mesh floorplan(높이 0.15~2.0 m 단면, 회색) 위에 **같은 grid**로 blink:
① mesh만 ② +GT 장애물 점군 ③ +GT 카메라 궤적 ④ +depth→world 점.

육안 확인: 궤적이 벽을 통과하지 않고 복도를 따라가고, depth 복원점이 의자·벽 윤곽에 얹힌다.

## 실행 결과

| 씬 | 에피소드별 mesh 표면거리 median | h_b / floor_z | 판정 |
|---|---|---|---|
| `17DRP5sb8fy` (52 ep, 191,625 verts) | 0.000029 / 0.000032 / 0.000031 m | 0.700 / −0.0139 | **PASS** (exit 0) |
| `s8pcmisQ38h` (99 ep, 261,701 verts) | 0.000028 / 0.000030 / 0.000028 m | 0.926 / −0.0015 | **PASS** (exit 0) |

**Negative test** (게이트가 항상 통과만 내지 않음을 보장):

| 씬 | 틀린 규약 | median | 판정 |
|---|---|---|---|
| `17DRP5sb8fy` | `cam2world` | 0.419 / 0.198 / 0.235 m | **FAIL** (exit 1) |
| `s8pcmisQ38h` | `world2cam` | 4.845 / 8.436 / 5.025 m | **FAIL** (exit 1) |

## 참고

- Artifact: https://claude.ai/code/artifact/e7f57d22-bfe9-449f-bd82-ce416c245013
- 설계: `docs/execution-staged.md` M1.1 절
