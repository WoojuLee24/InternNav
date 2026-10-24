# nav3d — navmesh 로 계단·여러 층을 다루는 경로 데이터 만들기

씬 mesh 에서 **navmesh**(로봇이 걸을 수 있는 면을 3D 다각형으로 만든 지도 — habitat/Recast)를 만들어 층과 계단을 한 지도에 담고,
그 위에서 경로를 계획해 vln_ce · vln_pe 형식 학습 데이터를 만든다. 대상은 MP3D(vln_ce · vln_pe 릴리스 씬)와 노은역.
2026-10-24 이전 기록(단계별 실험 · 결과 수치)은 [HISTORY.md](HISTORY.md) — 옛 이름 그대로다.

## 용어
| 용어 | 뜻 |
|---|---|
| wp / sg 경로 | wp = R2R 경유점을 따라 계획한 경로, sg = 시작·끝만 주고 계획한 경로 (노은역은 경유점이 없어 sg) |
| 사용 가능 판정 U1~U6 | U1 경로 있음 · U2 navmesh 위 · U3 벽 여유 ≥ 로봇 반지름 · U4 큰 mesh 구멍 옆 아님 · U5 평지는 한 층·3 m 이상 · U6 출발점 둘레 바닥 평평 |
| RBQ 걷기 | 노은역 경로를 RBQ10 사족보행 로봇이 Isaac 물리로 걷는 것 (전진 중 방향 보정 1.5 · 회전 문턱 30° · 높이 스캔 필터 −0.2) |
| 방식 A / B (vln_pe) | A = 경로 위에 카메라를 바로 놓고 렌더, B = H1 로봇이 물리로 걸으며 찍기 |

## 폴더 — 한 파일은 한 종류다

| 위치 | 종류 | 뜻 |
|---|---|---|
| `run_pipeline.py` | 실행기 | 데이터셋마다 단계를 순서대로. `--list` 로 순서를 본다 |
| `vlnce/` · `vlnpe/` · `noeun/` | **단계** | 그 데이터셋의 단계. 파일 이름 `<단계><순서>_<하는 일>.py` |
| `common/` | 공용 단계 | 여러 데이터셋이 같이 쓰는 단계의 실제 코드 (데이터셋 폴더에는 **연결 파일**이 있다) |
| `*_worker.py` · `common/vlm_server.py` | 보조 | 다른 스크립트가 띄운다. 직접 돌리지 않는다 |
| `*/tools/` | 도구 | 파이프라인 밖. 필요할 때만 |
| `lib/` | 모듈 | import 만 한다 |
| `checks/` | 검사 | 결과 확인용 (`vlnce_*` · `vlnpe_*` · `noeun_*` · 그 밖은 공용) |

**번호 앞자리 = 단계** (세 데이터셋 공통): 1 준비 · 2 경로 만들기 · 3 경로 보정 · 4 걷기 · 5 렌더(데이터 저장) · 6 라벨 · 7 검사.
뒷자리 = 그 단계 안 순서. 데이터셋에 없는 단계는 건너뛴다 (vln_ce 는 3 · 4 없음).

**연결 파일**: 계산 코드 없이 `common/` 또는 `checks/` 의 스크립트에 그 데이터셋의 고정 인자(`ARGS`)를 앞에 붙여 실행만 한다.
예: `vlnce/21_refine_paths.py` = `common/refine_paths.py --dataset vln_ce`.

## 실행 — 명령 전체는 [COMMANDS.md](COMMANDS.md), 리포트 위치는 [REPORTS.md](REPORTS.md)

```
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --list
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --stage 1,2,5,7                       # mini 전체 (학습 데이터와 같게)
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --data full --stage 1,2,5,7           # full (학습 서버)
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --scenes 17DRP5sb8fy,1LXtFkjw3qL --max_episodes 3   # 테스트
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --stage 3,4 --overwrite
```

| 인자 | 뜻 | 왜 |
|---|---|---|
| `vlnce` · `vlnpe` · `noeun` | 데이터셋 | 아래 세 표 |
| `--data mini \| full` | 공개 학습 데이터 종류 (루트만 다르고 구조는 같다) | 기본 mini. 지금은 vlnce 만. full 은 logs 를 `logs/full/` 로 나눈다 |
| `--list` · `--dry_run` | 단계 목록 / 명령만 출력 | `--dry_run` 이 각 단계를 따로 돌릴 명령을 그대로 보여 준다 |
| `--stage 3,4` · `--only a,b` · `--from a` · `--to b` | 돌릴 단계 | 단계 묶음 · 이름 · 구간 |
| `--overwrite` | 출력이 있어도 돌림 | 기본은 멈춤 — 지금 logs/·data/ 결과를 실수로 덮지 않게 |
| `--scenes` · `--max_episodes` | 테스트용 제한 (씬 · 개수) | 기본 = 전부 (vlnce: 씬마다 평지 경로 N 개 + 층 이동 전부 · 라벨 N 개) |
| `--jobs` | 씬마다 도는 단계(compare_gt · refine_paths)의 동시 수 | 4 |
| `--episodes_json` | `{씬: [릴리스 ep]}` 로 일부만 | vlnce 기본 = 전부. vlnpe 는 아직 `logs/common/episodes.json` 이 필요하다 |
| `--stairs_episode_ids` | noeun straighten_stairs 가 펼 ep | 기본 86 (v5 걷기를 보고 사람이 고른 값) |
| `--label_method` · `--label_model` | 라벨 방법 · VLM | 기본 b1 · qwen3vl8b-local |

컨테이너 (모두 저장소 루트에서): **habitat** = `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` 의 `/usr/bin/python`
(habitat-sim 0.3.3 — torch2.9.1 이미지에는 없다), **isaac** = `dnwn24/internnav:torch2.9.0-cuda13.0-issacsim5.1.0-ros2-jazzy-5090` 의
`/workspace/isaaclab/_isaac_sim/python.sh` (scikit-image 필요). 실행기는 다른 컨테이너 단계에 닿으면 멈추고 이어 돌릴 `--from` 명령을 출력한다.

## 데이터셋별 단계

### vlnce — MP3D → vln_ce (habitat)
기본은 **학습 데이터와 같게**: 릴리스 학습 데이터의 모든 경로(trajectory), **지시문마다 에피소드 하나** (R2R 은 경로 하나에 지시문 보통 3 개 —
경로는 한 번 렌더하고 지시문마다 같은 프레임으로 내보낸다). wp · sg 는 따로 (`nav3d_wp` · `nav3d_sg`).
렌더된 데이터의 지시문은 아직 릴리스 사람 지시문(GT 경로 설명)이다 — 생성 경로에 맞는 지시문은 instruction labeling(61)으로 만들고 있다.

| 번호 | 스크립트 | 하는 일 | 출력 |
|---|---|---|---|
| 11 | `vlnce/11_compare_gt.py` → `common/compare_vlnce_gt.py` | 릴리스 GT 를 같은 지도·도구로 다시 계획 | `logs/common/compare_vlnce_gt/scenes/` (51 이 층 이동 표시를 읽는다) |
| 21 | `vlnce/21_refine_paths.py` → `common/refine_paths.py` | 공식 navmesh 위 wp · sg 경로 다듬기 (실행기는 `--max_flat 0` = 전부) | `logs/common/refine_paths/scenes/<씬>_traj.npz` |
| 51 | `vlnce/51_render.py` | GreedyGeodesicFollower → habitat 렌더 → 릴리스 포맷 (`--instructions all`) | `data/.../vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}` · 색인 `.../traj_data/_index/` |
| 61 | `vlnce/61_label.py` → `common/label_paths.py` | b1 라벨 (VLM 서버 `common/vlm_server.py` 필요) | `data/.../instruction_labels_nav3d/` |
| 71 | `vlnce/71_loader_check.py` → `checks/loader_check.py` | 학습 로더로 읽기 | `logs/checks/loader_check/summary_vlnce_nav3d.json` |

### vlnpe — MP3D → vln_pe (habitat + isaac)
| 번호 | 스크립트 | 하는 일 | 출력 |
|---|---|---|---|
| 11 · 21 | vlnce 와 같은 연결 | GT 비교 · 경로 | (같은 logs) |
| 51 | `vlnpe/51_h1_pairs.py` | (A) vln_pe H1 궤적 ↔ vln_ce ep 짝 (21 결과가 있어야 한다) | `logs/vlnpe/51_h1_pairs/summary.json` |
| 52 | `vlnpe/52_camera_values.py` (+ 보조 `camera_values_worker.py`) | (A) vln_pe GT 에서 카메라 높이 · 앞 거리 · pitch | `logs/vlnpe/52_camera_values/prepare.json` |
| 53 | `vlnpe/53_camera.py --stage prepare` · `--stage check` | (A) sg 경로 → mesh 바닥 → 씬별 렌더 명령 / 판정 | `logs/vlnpe/53_camera/render_cmds.sh` · `summary.json` |
| 54 | `vlnpe/54_render_camera.py` (render_cmds.sh 가 씬마다) | (A) Isaac 렌더 → vln_pe 포맷 | `data/.../vln_pe_nav3d_sg` |
| 55 | `vlnpe/55_walk_h1.py --stage prepare · walk · compare` | (B) H1 물리 걷기 → vln_pe 포맷 · 비교 | `data/.../vln_pe_nav3d_h1` · `logs/vlnpe/55_walk_h1/` |
| 71 | `vlnpe/71_loader_check.py` → `checks/loader_check.py` | (A) 학습 로더로 읽기 (53 check 가 읽는다) | `logs/checks/loader_check/summary_06q.json` |

도구: `vlnpe/tools/build_navmesh_h1.py` — H1 몸 크기 navmesh (반지름 미정 실험, 쓰면 21 에 `--navmesh_dir`). H1 설정 `vlnpe/configs/h1_vlnpe_walk_cfg.py`.

### noeun — 노은역 → vln_ce · vln_pe (v6)
경로 파일은 `logs/noeun/paths/noeun_station_paths*.json`, 걷기 입력·결과는 `logs/noeun/walk/{eps,walk}_noeun_*.json`.

| 번호 | 스크립트 | 하는 일 | 출력 |
|---|---|---|---|
| 11 | `noeun/11_check_assets.py` | 충돌 mesh ↔ 3DGS 정합 · 바닥 구멍 | `logs/noeun/11_check_assets/noeun_station/holes.csv` (22 의 U4 입력) |
| 12 | `noeun/12_build_navmesh.py` | mesh → navmesh · 층 · 계단 | `logs/noeun/12_build_navmesh/noeun_station/` |
| 21 | `noeun/21_refine_paths.py` → `common/refine_paths.py` | 층 사이 경로 다듬기 | `logs/common/refine_paths/scenes/noeun_station_traj.npz` |
| 22 | `noeun/22_select_paths.py` | 층 이동 + 평지 고르기 (U1~U5) | `paths.json` |
| 31 | `noeun/31_floor_to_mesh.py` | 점별 바닥 z 를 장면 mesh 로 | `paths_meshfloor.json` |
| 32 | `noeun/32_straighten_narrow.py` | 개찰구 같은 좁은 통로 안을 곧게 | `paths_meshfloor_straight.json` |
| 41 | `noeun/41_make_walk_inputs.py` | 걷기 입력 (0.5 m 경유점 + 출발 yaw) | `eps_noeun_all_v2.json` |
| 42 | `noeun/42_walk_rbq.py` (+ 보조 `walk_rbq_worker.py`) | RBQ 걷기 (첫 걷기 · 후보 · 최종 세 번 쓴다) | `walk_noeun_{v3,cand,v6}.json` |
| 43 | `noeun/43_replace_fallen.py gen · eps · merge` | 넘어진 ep 를 같은 종류 새 경로로 (U1~U6, 후보를 걸어 보고 고른다) | `*_v4.json` |
| 44 | `noeun/44_shift_starts.py` | U6 실패 출발점을 경로 따라 앞으로 | `*_v5.json` |
| 45 | `noeun/45_straighten_stairs.py` | 계단 구간을 곧게 | `*_v6.json` |
| 51 | `noeun/51_render_rbq.py` | 걸은 자세로 D455 렌더 → vln_ce · vln_pe | `data/.../vln_{ce,pe}_noeun_nav3d_rbq_v6` |
| 71 | `noeun/71_render_check.py` → `checks/noeun_rbq_render_check.py` | 렌더 검사 M1~M8 | `logs/checks/noeun_rbq_render_check_v6/` |
| 72 | `noeun/72_loader_check.py` → `checks/loader_check.py` | 학습 로더로 읽기 | `logs/checks/loader_check/summary_rbq_v6.json` |

도구: `noeun/tools/rerender_rbq_subset.py` (바뀐 ep 만 다시 렌더) · `noeun/tools/render_vlnce_camera.py` (예전 방식: 경로 위에 카메라를 바로 놓기).

## 공용 모듈 (`lib/`)
| 모듈 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `nav3d_utils` | 좌표 변환 · navmesh 빌드·로드 · 층·연결부 · `NavSurface` · `MeshDistance` | 전부 |
| `nav3d_path` | 경로 다듬기 (탄성 밴드 + Chaikin, 계단은 폭 가운데) | refine_paths |
| `path_quality` | 다듬은 경로 평가 · 사용 가능 판정 U1~U5 | refine_paths · noeun 22 · 43 |
| `nav3d_report` · `nav3d_render` | report.html · habitat 1인칭 렌더 | 여러 단계 · 검사 |
| `render_poses` | 경로·걷기 기록 → 카메라 자세 | vlnpe 52 · 54, noeun 51 · tools |
| `floor_mesh` | 바닥 높이를 장면 mesh 기준으로 | vlnpe 52 · 53, noeun 31 · 32 |
| `route_vertices` | 라벨용 경로 단순화 | common/label_paths |
| `nav3d_scenes` · `mesh_probe` · `walk_inputs` · `turn_space` | 노은역 씬 표 · mesh 광선 측정 · 걷기 입력 · 회전 자리 | noeun |
| `h1_tunable_controller` · `depth_fill` | H1 걷기 제어 · depth 구멍 채우기 | vlnpe 55 · vlnce 검사 |

self-check: `/usr/bin/python scripts/dataset_converters/nav3d/lib/nav3d_utils.py`

## 옛 이름 → 새 이름 (2026-10-24)
HISTORY.md · `.claude/memory` 의 옛 리포트는 옛 이름을 쓴다.

| 옛 스크립트 | 새 스크립트 | | 옛 스크립트 | 새 스크립트 |
|---|---|---|---|---|
| 00_mesh_gs_check | noeun/11_check_assets | | 06h_vlnpe_h1_paths | vlnpe/51_h1_pairs |
| 01_build_navmesh | noeun/12_build_navmesh | | 06v_vlnpe_check · 06v_render_isaac | vlnpe/52_camera_values · camera_values_worker |
| 02_vlnce_gt_compare | common/compare_vlnce_gt | | 06q_vlnpe_mp3d_check | vlnpe/53_camera |
| 03_refine_paths | common/refine_paths | | 06p_render_noeun_vlnpe | vlnpe/54_render_camera |
| 03h1_build_navmesh | vlnpe/tools/build_navmesh_h1 | | 06r_h1_walk_vlnpe | vlnpe/55_walk_h1 |
| 04_export_paths | noeun/22_select_paths | | 06k_make_walk_inputs | noeun/41_make_walk_inputs |
| 04b_paths_mesh_floor | noeun/31_floor_to_mesh | | 06k_parallel · 06k_rbq_walk | noeun/42_walk_rbq · walk_rbq_worker |
| 04c_straighten_narrow | noeun/32_straighten_narrow | | 06m_render_rbq | noeun/51_render_rbq |
| 04d_replace_episodes | noeun/43_replace_fallen | | 06m2_rerender_subset | noeun/tools/rerender_rbq_subset |
| 04f_fix_start | noeun/44_shift_starts | | 06_render_noeun_vlnce | noeun/tools/render_vlnce_camera |
| 04g_straighten_stairs | noeun/45_straighten_stairs | | 07_label_paths · 07_vlm_server | common/label_paths · vlm_server |
| 05b_vlnce_render_paths | vlnce/51_render | | 공용 모듈 (nav3d_utils 등) | lib/ |

검사: 01h → noeun_navmesh_height_compare · 03ab → refine_stairs_mode_compare · 03c → refine_vs_2d_compare · 03r → refine_render_views ·
03s → refine_stairs_label_check · 04e → noeun_turn_space_check · 05a → vlnce_gt_rerender_check · 05c → vlnce_compare_viewer ·
05x → vlnce_isaac_rerender_check · 06b → noeun_camera_render_check · 06k_start_check → noeun_walk_start_check · 06n → noeun_rbq_render_check ·
06o → walk_action_semantics_check · 06s · 06t · 06u · 06w → turn_rule_check · turn_rule_geodesic · turn_rule_replay_sweep · turn_threshold_bound ·
06x → noeun_walk_metrics · 07b → label_compare · 07r → label_route_words_check · 08 → loader_check (모두 `checks/`).

logs: 폴더도 같은 규칙 — `logs/<데이터셋 또는 common · checks>/<스크립트 이름>/`. 노은역 경로 · 걷기 데이터만 내용 이름
(`04_export_paths/noeun_station` → `noeun/paths`, `06k_rbq_walk` → `noeun/walk`), `05_vlnce_gt_render/episodes.json` → `common/episodes.json`.

## 정리 검증 (2026-10-24) — 처음부터, 일부 에피소드, 정리 전 코드(4faf919) vs 지금 코드
빈 logs 에서 세 데이터셋을 첫 단계부터 끝까지 같은 입력으로 돌렸다 (vlnce · vlnpe: 2 씬 5 ep, noeun: 경로 20 · 렌더 3 ep).
지금 코드는 `run_pipeline.py` 로, 정리 전 코드는 같은 명령을 옛 이름으로 바꿔서. 출력 파일마다 비교 (json 안 경로 · 실행 시간은 맞춘 뒤).

| 데이터셋 | 같음 | 실행마다 달라지는 것 (같은 코드 두 번도 같은 크기로 다름) |
|---|---|---|
| vlnce | 11 · 21 · 51 렌더 (이미지 3,970) · 61 라벨 문장 · 71 | — |
| vlnpe | 11 · 21 · 51 · 52 · 53 · 71 · 53 판정, 54 의 자세 · action · depth | 54 rgb (평균 0.25/255, 같은 코드끼리 0.28) · 55 H1 걷기 NE (성공/실패는 같음) |
| noeun | 11 ~ 45 (RBQ 걷기 프레임까지 전부) · 51 vln_ce (이미지 1,072) · 51 vln_pe 자세·depth · 71 판정 · 72 | 51 vln_pe rgb (0.29/255) · 12 의 단면 그림 1 장 (seed 없는 점 찍기) |

43 은 걷기가 모두 성공해 바꿀 ep 가 없어서, 양쪽 첫 걷기 결과에 같은 2 ep 를 실패로 표시하고 돌렸다.
검증 사본의 `data/GS_USDZ/Subway/noeun_station_collision.usdz` 는 `/ws/assets/GS_USDZ/` 의 같은 파일에 링크했다 (이 PC 의 data 마운트에 없다).

## 주의
- `--overwrite` 는 실행기의 "출력 있음" 멈춤만 건너뛴다. 렌더 스크립트(vlnce 51 · vlnpe 54 · noeun 51)는 덧붙여 섞이지 않게 출력 폴더가 있으면
  스스로 멈추므로, 다시 렌더하려면 그 데이터셋 폴더를 먼저 옮기거나 지운다.

## 아직 안 되는 것
- **노은역 RBQ v6 데이터 라벨**: `common/label_paths.py --sources noeun` 은 `noeun/tools/render_vlnce_camera.py` 의 `frames_meta.json`
  (카메라 자세 · 몸 높이 · yaw)을 읽는데, `noeun/51_render_rbq.py` 는 `render_meta.json` 에 그 값을 남기지 않는다.
- vlnpe 는 아직 `logs/common/episodes.json` (4 씬 29 ep 시험 표본, 만든 스크립트 없음) 만 처리하고 sg 만 만든다 — 나중에 vlnce 처럼
  학습 데이터와 같게 · mini/full · wp 추가.
- full 은 이 PC 에 없어 `--dry_run` 경로까지만 확인했다 (씬 구성이 mini 와 같은지 모름).
- instruction labeling(61)은 만드는 중이다. 렌더된 데이터의 지시문 자리에 라벨을 넣는 단계는 아직 없다.
- 걷기(42 · 55)는 물리라 다시 돌리면 결과가 조금 다를 수 있다 → 43 이 바꿀 ep, 45 가 펼 ep 가 달라질 수 있다.
