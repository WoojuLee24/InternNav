# nav3d — navmesh 로 계단·여러 층을 다루는 3D 경로 계획

`gs_vlnpe` 의 2D occupancy 경로(한 층 단면)는 계단으로 층을 오가는 경로를 못 만든다.
nav3d 는 씬 mesh 에서 **navmesh**(로봇이 걸을 수 있는 면을 3D 다각형으로 만든 지도 — habitat/Recast)를 만들어
층과 계단을 한 지도에 담고, 그 위에서 경로를 계획한다. 노은역(B2·B1·지상)이 1차 목표다.

- **2D occupancy + 계단 연결 방식과는 분리해서 진행한다**(다른 세션). gs_vlnpe 파일은 고치지 않는다.
- 컨테이너 (모두 저장소 루트에서): habitat 단계 = `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` 의 `/usr/bin/python`
  (habitat-sim 0.3.3 — torch2.9.1 이미지에는 없다), isaac 단계 = `dnwn24/internnav:torch2.9.0-cuda13.0-issacsim5.1.0-ros2-jazzy-5090` 의
  `/workspace/isaaclab/_isaac_sim/python.sh`.
- **2D 방식과 비교** (같은 에피소드 · 같은 자, vln_ce): 둘 다 통과 평지 nav3d 99.3 % vs 2D 82.9 %, 층 이동 99.8 % vs 54.2 %(2D plan) — 맨 아래 절.

## 용어

| 용어 | 뜻 |
|---|---|
| navmesh | 로봇이 서 있을 수 있는 면. 벽에서 `agent_radius` 만큼 이미 깎여 있다 |
| 평면(level) | navmesh 의 수평 면적이 몰린 높이. 바닥이 완만히 기울면 한 층이 평면 2개가 된다 |
| 층(storey) | 높이 차 1.5 m 미만 평면끼리 묶은 것 (노은역: S0 B2 승강장 · S1 B1 대합실 · S2 지상 출구) |
| 연결부(connector) | 두 층을 잇는 기운 navmesh 덩어리 = 계단·에스컬레이터·경사로 |
| stepness | 연결부 위에서 원본 mesh 표면 높이를 평면으로 맞춘 잔차 RMS. 톱니면 계단(≥ 0.05 m) |
| 섬(island) | navmesh 에서 서로 이어진 덩어리. 다른 섬끼리는 경로가 없다 |
| 여유(clearance) | 경로 점(바닥 + 0.5 m, 바닥 + 키 − 0.05 m)에서 원본 mesh 까지 3D 거리 |

## 좌표 규약 (실측)

우리 = Z-up. trimesh 로 Z-up mesh 를 glb 로 내보내 habitat 에 올리면 **habitat (x, y, z) = 우리 (x, z, −y)**.
미리 Y-up 으로 돌려 넣으면 오히려 틀린다. `01` 의 V1 이 매번 검증한다(틀린 규약이면 6 m 어긋남).

## 실행 — `run_pipeline.py`

대상(vlnce · vlnpe · noeun)마다 단계를 순서대로 돌린다. 단계는 아래 번호 스크립트를 **입력·출력 경로를 다 적어서** 부를 뿐이라,
스크립트를 따로 돌려도 똑같다 (`--dry_run` 이 그 명령을 그대로 보여 준다).

```
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --list
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --dry_run
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --from floor_to_mesh --to straighten_stairs --overwrite
```

| 인자 | 뜻 | 왜 이렇게 |
|---|---|---|
| `vlnce` · `vlnpe` · `noeun` | 대상 | 아래 세 표 |
| `--list` · `--dry_run` | 단계 목록 / 명령만 출력 | 돌리기 전에 확인 |
| `--only a,b` · `--from a` · `--to b` | 돌릴 단계 고르기 | 한 단계만, 또는 구간만 |
| `--overwrite` | 출력이 이미 있어도 돌린다 | 기본은 멈춤 — 지금 logs/·data/ 결과를 실수로 덮지 않게 |
| `--scenes` · `--jobs` | 씬마다 도는 단계(02 · 03)의 씬 · 동시 수 | 기본 = vln_ce 릴리스 61 씬 전부 · 4 (이전 기록과 같음) |
| `--label_method` · `--label_model` | 07 라벨 방법 · VLM | 기본 `b1` · `qwen3vl8b-local` (이 PC 의 로컬 서버, `07_vlm_server.py`) |

컨테이너가 다른 단계(habitat ↔ isaac)에 닿으면 멈추고, 다른 컨테이너에서 이어 돌릴 `--from <단계>` 명령을 출력한다.

## 폴더

| 위치 | 무엇 |
|---|---|
| `NN_*.py` (최상위 번호) | 데이터를 만드는 단계. 출력이 다음 단계 입력인 검사(00 · 02 · 06h · 06v)도 여기 |
| `checks/` | 검증·실험 스크립트 (출력을 다른 단계가 읽지 않는다). 06n · 08 은 파이프라인 마지막 검사로 부른다 |
| 번호 없는 `.py` | 공용 모듈. `nav3d_utils` (좌표·navmesh·층·`NavSurface`·`MeshDistance`) · `nav3d_path` (다듬기) · `nav3d_render` · `nav3d_report` · `floor_mesh` · `depth_fill` · `route_vertices` · `turn_space` · `h1_tunable_controller` |
| ″ (2026-10-23 옮김) | 번호 스크립트끼리 importlib 로 꺼내 쓰던 함수 — `nav3d_scenes` (01) · `path_quality` (03 · 04) · `mesh_probe` (04c · 04d) · `walk_inputs` (04d) · `render_poses` (06 · 06p · 06m). 원래 파일도 같은 이름을 그대로 노출한다 |

## 단계 (대상별)

용어: **wp** = R2R 경유점을 따라 계획한 경로 · **sg** = 시작·끝만 주고 계획한 경로 (노은역 방식) ·
**U1~U5** = 04 사용 가능 판정 (경로 있음 · navmesh 위 · 벽 여유 · 큰 구멍 옆 아님 · 평지는 한 층) · **U6** = 출발점 둘레 바닥 평평 (04d) ·
**걷기** = 06k RBQ 사족보행 물리 시뮬레이션 (전진 중 방향 보정 1.5 · 회전 문턱 30° · 높이 스캔 필터 −0.2).

### vlnce — MP3D 61 씬 → vln_ce (habitat)
| # | 단계 | 스크립트 | 출력 |
|---|---|---|---|
| 1 | `gt_compare` | `02_vlnce_gt_compare.py` (씬마다 + `--aggregate`) | `logs/02_vlnce_gt_compare/scenes/` (05b · 06q 가 층 이동 표시를 읽는다) |
| 2 | `refine_paths` | `03_refine_paths.py --dataset vln_ce` (씬마다 + `--aggregate`) | `logs/03_refine_paths/scenes/<씬>_traj.npz` (wp · sg) |
| 3 | `render_habitat` | `05b_vlnce_render_paths.py` | `data/.../vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}` |
| 4 | `label` | `07_label_paths.py --method b1` (VLM 서버 필요) | `data/.../instruction_labels_nav3d/` |
| 5 | `loader_check` | `checks/08_loader_check.py` | `logs/08_loader_check/summary_vlnce_nav3d.json` |

### vlnpe — MP3D → vln_pe, 두 방식 (habitat + isaac)
1 `gt_compare` · 2 `refine_paths` 는 vlnce 와 같다.

| # | 단계 | 스크립트 | 출력 |
|---|---|---|---|
| | **(a) 경로 위에 카메라를 바로 놓기** | | |
| 3 | `h1_pairs` | `06h_vlnpe_h1_paths.py` | `logs/06h_vlnpe_h1/summary.json` (06q 입력) |
| 4 | `camera_values` | `06v_vlnpe_check.py --stage prepare` | `logs/06v_vlnpe/prepare.json` (카메라 높이·pitch, 06q 입력) |
| 5 | `camera_prepare` | `06q_vlnpe_mp3d_check.py --stage prepare` | `logs/06q_vlnpe_mp3d/render_cmds.sh` |
| 6 | `camera_render` (isaac) | `render_cmds.sh` (씬마다 `06p_render_noeun_vlnpe.py`) | `data/.../vln_pe_nav3d_sg` |
| 7 | `camera_loader_check` | `checks/08_loader_check.py` | `logs/08_loader_check/summary_06q.json` |
| 8 | `camera_check` | `06q_vlnpe_mp3d_check.py --stage check` | `logs/06q_vlnpe_mp3d/summary.json` |
| | **(b) H1 이 물리로 걷기** | | |
| 9 | `walk_prepare` | `06r_h1_walk_vlnpe.py --stage prepare` | `data/.../vln_pe_nav3d_raw` |
| 10 | `walk_h1` (isaac) | `06r ... --stage walk --split nav3d_gt` · `nav3d_sg` | `data/.../vln_pe_nav3d_h1` |
| 11 | `walk_compare` | `06r ... --stage compare` | `logs/06r_h1_walk/summary.json` |

H1 몸 크기 navmesh (`03h1_build_navmesh.py` → `03 --navmesh_dir`) 는 아직 반지름을 정하지 않은 실험이라 파이프라인 밖이다 (기본 = 공식 navmesh, 이전과 같음).

### noeun — 노은역 → vln_ce · vln_pe (v6)
경로 파일은 `logs/04_export_paths/noeun_station/noeun_station_paths*.json`, 걷기 입력·결과는 `logs/06k_rbq_walk/{eps,walk}_noeun_*.json`.

| # | 단계 | 스크립트 | 출력 |
|---|---|---|---|
| 1 | `mesh_gs_check` | `00_mesh_gs_check.py` | `logs/00_mesh_gs/noeun_station/holes.csv` (U4 입력) |
| 2 | `build_navmesh` | `01_build_navmesh.py` | `logs/01_build_navmesh/noeun_station/noeun_station.navmesh` |
| 3 | `refine_paths` | `03_refine_paths.py --scene noeun_station` | 층 사이 경로 100 |
| 4 | `export_paths` | `04_export_paths.py` | `paths.json` (층 이동 100 + 평지 100, U1~U5) |
| 5 | `floor_to_mesh` | `04b_paths_mesh_floor.py` | `paths_meshfloor.json` (바닥 z = 장면 mesh) |
| 6 | `straighten_narrow` | `04c_straighten_narrow.py` | `paths_meshfloor_straight.json` (개찰구 안 곧게) |
| 7 | `make_walk_inputs` | `06k_make_walk_inputs.py` | `eps_noeun_all_v2.json` (0.5 m 경유점 + 출발 yaw) |
| 8 | `walk_first` (isaac) | `06k_parallel.py` | `walk_noeun_v3.json` → 넘어진 ep 고르기 |
| 9 | `replace_candidates` | `04d_replace_episodes.py gen` | `paths_cand.json` (넘어진 ep 마다 후보 3, U1~U6) |
| 10 | `candidates_fix_paths` | `04b` → `04c` | `paths_cand_meshfloor_straight.json` |
| 11 | `candidates_walk_inputs` | `04d_replace_episodes.py eps` | `eps_noeun_cand.json` |
| 12 | `walk_candidates` (isaac) | `06k_parallel.py` | `walk_noeun_cand.json` |
| 13 | `replace_merge` | `04d_replace_episodes.py merge` | `*_v4.json` (paths · eps · walk) |
| 14 | `shift_unsafe_starts` | `04f_fix_start.py` | `*_v5.json` (U6 실패 출발점을 앞으로) |
| 15 | `straighten_stairs` | `04g_straighten_stairs.py --episode_ids 86` | `*_v6.json` (계단 구간 곧게) |
| 16 | `walk` (isaac) | `06k_parallel.py ... --record trigger` | `walk_noeun_v6.json` |
| 17 | `render` (isaac) | `06m_render_rbq.py` (vlnce · vlnpe) | `data/.../vln_{ce,pe}_noeun_nav3d_rbq_v6` |
| 18 | `check_render` | `checks/06n_check_rbq_render.py` | `logs/06n_check_rbq_render_v6/summary.json` |
| 19 | `loader_check` | `checks/08_loader_check.py` | `logs/08_loader_check/summary_rbq_v6.json` |

- **재현 확인 (2026-10-23)**: habitat 단계 5~7 · 9~11 · 13~15 를 `run_pipeline.py` 로 다시 돌린 결과가 지금 logs 의 파일과 같다
  (걷기 결과 `walk_noeun_v3` · `walk_noeun_cand` 는 기존 것을 입력으로).
- 걷기는 물리 시뮬레이션이라 다시 돌리면 결과가 조금 다를 수 있다 (06r 실측: 같은 설정에 성공 20 → 21). 그러면 04d 가 바꿀 ep 도 달라진다.
  `--episode_ids 86` 은 v5 걷기를 보고 사람이 고른 값이다.
- 지금 v6 데이터는 실제로는 v5 걷기 + 다시 걸은 23 ep 를 합치고 (`walk_noeun_v6.json` 의 `merge`), 바뀐 23 ep 만 `06m2_rerender_subset.py` 로
  다시 렌더한 것이다. `run_pipeline.py` 는 같은 입력·같은 설정으로 200 ep 를 처음부터 걷고 렌더한다. 일부만 다시 할 때는 `06m2` 를 따로.
- **아직 안 되는 것**: 노은역 RBQ 데이터 라벨 — `07 --sources noeun` 은 06 렌더의 `frames_meta.json`(카메라 자세·몸 높이·yaw)을 읽는데,
  06m 은 `render_meta.json` 에 그 값을 남기지 않는다. vlnce · vlnpe 의 `logs/05_vlnce_gt_render/episodes.json` 은 만든 스크립트가 없다.

self-check: `/usr/bin/python scripts/dataset_converters/nav3d/nav3d_utils.py`

---

## 00 — mesh ↔ 3DGS 정합 · 바닥 구멍 (`00_mesh_gs_check.py`, 2026-10-14)

충돌 mesh(navmesh·depth) 와 3DGS(Isaac RGB) 가 따로 만들어졌으므로, 둘이 같은 자리에 있는지와 바닥에 구멍이 있는지 본다.
GS 는 usdz 안 `*.nurec`(gzip + msgpack, fp16) 를 직접 읽는다 — Isaac 불필요, 약 45 s.

```
/usr/bin/python scripts/dataset_converters/nav3d/00_mesh_gs_check.py --scene noeun_station
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--op_min 0.5` | 불투명 가우시안 opacity 하한 (A1·A2·A4) | 반투명 안개·뜬 점 제외 |
| `--surf_smax 0.05` | 표면 가우시안 최대 축 [m] | 큰 덩어리는 중심이 표면에 안 붙음 |
| `--near_m 0.30` | 이보다 mesh 에서 멀면 "mesh 없는 GS"(A4), 가까우면 정합 대상 | 표면 간격 0.04 + 퍼짐보다 충분히 큼 |
| `--cover_m 0.15` | A3: mesh 점 주변 GS 를 찾는 반경 | 표면 간격 + 퍼짐 |
| `--local_bad_m 0.10` | A2: 1 m 복셀 중앙 거리가 이보다 크면 국소 어긋남 | 전체 중앙 0.045 m 의 두 배 |
| `--cell_m 0.05` | 바닥 격자 | navmesh cell_size 와 같게 |
| `--hole_min_diam 0.10` | 이 이상 내접 지름이면 발·바퀴가 빠질 구멍 | 두 셀 |
| `--alpha_min 0.5` | GS 누적 불투명도 하한 (A3·B2) | 절반 이상 가려야 "보임" |
| `--nav01` | 01 출력 (층 · navmesh → 통행 구역) | 없으면 mesh 로 층만 찾음 |
| `--criteria strict` | 판정 기준 프로파일 `strict` / `relaxed` (리포트엔 둘 다 나란히) | strict = 처음 정한 기준. relaxed = 노은역 실측을 본 뒤 통과하도록 정한 기준(2026-10-15 사용자 요청) — 현 상태 기록이지 품질 보증 아님 |

완화 기준 (`relaxed`, 스크립트 `CRITERIA` · 근거 `RELAXED_WHY`) — 리포트는 이 프로파일로 낸다(`--criteria relaxed`):

| ID | 완화 기준 | 근거 (물리적 기준점) |
|---|---|---|
| G0 · A1 | 엄격과 같음 (항등 · \|t\| ≤ 2 cm · 회전 ≤ 0.05°) | 틀리면 전부 틀림 / 2 cm ≈ 3 m 거리 1~2 px |
| A1b | \|δ\| ≤ 5 cm | D455 depth 오차(2.5 m 에서 약 2 % ≈ 5 cm) 수준, 로봇 턱 0.20 m 의 1/4 |
| A2 | 어긋난 1 m 복셀 ≤ 10 % | 공간 90 % 이상이 10 cm 안. 반사 가우시안이 섞이는 휴리스틱 |
| A3 | α<0.5 면적 ≤ 15 % | 휴리스틱 흔들림 폭(10~21 %) 위쪽. 정밀 판정은 렌더 비교 |
| B1 | mesh 구멍 면적 ≤ (통행 구역 + 계단) 1 % — **개수 → 면적** | navmesh 가 구멍 둘레를 r_b 만큼 깎아 경로가 피함 → 잃는 면적이 영향 |
| B2 | 층마다 · 계단 전체 각각 ≤ 3 % | 화면만의 문제. 100 m² 당 얼룩 3 m² |
| B3 | \|중앙\| ≤ 6 cm · p95 ≤ 15 cm | p95 < 턱 0.20 m. 1.25 m 카메라에서 6 cm ≈ 3 m 앞 바닥 15 cm |

숫자는 노은역 실측을 본 뒤 고른 값이다 — 근거는 "이 정도면 쓸 만하다"는 설명이지 측정 전 설계가 아니다. 노은역: relaxed 8/8 통과 · strict 2/8.

판정 기준·용어는 스크립트 `METRICS` 표(report 의 "지표와 기준" 절과 같은 것). 리포트: https://claude.ai/artifact/GnuyBR6aSibZABUQJkBF4T 출력 `logs/00_mesh_gs/<scene>/`: `report.html`, `summary.json`, `holes.csv`,
`gs_without_mesh_clusters.csv`, 그림(`floor_S*` 구멍 지도 · `floor_offset_S*` 바닥 높이차·GS α · `align_S*` · `hole_crops` · `dist_hist`),
3D `mesh_gs_3d.html`, PLY(`mesh_gs_alpha.ply` · `gs_without_mesh.ply` · `floor_holes.ply`).

### 노은역 결과

| 검사 | 결과 | 판정 |
|---|---|---|
| G0 GS 좌표 | xform 항등 · offset 0 · 쿼터니언 wxyz (짧은 축∥법선 0.875 vs xyzw 0.825) | ✅ |
| A1 강체 어긋남 | 이동 0.0095 m (대부분 z 0.009) · 회전 0.021° | ✅ |
| A1b 표면 간격 δ | **+0.036 m — GS 가 모든 면에서 mesh 뒤**(바닥 −0.052 · 천장 −0.053 · 벽 −0.023~−0.027) | ❌ |
| A2 국소 어긋남 | 7.5 % 복셀 (가장 큰 덩어리 x −17~−4, z 1~4 서쪽 계단 홀 · 대합실 −16~−8) | ❌ |
| A3 GS 없는 mesh | 로봇 시야 높이 11.6 % (948 m²), 전체 16.2 % | ❌ |
| A4 mesh 없는 GS | 불투명의 1.3 %, 덩어리 97 · 통행 구역 위 1 (x 29, y −1, 7 cm — 작은 물체) | 보고 |
| B1 mesh 구멍 (바닥 + 계단, ≥0.10 m) | 58 개 (S0 3 · S1 11 · S2 44 · 계단 0), 면적 5.72 m² / 2,357 m² (0.24 %) | ❌ |
| B2 GS 구멍 (바닥 층마다 + 계단 전체) | 바닥 S0 1.2 % · S1 0.97 % · S2 2.7 % · 계단 0.79 % (조각 8 개, 1.2 m² 조각 최대 33 %) | ❌ |
| B3 바닥 높이 GS − mesh | 층별 중앙 −0.045 / −0.050 / −0.043 m, p95 0.09 / 0.09 / 0.13 | ❌ |
| G0c 무게중심 거리 | GS 전체 6.70 m · opacity 가중 6.96 m · mesh 상자 안 7.05 m — GS 가 mesh 밖(x 185 까지)으로 퍼져 정합 지표로 못 씀 | 보고 |

읽는 법
1. **위치는 맞다(A1).** 전체가 밀리거나 돌지 않았다.
2. **mesh 가 GS 보다 방 쪽으로 2.5~5 cm 부풀어 있다(A1b·B3).** 바닥에서는 mesh 바닥이 GS 바닥보다 약 5 cm 높다
   = 로봇 발(충돌)이 RGB 바닥보다 5 cm 위. 강체 맞춤에 δ 항을 안 넣으면 이것이 "z 로 +3 cm 밀림"으로 잘못 나온다.
3. 구멍은 **가장자리**(서쪽 계단 홀 x ≈ −19, 출구 쪽 y 20~25, 지상 S2 동쪽 x 40~60)에 몰려 있다. 대합실 · 승강장 가운데는 깨끗하다.
   S0 승강장 (−0.2, 6.6) 에 GS 바닥 구멍 3.1 m² (H1) — 반사 바닥 의심, 렌더로 확인할 것.
4. 기둥은 "닫힌 미스캔"(노랑)으로 분류 — mesh 기둥 속이 비어 있어서이며 구멍이 아니다.
5. **계단은 B1·B2 에 합산**(2026-10-15 사용자 결정, 이전 B4). 바닥 검사는 층 띠만 봐서 계단 경사 구간을 못 보므로 01 navmesh 의 층 사이 삼각형(높이 폭 ≥ 1 m 조각) 위를 navmesh 높이 기준으로 따로 쏴서 같은 두 종류(mesh 구멍 · GS 구멍)로 센다. 계단 GS 구멍은 계단 전체를 한 구역으로 모으므로 작은 조각의 높은 비율이 희석된다(33 % → 전체 0.79 %).
6. 지붕 밑으로 내려가는 계단(S2 출구 계단 아래쪽)은 위에서 쏘면 천장 뒷면만 맞아 처음엔 흰색(아무것도 없음)으로 나왔다(사용자 지적).
   천장 뒷면을 지나 계속 쏴서 위향 면이 나오면 연보라 "천장 너머 아래층"으로 칠하고, 01 navmesh 계단을 빗금으로 겹친다. 구멍 판정은 그대로.
7. A4 대합실 가운데 빨간 점은 54 % 가 바닥 아래(반사 바닥이 만든 가우시안), 33 % 가 1.8 m 위(표지판·천장 설비)다.
8. ⚠️ A3 는 기하 근사다(시점 무관, 0.15 m 안 16 개 누적 α). 정확한 판정은 로봇 카메라 자세에서 GS 를 렌더해 mesh depth 와 픽셀 단위로 비교해야 한다(미구현).

---

## 01 — navmesh 만들기 + 층·계단 찾기 + 검증

```
/usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--scene` | 씬 이름 (`SCENES` 표에 mesh·2D 지도 경로) | `noeun_station` |
| `--agent_radius 0.25` | 로봇 몸 반지름 — navmesh 를 벽에서 이만큼 깎음 | 노은역 2D(gs_vlnpe `apply_real/02`)의 `r_b` 와 같게 |
| `--agent_height 1.8` | 계획 기준 키 — 머리 위가 이만큼 비어야 걷는 면 | **사용자 결정(2026-10-14)**: 실제 로봇은 0.9 m 지만 1.8 m 기준. 비용은 아래 01h (면적 −1 %, 경로 손실 거의 0) |
| `--agent_max_climb 0.20` | 한 번에 오를 수 있는 턱 높이 | 사용자 확인(로봇 0.20 m 통과). 계단 챌판(약 0.17 m)을 넘을 수 있어야 함 |
| `--agent_max_slope 45` | 걸을 수 있는 최대 경사 | habitat 기본. 계단 평균 경사 26~28° |
| `--cell_height 0.05` | navmesh 높이 해상도 | habitat 기본 0.2 는 계단 높이를 뭉갬 |
| `--storey_gap 1.5` | 이보다 가까운 평면은 한 층 | 노은역 바닥이 끝에서 끝까지 약 0.5 m 기움 |
| `--n_pairs 200` | V5 층간 경로 시험 수 | (층, 섬) 칸마다 점을 뽑아 이어진 층 쌍에 고르게 |
| `--no_blockers` | `SCENES` 의 수동 벽(blockers)을 넣지 않음 | 기본은 넣는다 — 3DGS 가 못 살린 유리 |

출력 `logs/01_build_navmesh/<scene>/`: `<scene>.navmesh`, `<scene>.glb`, `verify.json`, `report.html`, 그림.

### 검증 기준과 노은역 결과

| 검증 | 묻는 것 | 기준 | 노은역 |
|---|---|---|---|
| 섬 라벨 | 삼각형마다 붙인 섬 번호가 `get_island` 와 같나 | ≥ 98 % | ✅ 100 % |
| V1 좌표 | navmesh 꼭짓점이 원본 mesh 표면 위에 있나 | 중앙 < 0.05 m, p95 < 0.25 m | ✅ 중앙 0.039 · p95 0.093 m (틀린 규약 6.13 m) |
| V2 층 | 각 평면 높이에 mesh 바닥(위향 수평면)이 있나 | 0.3 m 안 | ✅ 최대 0.24 m |
| V3 2D 대조 | 대합실 층 navmesh vs 2D 지도 `nav_mask_ref` | 보고 | IoU 0.18, **mesh 바닥이 있는 곳만 0.85** |
| V4 연결 | 층이 이어지나, 계단 몇 개 | 보고 | 층 3개 · 계단 3개 · 셋 다 한 섬(1,489 m²) |
| V5 층간 경로 | 다른 층 두 점 사이 `find_path`, 경로가 수평으로 navmesh 위인가 | 성공 100 %, ≥ 99 % | ✅ 200/200 · 100 % |
| V5 여유 | 경로의 mesh 3D 최소 여유 ≥ r_b − 0.05 | 보고 | ✅ 200/200 (유리 벽 추가 후. 추가 전 117/200 — 미달 83개 전부 유리 칸막이 한 곳) |

**층과 계단** (navmesh 면적, 높이는 우리 z):

| 층 | 높이 [m] | 면적 | | 계단 | 잇는 층 | 경사 | stepness | 위치 (x, y) |
|---|---|---|---|---|---|---|---|---|
| S0 B2 승강장 | −5.60 ~ −5.06 | 479 m² | | C0 | S0↔S1 | 28° | 0.136 m | (16, 8) 북쪽 승강장 |
| S1 B1 대합실 | −0.50 ~ 0.00 | 711 m² | | C1 | S0↔S1 | 27° | 0.147 m | (16, −8) 남쪽 승강장 |
| S2 지상 출구 | +6.93 ~ +8.16 | 406 m² | | C2 | S1↔S2 | 26° | 0.163 m | (−16, −21) 출구 |

층간 경로 예: 승강장 → 대합실 → 지상 약 100 m, 경로 길이/직선거리 중앙 1.55.

### 그림 (report.html 에 모두 있음)

| 파일 | 보는 것 |
|---|---|
| `*_storeys_topdown.png` | 층별 navmesh(**색 = 섬** — 3D html 과 색 뜻이 다름) + 계단(빨강) |
| `*_side.png` | 옆모습 — 층 3개가 계단으로 이어진 모습 |
| `*_connector_sections.png` | **계단 단면** — mesh 표면은 톱니(디딤판·챌판), navmesh 는 그 위 경사면 + 중간 참 |
| `*_compare_2d.png` | 2D 지도 대조 (연파랑 = 2D 만 통행 + mesh 바닥 없음) |
| `*_paths.png` | 층간 경로 6개: 위에서 본 경로 + 거리별 z(계단·참) + 여유 |
| `*_navmesh3d.html` | 3D (돌려 보기, 8 MB). **색 = 층**(파랑 S0 · 주황 S1 · 초록 S2), 빨강 = 층 사이 계단, 회색 = 층 안 경사·작은 조각. 범례 클릭으로 끄고 켬 |
| `dbg_spot_gate.png` | 여유 미달 지점 확대 |

### 알아낸 것 · 주의

1. **노은역 mesh 로 navmesh 를 만들 수 있다.** 빌드 1.6 s. `noeun_station_mid.ply` 는 이름과 달리 **역 전체**
   충돌 mesh 다(usdz 안 `mesh.ply` 와 완전히 같음, `mesh.usd` 에 변환 없음 → world = ply 좌표).
2. **계단은 자동으로 잡힌다.** navmesh 는 계단을 "중간 참이 있는 경사면"으로 만든다. 층 사이를 잇는 경사 덩어리를
   찾고, 원본 mesh 표면이 톱니인지로 계단을 확인한다(계단 0.136~0.163 m vs 같은 층 안 경사 ≤ 0.023 m).
3. **2D 지도는 바닥 없는 곳도 통행으로 본다.** 2D 만 통행인 3,140 m² 중 3,105 m² 는 mesh 에 그 높이 바닥이 없다
   (역 벽 밖). navmesh 는 바닥이 있어야 만든다 — 3dloader 에서 확인한 "다층 집에서 다른 층 허공이 새어 들어옴"과 같은 원인.
4. **투명 유리는 mesh 에 없다 → 수동 벽(blockers)으로 막는다.** 대합실 남쪽 y = −15.5 칸막이(사용자 확인: 유리)는
   기둥(x −4.25 … 3.25, 약 1 m 간격) 사이 판넬의 바닥~0.9 m 가 비어 있어 navmesh 가 그 아래로 지나갔다(여유 미달 83개).
   x −4.3~0.8 에 바닥~0.95 m 상자를 넣고, 틀 하단이 1.9 m 인 **x 0.8~1.95 는 문**으로 남겼다.
   → 여유 200/200, 대합실↔출구 경로가 x = 1.17 에서 문을 지난다(`dbg_glass.png`).
   다른 씬에서도 유리문·유리벽은 같은 방식으로 `SCENES[...]['blockers']` 에 넣는다.
5. 지상층(S2)은 출구 계단 위 16 m² 만 이어지고, 나머지(x 30~55)는 스캔이 끊긴 별도 섬들이다.
6. 에스컬레이터·엘리베이터로 보이는 연결부는 없다(충돌 mesh 에 없거나 걸을 수 없는 면).
7. 측정 함정 2개를 고쳤다: (a) 섬별 삼각형을 이어 붙이면 전체 목록과 순서가 달라 라벨이 어긋난다,
   (b) `find_path` 꼭짓점 사이 직선 z 는 계단 구간에서 허공을 지난다 → `NavSurface.track` 으로 표면에 투영.

---

### 01h — 로봇 키 0.9 m vs 계획 기준 1.8 m (`01h_height_compare.py`, 2026-10-14)

키(agent_height) = 머리 위가 이만큼 비어야 걸을 수 있는 면으로 친다. 키를 키우면 천장 · 계단 밑 · 낮은 구조물 아래가 빠진다.
노은역 navmesh 를 0.9 m · 1.8 m 로 만들고(턱 0.20 m 같음), 같은 시작 · 끝 200 쌍(03 층간 100 + 무작위 100)으로 비교했다.

```bash
/usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station --agent_height 0.9 --out_dir scripts/dataset_converters/nav3d/logs/01_height/h0.9
/usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station --agent_height 1.8 --out_dir scripts/dataset_converters/nav3d/logs/01_height/h1.8
/usr/bin/python scripts/dataset_converters/nav3d/checks/01h_height_compare.py
```

| | 0.9 m | 1.8 m |
|---|---|---|
| navmesh 면적 | 2,083 m² | 2,067 m² — 0.9 m 에서 걷던 곳 중 **20.6 m² (1.0 %)** 가 빠짐 (B2 8.2 · B1 6.4 · 지상 2.2 · 계단 3.8) |
| 계단 연결 · 층간 경로(01 V5) | 3 개 · 200/200 | 3 개 · 200/200 |

| 경로 (같은 시작·끝) | 층간 100 | 무작위 100 |
|---|---|---|
| 0.9 m 로 계획 → 0.9 m 기준 사용 가능 | 100 | 100 |
| **그 경로를 1.8 m 기준으로 재면** 사용 가능 | 79 (21 개는 1.8 m navmesh 밖을 지남) | 80 (20 개 밖) |
| **1.8 m 로 다시 계획** → 사용 가능 | **100** | **98** (2 개는 끝점이 빠진 곳에 있음) |
| 다시 계획한 길이 / 0.9 m 길이 | 중앙 1.001 · 최대 1.007 | 중앙 1.000 · 최대 1.012 |

읽는 법
- **1.8 m 로 바꿔도 잃는 것이 거의 없다** — 빠지는 곳은 벽 · 기둥 가장자리의 얇은 띠(위에 튀어나온 구조물 · 표지판)와 몇 군데 덩어리
  (`logs/01_height/compare/lost_area.png`, 3D `noeun_lost_h1.8.ply` — intensity 1.0 = 빠진 곳). 경로는 그 띠를 피해 같은 길로 간다.
- 단 **0.9 m 로 만든 경로를 1.8 m 에 그대로 쓰면 20 % 가 걸린다** — 가장자리를 스쳐 가기 때문. 반드시 1.8 m navmesh 로 계획한다.
- 머리 위 검사 자를 고쳤다: 몸 = 반지름 r_b 구 두 개(중심 0.5 m · 키 − r_b) — 위 구 꼭대기 = 키. 옛 자(중심 키 − 0.05)는 머리 위가
  키 + r_b − 0.05(1.8 m 면 2.0 m)까지 비어야 통과해, 1.8 m 로 다시 계획한 경로도 33/100 만 통과했다. 01 V5 · 03 평가에 같이 반영.
- 그래서 01 기본 키를 1.8 m 로 바꾸고 01 · 03 · 03c 를 다시 돌렸다.

---

## 02 — vln_ce GT 와 직접 비교 (좌표 · 지도 · 경로 계획)

vln_ce GT 는 habitat **공식 MP3D navmesh** 위에서 R2R `reference_path`(사람이 고른 경유점)를 최단경로 추종기
(전진 0.25 m / 회전 15° 이산 action)로 이은 것이다. 같은 지도·같은 도구로 다시 만들어 단계별로 어디서 어긋나는지 가른다.

```
ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/02_vlnce_gt_compare.py --scene {} --recompute
/usr/bin/python scripts/dataset_converters/nav3d/02_vlnce_gt_compare.py --aggregate
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--scene` | 씬 하나 (씬별 프로세스로 병렬) | 61씬 4개씩 동시, 전체 수 분 |
| `--recompute` | E: glb 에서 habitat 기본 설정으로 navmesh 를 다시 만들어 공식본과 비교 | 우리 빌더(01)가 공식본을 재현하는지 |
| `--rig 125cm_30deg` | GT 카메라 rig — 높이 1.25 m 만 써서 발 위치를 구함 | gs_vlnpe 확정 설정과 같게 |
| `--goal_radius 0.25` | 추종기 경유점 도달 반경 | 생성값 미상 — 0.1/0.5/1.0 도 action 일치 0 (아래) |
| `--aggregate` | 씬별 결과 모아 summary.json · report.html · 그림 | |

출력 `logs/02_vlnce_gt_compare/`: `scenes/<scene>.json`(에피소드별 지표), `scenes/<scene>_traj.npz`(GT·추종기·계획 궤적),
`summary.json`, `report.html`, `dist.png`, `examples.png`.

"같은 루트" = 이산 Fréchet 거리(순서를 지키며 짝지은 점 사이 최대 거리, xy) < 0.8 m — gs_vlnpe 03 과 같은 정의.
층 이동 = GT 시작·끝 높이 차 > 0.5 m (gs_vlnpe 03u 와 같은 정의).

### 결과 (61씬 · 10,684 ep · 평지 9,092 · 층 이동 1,592 · 매칭 실패 0)

| 검사 | 묻는 것 | 결과 (전체 / 평지 / 층 이동) | 판정 |
|---|---|---|---|
| A 좌표 | 우리 좌표로 올린 GT 발 위치가 공식 navmesh 위에 있나 | navmesh 에서 10 cm 넘게 떨어진 점이 **하나도 없는 ep 10,668 / 9,080 / 1,588**. 높이 차 p99 0.0002 m | ✅ (계단 위 포함) |
| A2 시작점 | 우리 변환 시작점 = raw `start_position` | 최대 0.000 m | ✅ |
| B 지도+계획 | 공식 navmesh `find_path` 거리 = 데이터셋 `info.geodesic_distance` | 1 cm 안 **10,003 (93.6 %)**, 중앙 0.000 m, p99 0.48 m | ✅ (차이는 특정 씬에 몰림 — 아래) |
| C GT 재현 (위치) | reference_path 를 추종기로 이은 경로 ↔ GT | GT 점→추종기 거리 ep 중앙의 중앙 **0.074 / 0.088 / 0.026 m** | ✅ |
| C GT 재현 (action) | action 열이 같나 | 완전 일치 24 ep, 유사도 중앙 0.69 | ⚠️ 생성기가 다름 (아래) |
| **C2 경유점 + 우리 계획** | 경유점을 주고 이웃 경유점끼리 `find_path` 로 이으면 GT 와 같은 길인가 | 위치 오차 중앙 **0.083 / 0.092 / 0.047 m**, 같은 루트 **9,321 (87.2 %) / 7,736 (85.1 %) / 1,585 (99.6 %)** | ✅ |
| D nav3d 계획 | GT 시작·끝만 주고 `find_path` → GT 와 같은 길인가 | 같은 루트 **7,827 (73.3 %) / 6,518 (71.7 %) / 1,309 (82.2 %)**, 길이비 중앙 0.92, 높이 범위 일치 10,677 | ✅ (2D occupancy 73.6 % 와 비슷) |
| E 우리 빌더 | glb → 기본 설정 navmesh 가 공식본과 같나 | 면적비 0.997~1.010, 거리 차 중앙 0.000 m, 같은 루트 7,830 (공식본 7,827) | ✅ |

### 읽는 법 · 주의

1. **좌표·지도·계획은 맞다.** A·A2·B·E 가 사실상 0 오차 — 우리 좌표 변환 `habitat (x,y,z) = 우리 (x,z,−y)` 과
   navmesh 로드·`find_path` 가 데이터셋 생성 때와 같다. 계단 에피소드 1,592 개의 GT 도 navmesh 위에 올라간다.
2. **층 이동 에피소드도 `find_path` 하나로 계단을 지나 GT 와 같은 길을 82 % 재현한다** (평지보다 높음 — 계단이 길을 강제).
   `examples.png`: 5.5 m 내려가는 에피소드에서 GT·추종기·계획 경로의 높이 그래프(계단 + 중간 참)가 겹친다.
3. **D 의 나머지 27 % 는 계획 오류가 아니다.** GT 는 사람이 고른 경유점을 지나므로 최단경로가 아니다 —
   start→goal 만 주면 다른 길이 나오는 게 정상이다(같은 루트 비율은 참고 지표, instruction 을 새로 붙이므로).
4. ⚠️ **action 열은 재현 안 된다.** GT 는 회전과 전진을 번갈아(R F R F …) 지그재그로 가고, habitat
   `GreedyGeodesicFollower` 는 회전을 몰아서 한다. 위치는 7 cm 안에서 맞으므로 생성 알고리즘(추종기 구현)만 다르다.
   도달 반경 0.1/0.5/1.0 으로 바꿔도 일치 0.
5. C 이상치(ep 중앙 > 1 m) 175 개(1.6 %): 86 % 는 GT 가 경유점을 다 지나는데 **경유점 사이에서 장애물 반대편으로 돈**
   경우(route split — 3dloader 09 에서도 확인), 14 % 는 GT 가 경유점에서 1 m 넘게 벗어난 데이터 불일치.
   씬 몇 개에 몰린다(vyrNrziPKCB 42, p5wJjkQkbXX 30, Vvot9Ly1tCj 24).
6. B 1 cm 초과 681 개도 씬 몇 개에 몰린다(2n8kARJN3HM 51, Vvot9Ly1tCj 48) — 데이터셋 생성 당시 navmesh 판과 미세하게 다른 것으로 보임.
7. 분석 함정: `0.0 or 9` 처럼 0 을 거짓으로 처리하는 집계 버그가 있었다(높이 범위 일치가 41 % 로 잘못 나옴) — 고침.

### TODO (사용자, 2026-10-13 — 아직 구현 안 함)

- **start·goal 만 주면 GT 와 같은 길이 73 % 뿐이다(D).** 경유점을 주면 87 %(C2, 계단 에피소드 99.6 %).
  → 노은역처럼 GT 가 없는 곳에서는 start→goal 로 자연스럽게 가면서 **중요 경유점(문·계단·랜드마크 등)을 찍으며**
  가는 경로 생성이 필요하다. instruction 을 새로 붙일 때 경유점이 문장의 근거가 된다.

### 다음

- 02 결과로 좌표·지도·계획 검증 완료 → 03 경로 다듬기.

---

## 03 — 경로 다듬기 (find_path → 벽에서 떼고 매끄럽게)

`find_path` 최단경로는 벽(navmesh 경계)에 딱 붙어 꺾어 돈다(경계 여유 0). 03 은 이것을 길 안쪽으로 떼고 매끄럽게 한다.

```bash
/usr/bin/python scripts/dataset_converters/nav3d/03_refine_paths.py --scene noeun_station
ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/03_refine_paths.py --dataset vln_ce --scene {}
/usr/bin/python scripts/dataset_converters/nav3d/03_refine_paths.py --aggregate
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--scene noeun_station` | 01 navmesh 에서 서로 다른 층 무작위 점 쌍(01 V5 표본) | GT 없는 씬 — 층 이동 경로만 |
| `--n_pairs 100` | 노은역 경로 수 | 층 조합 3 개 × 30+ |
| `--dataset vln_ce --scene S` | 공식 navmesh, 에피소드마다 wp(GT 경유점을 find_path 로 이음)·sg(시작→끝) 두 경로 | GT 와 비교하려고. R2R 은 경로 1 개에 지시문 3 개 → `trajectory_id` 로 중복 제거 |
| `--max_flat 10` | 씬당 평지 에피소드 수(층 이동은 전부) | 계단 경로를 충분히, 평지는 표본으로 |
| `--push_margin 0.10` | 경계에서 떼려는 목표 거리 | 벽까지 실제 ≈ 여유 + agent_radius. 더 키우면 좁은 문에서 못 밂 |
| `--aggregate` | `summary.json`, `report.html`, `examples.png`, `scatter.png` | |

### 방법 (`nav3d_path.py`)

1. 5 cm 로 촘촘히 찍고 navmesh 표면에 수직 투영(z), 점마다 바닥/계단 라벨.
2. **바닥**: 꺾인 점은 남기고 0.5 m 로 솎음 → 탄성 밴드(이웃 중점 쪽으로 당김 + 8 방향 중 여유가 커지는 쪽으로 밂) →
   Chaikin 모서리 깎기 2 회. 움직임은 "점·양옆 선분의 최소 여유가 나빠지지 않을 때만" 받아들인다.
3. **계단**: 곡선으로 깎지 않는다(비스듬히 오르면 위험). 진행 방향 수직으로 **계단 삼각형 안에서만** 폭을 재서 가운데로 옮기고,
   오프셋을 2 m 중앙값 + 1 m 평균으로 다듬는다. 한쪽이 트였으면 가운데 맞추기 안 함. 구멍(난간 기둥) 가로지르면 오프셋을 반씩 줄임.
4. **안전장치**: 구간마다 5 cm 로 다시 찍어 navmesh 위인지 확인 → 아니면 덜 다듬은 것으로 물러섬
   (바닥: Chaikin 1 회 → 0 회 → 원본, 계단: 원본, 그래도 이음매가 밖이면 옆 계단 구간도 원본).

계단 같은 곳 = 층에 안 든 삼각형을 이은 덩어리 중 높이 폭 ≥ 0.3 m · 면적 ≥ 1 m² · 평균 경사 ≥ 10°.
단 층에 안 든 **넓은 평평한 조각(≥ 4 m², 층 사이 높이의 바닥)은 뺀다** — 디딤판·참 같은 작은 평평한 조각은 남긴다.

### 검증 기준

| 검사 | 묻는 것 | 기준 |
|---|---|---|
| navmesh 위 | 다듬은 경로 모든 점(5 cm)이 navmesh 표면 위인가 | 100 % |
| 관통 | navmesh 밖 점이 있거나 mesh 3D 거리 < 0.02 m (중심이 벽면에 닿음) | 0 |
| r_b 미달 | 관통 아님 & mesh 3D 거리 < r_b (몸이 벽에 걸침). r_b = navmesh agent_radius (vln_ce 0.10 · 노은역 0.25 m) | 줄어야 함 |
| 경계 여유 p5 | navmesh 경계까지 거리 하위 5 % (양 끝 0.3 m 제외) | 커져야 함 |
| GT 같은 루트 | Fréchet < 0.8 m | 줄지 않아야 함 |
| 회전량 °/m | 0.25 m 재표본 | 보고(GT 와 비교) |

mesh 3D 거리 = 경로 점(바닥 + 0.5 m, 바닥 + 키 − 0.05 m)에서 원본 mesh 까지 거리 — navmesh 와 독립인 두 번째 자.
부호가 없어 벽 **안쪽** 깊이는 못 잰다 → 관통은 navmesh 밖 여부와 함께 본다. **GT 도 같은 자로 잰다**(기준선).

### 결과 (2026-10-14, 노은역 키 1.8 m · 머리 위 검사 수정 후 전체 재실행)

경로 하나는 한 칸에만 든다(관통 > r_b 미달 > 사용 가능).

| 묶음 | 경로 | 관통 | r_b 미달 | **사용 가능** |
|---|---|---|---|---|
| 노은역 raw | 100 | 0 | 24 (24 %) | 76 (76 %) |
| 노은역 **ref** | 100 | 0 | 0 | **100 (100 %)** |
| vln_ce sg 평지 raw / **ref** | 573 | 0 / 0 | 84 (14.7 %) / 4 (0.7 %) | 489 (85.3 %) / **569 (99.3 %)** |
| vln_ce sg 층 이동 raw / **ref** | 530 | 0 / 0 | 54 (10.2 %) / 1 (0.2 %) | 476 (89.8 %) / **529 (99.8 %)** |
| vln_ce wp 평지 raw / **ref** | 573 | 0 / 0 | 24 (4.2 %) / 0 | 549 (95.8 %) / **573 (100 %)** |
| vln_ce wp 층 이동 raw / **ref** | 530 | 0 / 0 | 6 (1.1 %) / 0 | 524 (98.9 %) / **530 (100 %)** |
| vln_ce **GT** 평지 / 층 이동 (기준선) | 573 / 530 | 0 / 0 | 2 (0.3 %) / 5 (0.9 %) | 571 (99.7 %) / 525 (99.1 %) |
| **합계 ref** (노은역 + vln_ce) | 2,306 | 0 | 5 (0.2 %) | **2,301 (99.8 %)** |

| 품질 (중앙값 raw → ref) | 노은역 | sg 평지 | sg 층 이동 | wp 평지 | wp 층 이동 |
|---|---|---|---|---|---|
| 경계 여유 p5 (m) | 0 → 0.08 | 0 → 0.06 | 0 → 0.08 | 0.06 → 0.08 | 0.06 → 0.09 |
| GT 같은 루트 | – | 414 → 425 | 436 → 459 | 481 → 482 | 528 → 528 |
| 회전 °/m (GT ≈ 25) | 4.9 → 10.3 | 9.3 → 12.8 | 12.8 → 18.4 | 14.5 → 13.2 | 17.9 → 18.6 |
| 물러선 경로(안전장치) | 0 | 16 | 13 | 10 | 7 |

- 노은역은 키 1.8 m navmesh, 몸 검사 = 반지름 r_b 구 두 개(중심 0.5 m · 키 − r_b, 01h). r_b 미달로 남은 5 개도 걸친 점은 경로의 약 2 %(중앙)뿐이다. 04 에서 판정 기준을 정한다.
- 경계 **최소** 여유는 여전히 0 근처인 경로가 많다 — 계단 navmesh 의 작은 구멍(난간 기둥·노이즈) 옆.

### 그림 (`logs/03_refine_paths/`)

- `examples.png`: 행마다 전체 평면(빨강 = 계단 같은 곳) · 가장 많이 바뀐 곳 확대 · 경로 따라 경계 여유와 높이.
- `scatter.png`: 경로별 raw(가로) vs ref(세로) — 여유 p5, 회전, mesh 여유. 대각선 위 = 커짐.
- 계단 라벨 수정 전/후는 03s (아래).

### 알아낸 것 · 함정

1. 경계 **위** 점은 여유가 0 이라 "여유가 커지는 방향" 이 없다 → 8 방향 표본으로 밂.
2. 그냥 재표본하면 find_path 꼭짓점이 빠져 문틀을 가로지른다(노은역 문) → 꺾인 점 유지.
3. 받아들이는 조건을 절대값(≥ 0.02)으로 두면 이웃이 아직 경계 위라 아무것도 못 움직인다 → "나빠지지 않음".
4. 계단 navmesh 구멍 때문에 폭 가운데가 흔들렸다 → 넓은 중앙값 + 구멍 가로지르면 오프셋 반감.
5. 문턱 같은 작은 경사 조각까지 계단으로 보면 탁 트인 바닥에서 가운데 맞추기가 돌아 경로가 2 m 밀렸다(kEZ7 같은 루트 8 → 4) → 높이 폭·면적·경사 조건.
6. **층 사이 높이의 평평한 바닥이 경사 조각과 한 덩어리로 묶여 "계단" 이 됐다**(VFuaQ 102 m² 중 87 % 평평).
   v1 은 기울어진 삼각형만으로 덩어리를 만들었는데 노은역 계단의 디딤판(평평)까지 빠져 계단이 줄무늬로 끊겼다 →
   v2: 넓은 평평한 조각(≥ 4 m²)만 뺀다. 노은역 계단 길이 2,053 m 그대로, VFuaQ 평지 오탐지 51 → 1 m.
7. 다듬는 중 검사가 10 cm 간격이라 모서리를 1 cm 안팎 넘는 경로가 44 개(2 %) → 구간별 5 cm 재검사 + 물러섬. 이후 navmesh 밖 0.
8. 넓이 0 삼각형만 모인 덩어리에서 가중 평균이 0 나눗셈(3 씬) → 건너뜀.
9. 평지 경로 "계단 정면 각도" 묶음 중앙값이 36° → 50° 로 나빠 보였지만, 경로별로 짝지어 보면 차이 중앙값 0(나빠짐 15 · 좋아짐 9).
   raw/ref 계단 라벨이 달라 값이 있는 경로 집합이 달랐던 것 — 같은 경로끼리 비교해야 한다.

### 03s — 계단 라벨 수정 전/후 (`03s_stairs_label_check.py`)

03 raw 경로를 두 라벨(수정 전 `unleveled` · 수정 후 `sloped`)로 다시 재서 비교한다. 경로 계획은 새로 하지 않는다.

```bash
/usr/bin/python scripts/dataset_converters/nav3d/checks/03s_stairs_label_check.py --scene noeun_station
ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/checks/03s_stairs_label_check.py --scene {}
/usr/bin/python scripts/dataset_converters/nav3d/checks/03s_stairs_label_check.py --aggregate
/usr/bin/python scripts/dataset_converters/nav3d/checks/03s_stairs_label_check.py --ply VFuaQ6m2Qom noeun_station
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--scene S` | 씬 하나 재기 (03 결과 필요) | |
| `--aggregate` | 표·그림·ply (면적이 가장 많이 준 씬 `--n_scene_figs 6` + 노은역) | |
| `--ply S …` | 그 씬들의 3D ply 만 다시 | |

| 검사 | 수정 전 → 후 | 읽는 법 |
|---|---|---|
| 계단 면적 (62 씬 합) | 1,891 → 1,485 m² | 평평한 바닥이 빠짐 |
| 평지 경로 위 계단 길이 (1,146 경로) | 430 → 288 m | 오탐지 감소 |
| 층 이동 경로 높이 중 계단 구간 몫 | 91.5 → 91.2 % | **진짜 계단은 유지** |
| 노은역 계단 길이 · 높이 몫 | 2,053 m · 89.3 % → 같음 | 유지 |

그림 (`logs/03s_stairs_label/`)
- `scenes_before_after.png`: 색 = navmesh 높이, **빨간 빗금 = 계단 같은 곳**. 층 사이(색이 바뀌는 곳)의 빗금 = 진짜 계단,
  한 색 바닥 위 빗금 = 오탐지.
- `ply/<S>_cloud_{before,after}.ply`: **3D 점구름** — 원본 mesh 표면 점 + navmesh 점. `intensity` 씬 0 · 바닥 0.3 · 계단 1.0,
  RGB 도 회색 · 파랑 · 빨강. `ply/<S>_navmesh_{before,after}.ply`: navmesh 삼각형(같은 intensity·색). Z-up, m.
  CloudCompare 에서 scalar field = intensity 로 색칠 — 기본 색 막대(파랑→초록→노랑→빨강)라 **씬 = 파랑, navmesh 바닥 = 연두, 계단 = 빨강**
  으로 보인다(RGB 로 보면 회색 · 파랑 · 빨강). `ply_check.png` = 비스듬히 본 확인 그림.
- `examples_before_after.png`, `distribution.png`.

### 03r — habitat 1인칭 렌더 (`03r_render_paths.py`, `nav3d_render.py`)

경로를 habitat 안에서 vln_ce 125cm 리그(높이 1.25 m, hfov 79°, 0° · 30° 룩다운)로 찍어 눈으로 비교한다(데이터셋 렌더 아님).

```bash
/usr/bin/python scripts/dataset_converters/nav3d/checks/03r_render_paths.py --check_gt --scene 1LXtFkjw3qL --ep 0
/usr/bin/python scripts/dataset_converters/nav3d/checks/03r_render_paths.py --what stairs_label
/usr/bin/python scripts/dataset_converters/nav3d/checks/03r_render_paths.py --what compare_2d
```

| 인자 | 뜻 | 왜 |
|---|---|---|
| `--check_gt` | GT 자세에서 렌더 = 데이터셋 rgb.125cm_{0,30}deg 인지 | 렌더러 규약 검증. 실측 평균 차 2.9 / 255 (JPEG 수준), 5 넘으면 실패 |
| `--what stairs_label` | 계단 라벨 수정 전/후로 다듬은 경로를, 두 경로가 가장 많이 갈라진 곳에서 | 03s 결과에서 에피소드 선택 |
| `--what compare_2d` | GT · 2D · nav3d 를 같은 진행 비율(10~90 %) + 2D 가 navmesh 밖으로 가장 많이 나간 점에서 | 03c 결과에서 선택 |
| `--n_views 4` | 경로당 찍는 곳 | |

출력 `logs/03r_render/{stairs_label,compare_2d}/<scene>_<ep>.png` (왼쪽 평면도 + 번호 = 카메라 위치).
계단 판정 표시: 이미지 테두리 빨강 = 그 점 아래 navmesh 가 **STAIRS**, 파랑 = **floor**. 평면도 살구색 칠 = 수정 후 계단, 빨간 빗금 = 수정 전 계단.

#### 계단 라벨 전/후 렌더 확인 (`stairs_label/`, 2026-10-14) — **발밑 사진(90°)으로 판정**

판정은 카메라 **발밑** 기준이다. 정면·30° 사진에는 발밑이 안 찍혀(화면 아래 끝 ≈ 0.7 m 앞) "계단이 보이는데 floor" 처럼
보일 수 있다 — 계단 **앞**에 서서 계단을 보는 경우다. 그래서 발밑 사진(가운데 노란 십자 = 판정 지점)으로 맞았는지 본다.
칸 = 수정 전 / 후 판정 → 발밑 사진에 보이는 것.

| 파일 | [1] | [2] | [3] | [4] |
|---|---|---|---|---|
| `VFuaQ6m2Qom_3_wp` | S / f → 마당 바닥 ✅ | S / f → 마당 ✅ | S / f → 마당 ✅ | S / f → 마당 ✅ |
| `b8cTxDM8gDG_6_wp` | S / f → 바닥 ✅ | S / f → 바닥 ✅ | S / f → 복도 ✅ | S / S → 평평한 마루 ❌ 오탐 |
| `dhjEzFoUFzH_3_wp` | S / S → 평평한 타일로 보임 ⚠️ | S / f → 타일 ✅ | S / f → 타일 ✅ | S / f → 타일 ✅ |
| `uNb9QFRL6hY_1_wp` | S / S → 발밑에 디딤판 모서리 ✅ | S / f → 바닥(계단은 앞) ✅ | S / f → 모서리 근처 ⚠️ | S / f → 바닥 ✅ |
| `VLzqgDo317F_10_wp` | f / f → 계단 아래 대리석 바닥 ✅ | S / S → 디딤판 ✅ | S / S → 디딤판 ✅ | S / f → 위층 참(평평) ✅ |
| `gTV8FGcVJC9_88_wp` | S / S → 계단 옆 바닥 ⚠️ | S / S → 곡선 대리석 계단 ✅ | f / f → **카펫 계단 위** ❌ 놓침 | S / S → 카펫 디딤판 ✅ |

(정면 사진만 보고 gTV8 [2][4] 를 오탐이라 적었던 것은 틀렸다 — 발밑은 계단이다.)

- 수정으로 바뀐 판정 14 곳(S → f) 중 13 곳은 발밑이 바닥 ✅, 1 곳(uNb9 [3])은 디딤판 모서리 근처라 애매 ⚠️.
- 진짜 계단 유지: uNb9 [1] · VLzqg [2][3] · gTV8 [2][4] ✅.
- **남은 오류 (수정 전·후 같음)** — navmesh 삼각형 기준 판정의 한계:
  - 놓침 gTV8 [3]: 계단 위 navmesh 가 경사 11° 로 뭉개져(< 15° = 평평으로 봄) 층에 붙음 → floor. dhjEz 성가대석 오르막(경사 8~11°, 높이 0.6 m)도 같은 이유로 floor.
  - 오탐 b8cTx [4]: 층(≥ 10 m²) 에 안 든 작은 평평한 방(z 0.05, 조각마다 < 4 m²)이 옆 경사 조각과 묶여 계단.
  - → 원본 mesh 높이 톱니(stepness)로 판정하는 방안이 있었으나, 03ab 에서 라벨을 통째로 꺼도 결과가 거의 같아 **하지 않기로** 했다.

렌더 원본 낱장: `stairs_label/<scene>_<ep>/<번호>_<before|after>_pitch<0|30|90>.png`, `compare_2d/<scene>_ep<ep>/<번호>_<GT|2D|nav3d>_pitch<0|30>.png`.

### 03ab — 계단을 따로 다듬어야 하나? (`03_refine_paths.py --stairs_mode`, `03ab_stairs_compare.py`, 2026-10-14)

3D 지도(navmesh)에서는 경로 **계획**에 계단 구별이 필요 없다(find_path 가 계단을 지나 층을 오감). 계단 라벨은 **다듬기 방식**만 바꾼다:
split(기본) = 계단은 폭 가운데로 곧게, 바닥은 곡선 / none = 계단 구별 없이 전부 바닥처럼 곡선.
둘을 **같은 코드 · 같은 경로 · 동시에** 돌려 비교했다. 평가 자(계단 위 점 = 기본 라벨)도 같다.

```bash
for M in split none; do ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/03_refine_paths.py --dataset vln_ce --scene {} --stairs_mode $M --out_dir scripts/dataset_converters/nav3d/logs/03_ab_stairs/$M; done
/usr/bin/python scripts/dataset_converters/nav3d/checks/03ab_stairs_compare.py
/usr/bin/python scripts/dataset_converters/nav3d/checks/03r_render_paths.py --what stairs_ab --n_examples 4
```

| `--stairs_mode` | 뜻 | 왜 |
|---|---|---|
| `split` (기본) | 계단 위는 폭 가운데로 곧게, 바닥은 탄성 밴드 + Chaikin | 계단을 비스듬히 오르거나 계단 위에서 돌지 않게 |
| `none` | 계단 구별 없이 전부 바닥처럼 | 비교용 |

계단 정면 각도 = 계단 위 점에서 진행 방향과 계단 경사 방향 사이 각(0° = 정면으로 오름, 90° = 옆으로 가로지름).

| 묶음 (경로 수) | 사용 가능 split / none | GT 같은 루트 | 경계 여유 p5 (m) | 회전 °/m | 계단 정면 각 (경로별 차이 중앙 · none 이 10° 넘게 나쁨 / 좋음) |
|---|---|---|---|---|---|
| 노은역 (100) | 100 / 100 | – | 0.082 / 0.077 | 9.8 / 7.1 | −0.7° · 0 / 0 |
| sg 평지 (573) | 568 / 568 | 425 / 422 | 0.060 / 0.059 | 12.8 / 11.9 | 0° · 7 / 2 |
| sg 층 이동 (530) | **529 / 524** | **459 / 450** | **0.076 / 0.053** | 18.4 / 14.9 | 0° · 51 / 30 |
| wp 평지 (573) | 572 / 572 | 482 / 482 | 0.082 / 0.081 | 13.2 / 12.9 | 0° · 7 / 9 |
| wp 층 이동 (530) | **530 / 528** | 528 / 528 | **0.092 / 0.075** | 18.6 / 16.6 | +0.3° · 45 / 44 |

읽는 법
- **차이는 작다.** 계단 정면 각도는 경로별로 짝지어 보면 차이 중앙값 0°. 렌더(`03r_render/stairs_ab/`, 차이가 가장 큰 4 경로)로 봐도
  두 경로는 계단 위에서 0.1 m 안팎만 다르고 둘 다 계단을 정면으로 오른다 — 각도 차이는 계단 끝 짧은 구간의 숫자 흔들림.
- split 이 나은 곳은 **층 이동 경로의 벽 여유**(p5 0.076 vs 0.053 m)와 그 결과 사용 가능 +5 · +2, 같은 루트 +9 —
  계단 폭 가운데로 가서 난간에서 떨어진다. none 은 회전이 조금 적다(더 매끄럽다).
- **결론: split 을 기본으로 유지하되, 계단 라벨을 더 정교하게 만들 필요는 없다.** 라벨을 통째로 꺼도(none) 결과가 거의 같으므로
  남은 라벨 오류(놓침 · 오탐 1~2 곳, 위 03r)가 결과에 주는 영향은 더 작다. 원본 mesh stepness 로 판정하는 작업은 하지 않는다.
- 실제 로봇(vln_pe H1 · 노은역)에서 계단 위 자세가 중요해지면 그때 다시 본다(지금 자는 navmesh 기하 — 물리 실행 아님).

### 다음

- 04: 사용 가능 판정(위 표 기준) + paths json 내보내기 + 2D 방식과 비교 → 05 렌더 확인.
- sg 모드에서 경로가 약 2 m 길어지는 사례(kEZ7 26/121/130) — 04 에서 길이비로 걸러낼지 결정.

---

## 2D occupancy(gs_vlnpe) vs nav3d 비교 — `03c_compare_2d.py` (2026-10-14)

**같은 에피소드 · 같은 자**로 잰다(CLAUDE.md 9-1). 에피소드 = 03 표본(평지 573 · 층 이동 530), nav3d = sg(시작→끝).
2D = gs_vlnpe 확정 설정(평지 `multifloor/on/paths`, 층 이동 plan `on_plan5/paths_mf` · gt `on/paths_mf`). gs_vlnpe 는 읽기만.

### 같은 자

| 자 | 무엇 | 주의 |
|---|---|---|
| mesh 3D 거리 | 경로(바닥 + 0.5 m, + 키 − 0.05 m)에서 원본 mesh 까지 | 두 지도와 무관(중립). **낮은 가구(소파 등)는 놓친다** |
| navmesh 위 | GT 가 나온 지도. 밖 = navmesh 가 걸을 수 없다고 본 곳(가구·턱·구멍, 벽 0.10 m 안) | nav3d 계획 지도라 nav3d 에 유리 |

### 실행

```bash
ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/checks/03c_compare_2d.py --scene {}
/usr/bin/python scripts/dataset_converters/nav3d/checks/03c_compare_2d.py --aggregate
```

### 결과 — 사용 가능

열마다 따로 센 수(한 경로가 두 열에 들 수 있다). **둘 다 통과** = 생성됨 & mesh 3D 거리 ≥ r_b(0.10) & navmesh 밖 점 없음.

| | 에피소드 | 생성 실패 | mesh 관통 (< 0.02) | mesh r_b 미달 | navmesh 밖 (≤ 0.10 m / > 0.10 m) | **둘 다 통과** | 자기 지도 판정 사용 가능 |
|---|---|---|---|---|---|---|---|
| 평지 nav3d | 573 | 0 | 0 | 4 | 0 / 0 | **569 (99.3 %)** | 569 |
| 평지 2D | 573 | 53 | 4 | 3 | 18 / 27 | **475 (82.9 %)** | 520 |
| 층 이동 nav3d | 530 | 0 | 0 | 1 | 0 / 0 | **529 (99.8 %)** | 529 |
| 층 이동 2D plan | 530 | 151 | 0 | 2 | 50 / 42 | **287 (54.2 %)** | 282 |
| 층 이동 2D gt (참고) | 530 | 35 | 1 | 4 | 59 / 9 | **426 (80.4 %)** | 495 |

### 결과 — 품질

| 품질 (셋 다 생성된 에피소드, 중앙값) | 평지 nav3d | 평지 2D | 층 이동 nav3d | 층 이동 2D plan | 층 이동 2D gt |
|---|---|---|---|---|---|
| 에피소드 | 520 | 520 | 361 | 361 | 361 |
| GT 같은 루트 | 390 | 386 | 321 | 311 | 339 |
| 회전 °/m (GT ≈ 25) | 12.1 | 20.7 | 18.7 | 31.9 | 26.2 |
| mesh 최소 거리 (m) | 0.17 | 0.23 | 0.19 | 0.21 | 0.21 |
| navmesh 경계 여유 p5 (m) | 0.06 | 0.12 | 0.08 | 0.09 | 0.10 |

### 읽는 법
- 2D 가 "자기 지도로 사용 가능" 이라 한 것 중 navmesh 밖으로 나가는 경로: 평지 45, 층 이동 plan 61, gt 69.
  렌더로 보면(03r `759xd9YjKW5_ep8`) **소파 사이를 지나간다** — mesh 3D 거리(0.5 m 높이)는 소파 좌판을 못 봐 0.15 m 로 통과.
  2D 지도의 밴드(바닥 + h_nav ~ 키)와 바닥 높이 차이로 낮은 가구가 새는 것으로 보인다 → 2D 쪽 확인 필요(다른 세션).
- 0.10 m 이하로 살짝 나간 것(평지 18)은 navmesh 가 벽에서 0.10 m 깎여 있어 생기는 경계 차이일 수 있다.
- 2D 는 벽에서 더 멀다(mesh 0.23 vs 0.17) — plan_r_b 0.15 로 계획. nav3d 는 회전이 적다(12 vs 21 °/m).
- 같은 루트는 비슷(390 vs 386). 층 이동은 2D plan 이 151 개 생성 실패(계단 길 못 찾음 등), nav3d 는 0.

### 렌더로 본 예 (`logs/03r_render/compare_2d/`, 왼쪽 평면도 번호 = 카메라 위치)

| 파일 | 2D | nav3d | 보이는 것 |
|---|---|---|---|
| `759xd9YjKW5_ep8.png` | navmesh 밖 | 통과 | 2D 가 소파 사이로 지나감 (mesh 3D 거리는 0.15 m 로 통과 — 낮은 가구를 못 봄) |
| `82sE5b5pLXE_ep9.png` | navmesh 밖 0.08 m | 통과 | 야외 가장자리 — 살짝 벗어남(경계 차이 수준) |
| `7y3sRwLe3Va_ep1.png` | 통과 | r_b 미달 | nav3d 가 벽에 붙는 사례 |
| `1LXtFkjw3qL_ep14` · `1pXnuDYAj8r_ep0` · `29hnd4uzFmX_ep8` | 통과 | 통과 | 층 이동 — 둘 다 계단을 오름 |

---

## 04 — 노은역 경로 고르기 · 내보내기 (`04_export_paths.py`, 2026-10-16)

`/usr/bin/python scripts/dataset_converters/nav3d/04_export_paths.py` (habitat 컨테이너)

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--n_multifloor` | 층 이동 경로 수 (03 결과에서) | 03 이 만든 100 개 전부 (사용자 결정) |
| `--flat` | `층:개수` 평지 경로 | `0:50,1:50` — 승강장 50 · 대합실 50 (사용자 결정) |
| `--flat_min_m` / `--flat_max_m` | 평지 길이 하한 / 시작·끝 직선 거리 상한 | 3 m (너무 짧은 것 제외) / 40 m (R2R 수준) |
| `--holes_csv` | 00 의 큰 mesh 구멍 목록 (U4) | 구멍 위를 지나면 렌더·충돌이 틀린다 |

판정 U1 경로 있음 · U2 navmesh 위 100 % · U3 mesh 여유 < r_b 점 0 · U4 큰 구멍(+ r_b) 안 점 0 · U5 평지(높이 폭 < 0.5 m, 길이 ≥ 3 m).
결과: 층 이동 100 (중앙 61.0 m) · S0 50 · S1 50 (중앙 16.0 m), 모두 통과. 제외 106 은 전부 평지 후보의 U5. mesh 여유 최소 0.287 m.
출력 `logs/04_export_paths/noeun_station/noeun_station_paths.json` — episode 마다 `trajectory`(점마다 z) · `segments`(floor / stairs) · `kind`.

## 05 — vln_ce 렌더 (habitat, 2026-10-15)

vln_ce 는 릴리스가 habitat 산출이라 habitat 으로 렌더한다 (사용자 지시). 렌더러 = `instruction_labeling/render` (RigSim: 640×480 · hfov 79° · 1.25 m · pitch 0/30°).

- **05a GT 재렌더** — `/usr/bin/python scripts/dataset_converters/nav3d/checks/05a_vlnce_gt_render_check.py` : 릴리스 action 을 그대로 재생.
  4 씬 685 ep: 자세 최대 0.08 mm, RGB SSIM 최소 0.9745 → 렌더러 = 릴리스. depth 는 릴리스가 no-hit 구멍을 사후 채워 참고만 (규칙 재현 못 함).
- **05b 생성 경로** — `/usr/bin/python scripts/dataset_converters/nav3d/05b_vlnce_render_paths.py` : 03 wp·sg 경로 → 0.5 m 경유점 → GreedyGeodesicFollower → 릴리스 포맷 `vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}/`.
  P1 완주 56/56 · P2 충실도 · P3 충돌 · P4 계단 |Δz| ≤ 0.067 m · P5 형식 모두 통과. `--depth_fill none` 기본 (채우기 전후 차이 작음, 사용자 결정).
- **05c 뷰어** — `--page_only` 는 렌더 없이 페이지만 다시.

## 06 — 노은역 렌더 (Isaac, `06_render_noeun_vlnce.py`, 2026-10-16)

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06_render_noeun_vlnce.py` (isaac 컨테이너)

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--rig` | vln_ce 스키마의 카메라 칸 | `60cm_15deg` — D455 값을 넣을 칸 (사용자 결정). 640×480, hfov 68.7° (이 칸 규약) |
| `--mount_height_m` / `--pitch_down_deg` | 카메라 높이 / 아래로 기운 각 | 0.584 m / 14.76° — RBQ10 D455 장착값 |
| `--forward_m` | 카메라가 몸 중심보다 앞선 거리 | 0.365 m — camera_link 를 base_link 로 접은 값 |
| `--usd_path` | 장면 | 역 전체 `noeun_station_collision.usdz` (GS + 충돌 mesh) |
| `--out_scene` | 출력 scene 이름 | `noeunstation` — loader 가 `_` 로 자르므로 `_` 금지 |

몸 위치 = 04 경로를 전진 0.25 m / 회전 15° 로 이산화, 몸 바닥 z = 04 경로 z 를 몸 위치에 투영(계단을 따라 오르내림).
검사 `06b_check_noeun_render.py` (habitat 컨테이너): R1 형식 · R2 카메라 자세(전진은 **수평** 0.25 m) · R3 경로 · R4 평지 카메라 높이 · R5 depth · R6 RGB 밝기.
시험 2 ep: 904 프레임 / 212 s, R4 외 통과(평지 없음). 함정: 계단에서 전진을 3D 로 재면 58.9 mm 오차처럼 보임 · 긴 strip JPEG 폭 65,535 px 한계(200 프레임씩 나눔).

### 04b · 06v — 바닥 z 를 navmesh 가 아니라 mesh 로 (2026-10-16)

06v: 06 의 `episode_poses` 에 vln_pe 카메라 값(GT 에서 잼: navmesh 위 1.536 / mesh 위 1.656 m · 앞 0.239 m · pitch 18.1°)을 넣어
MP3D 7 씬 46 ep 를 만들고 vln_pe GT 와 짝 프레임 비교 + 같은 세션 Isaac 렌더.

```
/usr/bin/python scripts/dataset_converters/nav3d/06v_vlnpe_check.py --stage prepare
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06v_render_isaac.py
/usr/bin/python scripts/dataset_converters/nav3d/06v_vlnpe_check.py --stage report
/usr/bin/python scripts/dataset_converters/nav3d/04b_paths_mesh_floor.py
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--scenes` | 비교할 씬 | 05 와 같은 4 씬 |
| `--stairs_scenes` | 계단 ep 만 더 가져올 씬 | vln_pe 2,813 ep 중 로봇 z 범위 > 0.5 m 는 이 3 씬의 3 ep 뿐 |
| `--max_eps` | 씬당 평지 ep 상한 (wp 경로 있는 ep 는 항상) | 8 — 렌더 10 분 안 |
| `--pair_r` | 짝 프레임 반경 | 0.3 m — 한 걸음(0.25) + 흔들림 |

결과: **GT 카메라의 mesh 위 높이는 모든 ep 에서 1.656 ± 0.004 m, navmesh 위는 1.46~1.61 m** — 로봇은 mesh 를 딛는다.
카메라 높이 |Δz| p95: navmesh z 0.079/0.102 m → mesh z 0.021/0.022 m. 노은역 navmesh 는 mesh 보다 중앙 3.1 cm 위(04b) = 06b R4 의 원인.
V1(위치)·V5(SSIM) FAIL 은 자 설계 문제 (프레임 간격이 달라 짝이 0.16 m·4° 어긋남) — 그림으로는 같은 시점.
