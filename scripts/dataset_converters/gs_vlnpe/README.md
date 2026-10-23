# gs_vlnpe — 씬에서 경로를 만들고 렌더해 학습 데이터셋으로 내보낸다

리포트 링크 모음과 실험 경위는 [`reports.md`](reports.md). 이 문서는 **뭘 어떤 순서로 돌리고,
무엇이 들어가고 나오며, 어떻게 검증하는지**만 다룬다.

실행은 전부 저장소 루트에서, Isaac Sim 포함 Python으로 한다.

```
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
GS=scripts/dataset_converters/gs_vlnpe
```

---

## 1. 세 대상

같은 파이프라인이 세 가지 씬·출력 포맷을 다룬다. **다른 건 입력 자산과 GT 유무뿐**이다.

| | **vln_ce** | **vln_pe** | **노은역** |
|---|---|---|---|
| 씬 자산 | Matterport3D `.obj` | Matterport3D `.obj` | 3DGS `.usdz` (NuRec) |
| 데이터 규약 | Habitat | Isaac Sim (H1 휴머노이드) | — (GT 없음) |
| 기존 GT 궤적 | 있음 | 있음 | **없음** |
| 이미지 | 640×480 | 256×256 | 480×270 |
| action | 이산 0.25 m / 15° | 이산(물리 롤아웃) | 이산화해서 생성 |
| 쓰는 모델 | InternVLA-N1 | CMA / Seq2Seq / RDP | (vln_ce로 내보냄) |
| 스크립트 위치 | `$GS/` | `$GS/` | `$GS/apply_real/` |

`vln_n1`은 네 번째 대상이 아니라 **`$GS`가 읽을 수 있는 또 하나의 원본 데이터셋**이다.
`--dataset`의 기본값은 **`vln_pe`**다(2026-10-11 변경). `apply_real/`만 `vln_n1`을 유지한다 —
노은역은 둘 중 어느 쪽도 아니고, `--dataset`이 정하는 렌더 해상도(480×270)와 출력 접미사('')를
그대로 써야 기존 산출물과 맞기 때문이다.

**노은역만 다른 점** — GT가 없어서:
- `02`가 geometry만으로 지도를 만든다 (`--geometry usd` 필수, `--ref_h_b`로 로봇 키를 직접 준다)
- `03`은 `--mode random`만 가능하다 (`reproduce`는 재현할 GT가 없다)
- `04`는 `--camera` 프로파일을 반드시 지정한다 (기본 `dataset`은 exit 2)
- `--mode random`이라 `--scan_coverage`가 필요하다 (아래 §3 참고)

---

## 2. 파이프라인 — 경로 생성부터 렌더링까지

```
01 → 02 → 03 → 03u → 04 → 05 → 05b
```

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 01 | `01_prepare_scene.py` | 씬 mesh/USD + GT | `<out>/scene_meta/<scene>.json` (floor_z 등) |
| 02 | `02_build_freemap_esdf.py` | 씬 mesh/USD | `<out>/esdf/<scene>.npz` (occupancy + ESDF) |
| 03 | `03_sample_gt_paths.py` | ESDF (+ GT의 start/goal) | `<out>/paths/<scene>.json` (경로 + GT 대조 지표) |
| 03mf (선택) | `03mf_plan_multifloor.py` | ESDF + GT | `<out>/paths_mf/<scene>.json` — **층 이동 에피소드만** 층별 A\* + 계단 GT (아래 "층 이동") |
| 03u | `03u_filter_usable.py` | `<out>/paths/` (+ `run_logs/`, GT) | `<out>/usable/paths/<scene>.json` — **사용 가능 경로만** (04 입력) |
| 04 | `04_render_obs_isaac.py` | 경로(`--out_dir <out>/usable`) + 씬 | `<out>/obs/<scene>_<mode>_isaac_<cam>/` (rgb/depth) |
| 04' | `04_render_obs.py` (Open3D) | 경로 + 씬 | `<out>/obs/<scene>/episode_%06d/` — **검증용 대체 렌더러** |
| 05 | `05_export_vlnce.py` / `05_export_vlnpe.py` | 경로 + 씬 | `data/.../vln_{ce,pe}_render/` (데이터셋) |
| 05b | `05b_verify_vlnce.py` / `05b_verify_vlnpe.py` | 산출물 + GT | 게이트 판정 + `verify.json` |

`<out>` = `--out_dir` (기본 `$GS/logs`). 파일명에 `vln_pe`는 `_vlnpe`, `--mode random`은 `_random` 접미사가 붙는다.
리포트는 `--log_dir/<script_name>/<scene>[_random|_reproduce]/report.html`.

### 커맨드

```
$ISAAC_PY $GS/02_build_freemap_esdf.py --dataset vln_pe --scene 17DRP5sb8fy --r_b 0.20 --h_nav_ratio 0.12 --out_dir $GS/logs --scene_meta_dir $GS/logs
```
```
$ISAAC_PY $GS/03_sample_gt_paths.py --dataset vln_pe --scene 17DRP5sb8fy --path_profile reproduce_v1 --out_dir $GS/logs
```
```
timeout --signal=KILL 900 $ISAAC_PY $GS/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode reproduce --preset measured --out_dir $GS/logs
```

**노은역** (`$AR=$GS/apply_real`, 전용 인자만 다름):

```
timeout --signal=KILL 3600 $ISAAC_PY $AR/02_build_freemap_esdf.py --scene noeun_station_mid --geometry usd --ref_h_b 0.875 --scene_meta_dir $AR --out_dir $AR --log_dir logs/gs-vlnpe/apply_real
```
```
$ISAAC_PY $GS/03_sample_gt_paths.py --scene noeun_station_mid --mode random --num_episodes 20 --esdf_dir $AR/esdf --out_dir $AR --log_dir logs/gs-vlnpe/apply_real
```
```
timeout --signal=KILL 14400 $ISAAC_PY $AR/04_render_obs_isaac.py --scene noeun_station_mid --mode random --num_episodes 20 --camera d455_nominal --out_dir $AR --log_dir logs/gs-vlnpe/apply_real
```

### 렌더러는 두 개다 — 생산은 Isaac

| | `04_render_obs.py` | `04_render_obs_isaac.py` |
|---|---|---|
| 엔진 | Open3D (mesh 래스터) | **Isaac Sim (RTX)** |
| 조명·톤맵 | 없음 | 있음 (`--preset measured`) |
| 쓰임 | depth 정확도 검증 | **생산** |

`05_export_vlnce` · `05_export_vlnpe` · `04e_sample_compare` 셋 다 `04_render_obs_isaac.py`를
모듈로 로드한다. **`vln_ce`도 Isaac으로 렌더한다**(`05_export_vlnce.py`가
`apply_real/04_render_obs_isaac.py`를 쓴다).

> **habitat은 쓰지 않는다.** `gs_vlnpe` 전체가 habitat을 import하지 않는다 — 코드에 나오는
> `habitat`은 전부 좌표계 설명 주석이다. habitat은 평가(eval)와 `3dloader_vlnce/`의 navmesh
> 검증에서만 쓴다.

### 61씬 전수로 경로만 (CPU, Isaac 안 띄움)

`01` → `02` → `03`을 필요한 것만 돌려 **같은 루트 비율**을 잰다.

```
timeout --signal=KILL 21600 $ISAAC_PY $GS/03f_reproduce_batch.py --dataset vln_pe --n_scenes 0 --path_profile reproduce_v1 --work_dir $GS/logs --log_dir logs/gs-vlnpe/03f_v1
```

실측 45씬 2,274 ep: 같은 루트 **78.8 %**, 하드 충돌 **0**, 쓸 수 있는 ep **1,700**.

> ⚠️ 이 경로들을 **렌더해서 저장하는 `07_render_paths_to_vlnpe.py`는 아직 없다**(Phase C).
> action 이산화(`docs/gs-vlnpe-l1-l2a.md` §5.2)와 조립이 필요하다. 라이터 `vlnpe_writer.py`는 있다.

### occupancy 기반 경로 생성 — 요약 (02 → 03 → 03u)

씬을 **한 층짜리 2D 지도**로 펼치고, 그 위에서 GT 와 같은 start/goal 사이 경로를 새로 만든 뒤,
**벽에서 충분히 떨어진 경로만** 남긴다.

| 단계 | 하는 일 | 핵심 |
|---|---|---|
| **02** 지도 | mesh 를 0.05 m 격자로 나눠 바닥 위 밴드 `(floor+h_nav, floor+h_b]` 에 물체가 있으면 벽 → 벽까지 거리(ESDF) | 텍스처는 읽지 않는다(`skip_materials`, RAM 100 GB+ → 0.16 GB) |
| **03** 경로 | ESDF < `plan_r_b` 를 막고 → 0.2 m 격자로 줄여(`majority`) A\* → ② waypoint 를 벽에서 먼 쪽으로 다시 밀기(`--post_refine`) → ③ 얇은 벽 보정(`--thin_safe`) → ④ bezier 스무딩 | 계획은 `plan_r_b`(여유 포함), 판정은 `r_b`. waypoint 간격 0.8 m 유지(줄이면 회전 증가) |
| **03u** 필터 | 궤적을 1 cm 간격으로 찍은 벽까지 최소 거리 `points_min_m ≥ r_b` 인 것만 남김. 층 이동 에피소드 제외 | 원본 `paths/` 는 분석용으로 유지, 04 는 `usable/paths/` 만 읽는다 |

결과 (2026-10-13, 61씬 전수):

| | 설정 | 사용 가능 | 관통 |
|---|---|---|---|
| vln_ce | `r_b 0.10` · `plan_r_b 0.15` · `floor_tol_m 0.1` + ②③④ | **8,450 / 9,092 (92.9 %)** | 0 |
| vln_pe | `r_b 0.20` · `plan_r_b 0.25` · `floor_tol_m 0.1` + ②③④ | **2,560 / 2,810 (91.1 %)** | 0 |

**층 이동 — 기본은 뺀다, `03mf` + `--multifloor_mode stairs` 로 켜면 살린다.** 지도가 한 층 단면이라 계단으로 층을
오가는 에피소드(vln_ce 1,592 개 = 14.9 %, vln_pe 3 개)는 기본(off)으로 03u 에서 뺀다. 켜면
`출발 층 A* → 계단 → 도착 층 A*` 로 이어 붙여 살린다. 계단은 두 모드: **gt**(GT 길 그대로 — vln_ce 1,486 · vln_pe 3)와
**plan**(GT 에서 계단 위치만 쓰고 길은 전부 새로 — vln_ce 845 · vln_pe 1). 아래 "층 이동" 절.
노은역 B1/B2 같은 3D(navmesh) 방식은 별도 작업(nav3d).

### 경로 생성 setup — 2026-10-13 임시 확정 (occupancy 기준)

**판정 기준.** GT 에피소드 하나는 아래 중 정확히 한 칸에 들어간다(겹치지 않음).

| 칸 | 뜻 |
|---|---|
| 퇴화 GT | GT 가 5 프레임 미만 — 03 이 건너뜀 |
| 끝점 막힘 | GT start/goal 이 우리 지도에서 통행 불가 (`start_or_goal_not_navigable`) |
| A* 실패 | 끝점은 되지만 연결 경로 없음 |
| 관통 | 생성됐지만 궤적 최소 여유 ≤ 0 (벽 안) |
| r_b 미달 | 생성됐지만 0 < 최소 여유 < `r_b` |
| **사용 가능** | 생성 & 최소 여유 ≥ `r_b` — **이것만 쓴다** |

`--multifloor_mode stairs` 이면 층 이동 에피소드도 아래 칸으로 나눈다(합 = 층 이동 수, 평지 칸은 그대로):

| 층 이동 칸 | 뜻 |
|---|---|
| 퇴화 GT | 5 프레임 미만 |
| 계단 입구 막힘 | 계단 입구/출구를 GT 따라 2 m 물려도 그 층 지도에서 통행 가능한 점 없음 |
| 층 끝점 막힘 | 층 지도에서 A\* 시작/끝(GT start·goal 또는 물린 입구·출구)이 통행 불가 |
| 층별 A\* 실패 | 층 지도에서 연결 경로 없음 |
| 평지 계획 없음 | (gt 모드) 평지 구간이 전부 짧아 새로 계획한 곳이 없음(GT 복사본) |
| 계단 길 못 찾음 | (plan 모드) 높이 지도에서 계단 아래 → 위 경로 없음 |
| 결과 없음 | paths_mf 에 없음 |
| 관통 / r_b 미달 | 최소 여유 ≤ 0 / < `r_b` — gt 모드는 평지 부분만, plan 모드는 평지 + 계단 |
| **사용 가능** | 최소 여유 ≥ `r_b` (gt 모드: 계단은 GT 가 지난 길이라 통과로 본다) |

**파이프라인 단계 `03u_filter_usable.py` 가 판정하고 사용 가능만 남긴다.** 03 이 경로마다 기록한
`check_trajectory.points_min_m`(0.05 m ESDF 에서 궤적을 1 cm 간격으로 찍은 벽까지 최소 거리)과 paths json 의
`params.r_b` 로 칸을 나누고, 층 이동도 직접 계산해 뺀다. 출력은 `<out>/usable/` —
`paths/`(사용 가능 에피소드만, 나머지는 버림), `usable.json`(칸별 개수 + 목록), `multifloor_episodes.json`.
원본 `<out>/paths/` 는 분석용으로 그대로 둔다. **04 렌더는 `--out_dir <out>/usable` 로 이것만 읽는다.**

"같은루트"(GT 와 같은 길)는 **참고 지표**다 — instruction 을 새로 붙이므로 필터 기준이 아니다.
**층 이동 에피소드**(시작·끝 카메라 z 차 > 0.5 m)는 2D 계획으로 다룰 수 없어 **제외**한다:
vln_ce 1,592 / 10,684 (14.9 %), vln_pe 3 / 2,813. 목록 `logs/vlnce_full/multifloor_episodes.json`,
`logs/vlnpe_full/multifloor_episodes.json`. (InternNav 도 VLN-PE 에서만 `filter_stairs` 로 계단을 뺀다.)

#### vln_ce — **확정(임시)**: `plan_r_b 0.15` + ②③④ + `floor_tol_m 0.1`

61씬 전체 (habitat 컨테이너, CPU 만, 동시 4개 — 02 는 씬당 RAM < 1 GB, 03 은 수백 MB):

```
export GS=scripts/dataset_converters/gs_vlnpe PY=/usr/bin/python OUT=scripts/dataset_converters/gs_vlnpe/logs/vlnce_setup
mkdir -p $OUT/run_logs; ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r > $OUT/scenes.txt

# ① ESDF — 없는 씬만. 게이트 FAIL 은 exit 1 이지만 esdf 는 저장되므로 && 로 03 과 묶지 말 것
cat $OUT/scenes.txt | xargs -P 4 -I{} sh -c '[ -f $GS/logs/esdf/{}_vlnce.npz ] || $PY $GS/02_build_freemap_esdf.py --dataset vln_ce --rig 125cm_30deg --scene {} --scene_meta_dir $GS/logs --out_dir $GS/logs --log_dir logs/gs-vlnpe > $OUT/run_logs/{}_02.log 2>&1'

# ② 경로 — 로그 이름은 <scene>.log 로 둔다(③ 이 성공 0건 씬의 사유를 여기서 읽는다)
cat $OUT/scenes.txt | xargs -P 4 -I{} sh -c '$PY $GS/03_sample_gt_paths.py --dataset vln_ce --rig 125cm_30deg --scene {} --r_b 0.10 --plan_r_b 0.15 --downsample_mode majority --floor_tol_m 0.1 --post_refine --post_refine_margin 0.05 --thin_safe --smooth bezier --num_episodes 999 --esdf_dir $GS/logs/esdf --out_dir $OUT --log_dir $OUT/gs-vlnpe > $OUT/run_logs/{}.log 2>&1'

# ③ 03u: 사용 가능만 남김 -> $OUT/usable/paths/ (층 이동·끝점 막힘·A* 실패·관통·r_b 미달은 버림, 원본 $OUT/paths 는 유지)
$PY $GS/03u_filter_usable.py --dataset vln_ce --rig 125cm_30deg --out_dir $OUT
```

평지 9,092 ep (61씬) — 사용 가능 **8,450 (92.9 %)**, 관통 0, r_b 미달 6, 끝점 막힘 489, A* 실패 147,
같은루트(참고) 6,223 / 8,450 (73.6 %), 최소 여유 중앙 0.20 m, 회전 22.4 °/m.

| vln_ce 설정 | 끝점 막힘 | A* 실패 | 관통 | r_b 미달 | **사용 가능** | 같은루트 (참고) | 여유 < r_b+0.05 |
|---|---|---|---|---|---|---|---|
| 0.10 기준선 | 345 | 54 | 736 | 660 | 7,297 (80.3 %) | 73.2 % | 26 % |
| 0.10 + ②③④ + edge + 바닥0.1 | 276 | 57 | 21 | 141 | 8,597 (94.6 %) | 71.2 % | 25 % |
| 0.15 + ②③④ | 585 | 117 | 0 | 9 | 8,381 (92.2 %) | 74.1 % | 3 % |
| **0.15 + ②③④ + 바닥0.1** | 489 | 147 | 0 | 6 | **8,450 (92.9 %)** | 73.6 % | **3 %** |
| 0.20 + ②③④ | 828 | 378 | 0 | 0 | 7,886 (86.7 %) | 76.1 % | 0 % |

> ④ 렌더: **vln_ce 의 새 경로 렌더는 아직 없다** — `04_render_obs_isaac.py` 는 `--dataset {vln_n1,vln_pe}` 만
> 받고, `05_export_vlnce.py` 는 GT pose 재생 전용이다. 입력은 `$OUT/usable/paths/` 로 정해 둔다.

0.10+edge 가 147 개 더 많지만 경로 1/4 이 벽에서 0.10~0.15 m(지도 오차 1 셀 이내)로 지나간다 → 0.15 선택.
`--astar_edge_check` 는 0.15 에서 결과가 같아 넣지 않는다.

#### vln_pe — **제안**: legacy + `plan_r_b 0.25` + ②③④ + `floor_tol_m 0.1`

```
export GS=scripts/dataset_converters/gs_vlnpe PY=/usr/bin/python OUT=scripts/dataset_converters/gs_vlnpe/logs/vlnpe_setup
mkdir -p $OUT/run_logs; ls data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r > $OUT/scenes.txt

# ① ESDF — 없는 씬만 (scene_meta 가 없으면 01 --dataset vln_pe 먼저)
cat $OUT/scenes.txt | xargs -P 4 -I{} sh -c '[ -f $GS/logs/esdf/{}_vlnpe.npz ] || $PY $GS/02_build_freemap_esdf.py --dataset vln_pe --scene {} --r_b 0.20 --h_nav_ratio 0.12 --scene_meta_dir $GS/logs --out_dir $GS/logs --log_dir logs/gs-vlnpe > $OUT/run_logs/{}_02.log 2>&1'

# ② 경로
cat $OUT/scenes.txt | xargs -P 4 -I{} sh -c '$PY $GS/03_sample_gt_paths.py --dataset vln_pe --scene {} --r_b 0.20 --plan_r_b 0.25 --h_nav_ratio 0.12 --refine_radius 0.20 --floor_tol_m 0.1 --post_refine --post_refine_margin 0.05 --thin_safe --smooth bezier --num_episodes 999 --esdf_dir $GS/logs/esdf --out_dir $OUT --log_dir $OUT/gs-vlnpe > $OUT/run_logs/{}.log 2>&1'

# ③ 03u: 사용 가능만 남김 -> $OUT/usable/paths/ (원본 $OUT/paths 는 유지)
$PY $GS/03u_filter_usable.py --dataset vln_pe --out_dir $OUT
```

| vln_pe 설정 (평지 2,810 · 퇴화 23) | 끝점 막힘 | A* 실패 | 관통 | r_b 미달 | **사용 가능** | 같은루트 (참고) | 여유 < r_b+0.05 | 회전 °/m |
|---|---|---|---|---|---|---|---|---|
| legacy (plan 0.20) | 69 | 5 | 21 | 455 | 2,237 (79.6 %) | 75.8 % | 17 % | 31.5 |
| legacy + ②③④ (edge 유무 동일) | 69 | 5 | 0 | 169 | 2,544 (90.5 %) | 75.0 % | 16 % | 23.9 |
| **legacy + 0.25 + ②③④ + 바닥0.1** | 132 | 28 | 0 | 67 | **2,560 (91.1 %)** | 76.4 % | **6 %** | 24.3 |
| legacy + 0.28 + ②③④ (10-12 선택) | 231 | 115 | 0 | 25 | 2,416 (86.0 %) | 77.3 % | 3 % | 24.7 |
| `--path_profile reproduce_v1` | 171 | 44 | 0 | 110 | 2,462 (87.6 %) | 80.0 % | 9 % | **53.1** |

> ④ 렌더 (rgb/depth 만, Isaac 컨테이너): 사용 가능만 남긴 폴더를 `--out_dir` 로 준다.
> `timeout --signal=KILL 3600 $ISAAC_PY $GS/04_render_obs_isaac.py --dataset vln_pe --scene <S> --mode reproduce --preset measured --num_episodes 999 --frame_step_m 0.141 --out_dir $OUT/usable`
> 결과는 `$OUT/usable/obs/<S>_vlnpe_reproduce_isaac/episode_%06d/` — **vln_pe 데이터셋 구조(parquet·action·instruction)로
> 묶는 단계(Phase C)는 아직 없다**(위 61씬 절의 ⚠️ 참고).

0.25 가 사용 가능 수 최대이면서 벽 근접도 6 %. 0.28 은 생성 단계 손실이 커서 총량이 준다.
`reproduce_v1` 은 waypoint 간격 0.2 m 라 회전이 2 배(53 °/m)로 거칠다.

#### 층 이동 (계단) — 선택: `03mf` + `03u/04 --multifloor_mode stairs` (2026-10-13)

기본은 꺼져 있다(03 은 무변경, 03u·04 는 인자를 안 주면 종전과 바이트 동일 — 실측). 켜는 법: 위 확정 03 명령의
`03_sample_gt_paths.py` 를 `03mf_plan_multifloor.py` 로 바꿔 **같은 인자로** 한 번 더 돌리고, 03u·04 에 `--multifloor_mode stairs`.

```
cat $OUT/scenes.txt | xargs -P 4 -I{} sh -c '$PY $GS/03mf_plan_multifloor.py <위 03 확정 인자 그대로> --scene {} --out_dir $OUT > $OUT/run_logs_mf/{}.log 2>&1'
$PY $GS/03u_filter_usable.py --dataset vln_ce --rig 125cm_30deg --out_dir $OUT --multifloor_mode stairs
```

| 03mf 인자 | 뜻 | 기본 / 이유 |
|---|---|---|
| `--mf_dz_m` | 층 이동 판정 z 차 | 0.5 — 03u `--dz_m` 과 같아야 목록 일치 |
| `--mf_level_tol_m` | 같은 평지로 볼 바닥 z 흔들림 | 0.1 — GT 평지 흔들림 실측 ±0.08 |
| `--mf_plateau_min_m` | 평지로 인정할 최소 길이 | 1.0 — 짧은 참은 계단에 흡수 |
| `--mf_lead_m` / `--mf_snap_max_m` | 계단 입구/출구를 GT 따라 평지 쪽으로 물리는 최소 / 최대 | 0.4 / 2.0 — 계단 칸이 지도에서 벽일 때 |
| `--mf_seam_m` | 평지 경로↔계단 이음매를 깎는 Bézier 반폭 | 0.3 |
| `--mf_min_plan_m` | 이보다 짧은 평지는 A\* 없이 GT(plan 모드: 두 점 직선) | 0.4 (A\* 2칸) |
| `--mf_stairs {gt,plan}` | 계단 만드는 법 | **gt**(기본) = GT 길 그대로(계단 앞·짧은 평지도 GT) · **plan** = GT 에서 **계단 위치만** 쓰고 계단·평지 길 전부 새로 계획 |
| `--mf_stairs_spacing_m` | plan: 계단 경로 펴기(Bézier) 솎기 간격 | 0.4 — 벽에 붙으면 0.2 → 0(안 폄)으로 자동 재시도 |

plan 모드는 이음매 폭도 `--mf_seam_m` → 절반 → 0 순서로 시도해 판정을 통과하는 가장 부드러운 폭을 쓴다(사용 가능 845 중 0.3 m 770 · 0.15 m 48 · 0 m 27).

**plan 모드 계단 계획** (`stairs_surface.py`): GT 계단 점 범위 ±1.5 m 상자 안에서 **높이 지도**를 만든다 — 칸마다 디딜 면
(점유 voxel 윗면 + 그 위 `(h_nav, h_b]` 머리 공간 비어 있음, 층 지도와 같은 밴드), 이웃 칸(1~2칸) 면과 높이 차 ≤ 0.25 m 면 연결,
연결이 끊기는 곳 = 벽·난간·낭떠러지까지의 거리 = 여유. 여유 ≥ `plan_r_b` 인 면만 A\*. 계단 아래·위 점(E·X)은 층 높이·여유·층 지도 통행 가능을 모두 만족하는 면 중에서
GT 계단 끝 2 m 안을 먼저, 없으면 상자 전체에서 고른다. 여유 `plan_r_b` 로 길이 없으면 `r_b` 로 한 번 더.
층 지도에는 계단 주변의 큰 낭떠러지(바닥보다 0.4 m 넘게 낮은 면만 있는 0.1 m² 이상 덩어리)를 벽으로 찍은 occupancy 사본을 넘긴다. 판정은 평지 = 층 ESDF, 계단 = 높이 지도 여유,
계단 끝 0.3 m 안의 평지 = 높이 지도(층 지도는 첫 디딤판을 벽으로 본다).

출력 json 은 03 과 같은 키 + `trajectory_z`(점마다 바닥 z) · `segments`([{kind: floor\|stairs, floor_z, i0, i1}]) · `multifloor`(진단).
`check_trajectory` 는 **계획한 평지 부분만** 그 층 ESDF 로 잰 값. 04 는 `trajectory_z` 가 있으면 카메라 z = 바닥 z + h_b.

| 61씬 (확정 설정) | 층 이동 | 입구 막힘 | 층 끝점 막힘 | 층별 A\* 실패 | 평지 계획 없음 | 관통 / r_b 미달 | **사용 가능** |
|---|---|---|---|---|---|---|---|
| vln_ce · gt | 1,592 | 1 | 27 | 12 | 66 | 0 / 0 | **1,486 (93.3 %)** |
| vln_pe · gt | 3 | 0 | 0 | 0 | 0 | 0 / 0 | **3** |
| vln_ce · plan | 1,592 | 54 | 39 | 24 | 계단 길 없음 338 | 157 / 135 | **845 (53.1 %)** |
| vln_pe · plan | 3 | 0 | 0 | 0 | 계단 길 없음 1 | 0 / 1 | **1** |

**전체 대비 사용 가능** (분모 = GT 전체, 평지 결과 8,450 · 2,560 은 모드와 무관하게 그대로):

| | 평지만 | + 층 이동 gt | + 층 이동 plan |
|---|---|---|---|
| vln_ce (10,684) | 8,450 (79.1 %) | 9,936 (**93.0 %**) | 9,295 (**87.0 %**) |
| vln_pe (2,813) | 2,560 (91.0 %) | 2,563 (**91.1 %**) | 2,561 (**91.0 %**) |

⚠️ gt 모드 사용 가능 중 201 개는 새로 계획한 길이가 30 % 미만(대부분 GT 복사). 새 경로만 원하면 plan 모드.
plan 관통 157 은 대부분 계단 끝 근처에서 계단 옆 가장자리(높이 지도 여유 0)를 지나는 것 — 판정 오류가 아니라 실제 위험.

**Isaac 렌더 확인 (2026-10-13)**: vln_pe `b8cTxDM8gDG` ep8(gt) · vln_ce `5q7pvUzZiYa` ep87 · `1pXnuDYAj8r` ep108(plan, U자 계단) 모두
카메라가 디딤판을 따라 오르내림(z 1.65→0.96 / 4.45→1.21 / 4.76→3.13(참)→1.36). 04 `--multifloor_mode` off·stairs 둘 다 평지 에피소드의
extrinsic·depth 가 패치 전과 56/56 바이트 동일, rgb 는 같은 코드 두 번도 평균 0.8/255 차이(렌더러 비결정성 — 코드 무관).
⚠️ 04 의 메시 거리 검사(trimesh)가 25 분 넘게 멈춘 적이 있다 — 이미지·pose 는 그 전에 저장되므로 프레임 저장이 멈추면 PID 로 종료.
vln_ce 는 04 가 `--dataset vln_ce` 를 안 받아 json 을 `_vlnpe` 이름으로 복사해 `--dataset vln_pe`(256×256) 로 렌더했다(검증용). 근거 `.claude/memory/261013_multifloor_stairs_result.md`,
산출 `logs/vlnce_full/multifloor/on/`, `logs/vlnpe_full/multifloor/on/`, 그림 `logs/vlnce_full/multifloor/viz/`.

①=`--plan_r_b` ②=`--post_refine --post_refine_margin 0.05` ③=`--thin_safe` ④=`--smooth bezier`.
근거·전체 표: `.claude/memory/261013_astar_edge_check_result.md`, `261013_vlnce_planrb_sweep_result.md`.
산출물: `logs/vlnce_full/`, `logs/vlnpe_full/` (실행 스크립트 `run_*.sh` 같은 폴더).

---

## 3. 주요 인자

| 인자 | 어디 | 뜻 |
|---|---|---|
| `--dataset {vln_n1,vln_pe}` | 02·03·04 | GT를 읽는 방식. 기본 **`vln_pe`**. 출력 접미사·렌더 해상도도 여기서 정해진다 |
| `--astar_edge_check` | 03 | A* 이동 선분이 0.05 m 격자의 얇은 벽을 가로지르면 막는다. 기본 off(= 종전과 바이트 동일). `plan_r_b` 를 r_b 로 둘 때 필요 |
| `--mode {reproduce,random}` | 03 | `reproduce`=GT의 start/goal로 재계획 · `random`=임의 start/goal |
| `--mode {gt_replay,reproduce,random}` | 04 | `gt_replay`=GT pose 그대로(검증용, §4) |
| `--path_profile` | 03·03f | `legacy` / **`reproduce_v1`**(현재). 생략 시 개별 기본값 |
| `--preset measured` / `--render_profile` | 04 | 확정 렌더 설정. **생략보다 지정이 안전** |
| `--camera {dataset,d455_nominal,d455_30m}` | 04 | 노은역은 반드시 지정 |
| `--r_b` / `--h_nav_ratio` | 02·03 | 로봇 반경 / 네비 높이비 |
| `--geometry {obj,usd,both}` | 02 | 노은역은 `usd` 필수(obj 없음) |
| `--scan_coverage` | 03b·03c | mesh가 **아예 없는 칸**(건물 밖 허공)을 navigable에서 뺀다. `--mode random`에 필요 |
| `--out_dir` / `--log_dir` | 전부 | 산출물 / 리포트 위치 |

### `--scan_coverage` — 왜 필요한가

`derive_obstacle_2d`는 `(h_nav, h_obs]` 밴드에 장애물이 없으면 navigable로 친다. 그런데 밴드가
빈 이유가 **"정말 뚫린 공간"인지 "mesh가 아예 없는 스캔 밖 허공"인지 구분하지 않는다.**
실측(`s8pcmisQ38h`): navigable로 분류된 셀의 **~40 %**가 전 높이에서 mesh 점유가 0이었다.

GT는 항상 건물 안 좌표만 쓰므로 `--mode reproduce`에서는 문제가 안 된다. **`--mode random`은
navigable 전체에서 균등하게 뽑으므로 건물 밖을 "갈 수 있는 곳"으로 뽑아버린다** — 그래서 필요하다.
노은역 전용이 아니라 **mp3d에서도 `random`이면 똑같이 걸린다.**

프로파일 값은 `path_profiles.py` · `render_profiles.py` · `camera_profiles.py`에 있다.
우선순위는 **개별 인자 > 프로파일 > 스크립트 기본값**.

```
$ISAAC_PY $GS/path_profiles.py --diff legacy reproduce_v1     # 프로파일 차이 보기
$ISAAC_PY $GS/render_profiles.py                              # 렌더 프로파일 목록
```

### 확정 렌더 설정

`--light ambient_only --film_iso 70`, ambient는 vln_n1 `6.0` / vln_pe `10.0`.
`--preset measured`가 이 값을 넣는다. SSIM vln_n1 0.876 · vln_pe 0.775~0.865.

---

## 4. GT pose만 재생하는 경로 — **검증 전용**

경로 생성을 거치지 않고 **GT에 적힌 camera pose를 그대로 재생**해 이미지만 우리 렌더로 바꾼다.
"우리 렌더러가 원본과 같은 그림을 만드는가"만 묻는다.

| | 생성 경로 (§2) | GT 재생 (검증) |
|---|---|---|
| 궤적 | 우리가 계획 | GT 그대로 |
| parquet·meta | 직접 써야 함 | GT에서 **바이트 복사** |
| 쓰는 곳 | 데이터 증강 | 렌더러 검증 |

한 씬:
```
timeout --signal=KILL 900 $ISAAC_PY $GS/04_render_obs_isaac.py --dataset vln_pe --scene 17DRP5sb8fy --mode gt_replay --preset measured --out_dir $GS/logs
```

전수(데이터셋으로 저장):
```
timeout --signal=KILL 86400 $ISAAC_PY $GS/05_export_vlnpe.py --dataset vln_pe --n_scenes 0 --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe
```

실측: **61씬 · 2,813 ep · 271,252 frame · 116 GB · 5.2 h.**

> ⚠️ **씬당 프로세스 1개.** 한 프로세스에서 `build_renderer`를 두 번 부르면 `/World/RenderCamera`
> 충돌로 죽는다. 성공 판정은 **종료코드가 아니라 sentinel 파일**로 한다 — Isaac은 정상 종료에도
> 세그폴트를 내고 `simulation_app.close()`가 20분 넘게 멈추기도 한다(이미지는 그 전에 이미 저장됨).

---

## 5. 검증과 리포트

### 5.1 데이터셋 구조 게이트 — `05b_verify_*.py` (디스크만 읽음)

```
$ISAAC_PY $GS/05b_verify_vlnpe.py --render_root data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scenes all --require_complete --negative --deep --log_dir logs/gs-vlnpe
```

| 게이트 | 무엇 |
|---|---|
| V1 | 트리: 디렉토리·파일명·개수가 GT와 동일 |
| V2 | npy dtype/shape 동일, rgb·depth·parquet **행수 일치** |
| V3 | parquet 컬럼·Arrow 타입·`huggingface` 메타 동일 (복사면 sha256도) |
| V4 | depth 값역 `0 < v <= 1`, NaN 0, rgb uint8 |
| V5 | `LerobotAsLmdb` 키 수 == GT, 에피소드당 지시문 3개 |
| V6 | `cma_collate_fn` 첫 배치가 예외 없이 생성 |
| V7 | GT 대조 SSIM — **보고용, FAIL 아님** |

| 인자 | 뜻 |
|---|---|
| `--require_complete` | GT에 있는데 빠진 에피소드도 잡는다. **전수 검사 필수** |
| `--deep` | parquet·meta sha256까지 비교 |
| `--negative` | 복사본을 5가지로 망가뜨려 해당 게이트만 FAIL 하는지 확인 |
| `--self_test` | GT를 산출물 자리에 넣어 검증기 자체를 검사 |

**전수 실측** (`logs/gs-vlnpe/05b_verify_vlnpe/verify.json`): V1~V5 **61/61 PASS** ·
V7 SSIM 씬 중앙값 **0.8295**(최악 0.6669, ≥0.75가 56/61) · negative **5/5 검출**.
에피소드별 값은 `scenes.<씬>.V7.per_episode`에 있다.

### 5.2 눈으로 비교 — `04e_sample_compare.py`

```
timeout --signal=KILL 3600 $ISAAC_PY $GS/checks/04e_sample_compare.py --dataset vln_pe --n_scenes 6 --both_settings --n_frames 4 --log_dir logs/gs-vlnpe
```
`--both_settings` = 기존·확정 설정을 둘 다 렌더해 같은 프레임에서 비교.

### 5.3 경로 진단 (선택)

| 스크립트 | 무엇을 묻나 | 인자 |
|---|---|---|
| `03b_verify_reproduction.py` | 03이 낸 거리값이 **좋은 값인지** | `--mode random` 이면 `--scan_coverage` 같이 |
| `03c_compare_refine.py` | refine 해석이 `argmax`인가 `min_move`인가 | 〃 |
| `03d_grid_search_params.py` | 8개 파라미터 중 어디를 더 만질 수 있나 | 노은역은 `--mode random` |
| `03g_compare_to_reference.py` | 계획 GT(`reference_path`) 대비 거리 | — |
| `03h_violation_breakdown.py` | 관통 / r_b 미달 분리 + 출처(waypoint 점·선분 vs spline) | `--r_b --plan_r_b` |

이 넷은 **이미 답이 나온 진단**이다. 결론: `min_move` 채택, 기본값 불변, 파라미터 천장 78~79 %.

### 5.4 라이터 자기검사 (GPU 불필요)

```
$ISAAC_PY $GS/vlnpe_writer.py --self_test
$ISAAC_PY $GS/vlnce_writer.py --self_test
$ISAAC_PY $GS/vlnce_rig.py            # pixel goal 투영식 V1 — N=1701, MAE 1.22/0.94 px
```

### 5.5 리포트 위치

- 로컬: `--log_dir/<script_name>/<scene>[_random|_reproduce]/report.html` — 브라우저로 바로 열기
- 발행본 링크 모음: [`reports.md`](reports.md)

---

## 6. 파일 지도

**파이프라인 — 돌리는 것** (`gs_vlnpe/` 최상위)

| 스크립트 | 무엇 |
|---|---|
| `01_prepare_scene` | 씬 자산이 GT와 같은 world 프레임인지 판정 → `scene_meta/` |
| `02_build_freemap_esdf` | 씬을 occupancy + ESDF 지도로 → `esdf/` |
| `03_sample_gt_paths` | 그 지도 위에서 A\*로 경로 생성 → `paths/` |
| `03mf_plan_multifloor` | (선택) 층 이동 에피소드를 층별 A\* + 계단 GT 로 계획 → `paths_mf/` |
| **`03u_filter_usable`** | **사용 가능 경로만 남기고 나머지는 버림** → `usable/paths/` (04 입력). 층 이동은 기본으로 빼고, `--multifloor_mode stairs` 면 판정해 포함 |
| **`03f_reproduce_batch`** | **61씬 배치 드라이버.** `01`→`02`→`03`을 필요한 것만 돌려 "같은 루트 비율"을 잰다. 전부 CPU |
| `04_render_obs_isaac` | 경로를 따라 Isaac으로 rgb/depth 렌더 → `obs/` |
| `05_export_vlnce` / `05_export_vlnpe` | 데이터셋 레이아웃으로 저장 |
| **`05b_verify_vlnce` / `05b_verify_vlnpe`** | **export 직후 돌리는 생산 게이트.** 구조 V1~V6 + GT 대조 V7 (§5.1) |

`03f`와 `05b`는 이름이 보조처럼 보이지만 **파이프라인**이다 — `03f`는 전수 실행의 진입점이고,
`05b`는 내보낸 데이터셋이 로더에 실제로 먹히는지 확인하는 마지막 관문이다.

**라이브러리 — import 대상**

| 파일 | 내용 |
|---|---|
| `esdf_utils.py` | occupancy → ESDF, A\*, refine, 스무딩, 경로 지표 |
| `geometry_utils.py` | pose 변환, unprojection, mesh 거리 |
| `dataset_utils.py` | vln_n1 / vln_pe GT 로더, 데이터셋 레지스트리 |
| `viz_utils.py` | report.html 생성 |
| `profiles.py` / `path_profiles.py` / `render_profiles.py` / `camera_profiles.py` | 이름 붙인 설정 묶음 |
| `vlnce_writer.py` / `vlnpe_writer.py` | 데이터셋 레이아웃 라이터 |
| `vlnce_discretize.py` | 연속 경로 → 이산 action (전진 0.25 m / 회전 15°) |
| `vlnce_rig.py` / `vlnce_pixel_goal.py` | rig 기하, pixel goal 투영·가시성 |
| `vlnce_align.py` | vln_ce start-relative pose → mesh 절대좌표 |
| `usdz_scene_utils.py` | usdz(NuRec) 전용 헬퍼 |
| `multifloor_utils.py` / `stairs_surface.py` / `multifloor_filter.py` / `multifloor_render.py` | 층 이동(계단) — 분할·물리기·이음매(gt) / 높이 지도 계단 계획(plan) / 03u 위임 / 04 위임 |

**`checks/` — 검증·근거 (12개, 2026-10-11에 분리)**

| 스크립트 | 무엇을 물었나 |
|---|---|
| `00_inspect_vln_n1` | 자산 계약 — 데이터를 어떻게 읽고 쓰나 |
| `03b_verify_reproduction` | `03`이 낸 거리값이 **좋은 값인지** |
| `03c_compare_refine` | refine 해석이 `argmax`인가 `min_move`인가 |
| `03d_grid_search_params` | 8축 중 어디를 더 만질 수 있나 (51조합) |
| `03e` / `03g_compare_*` | 계획 GT(`reference_path`) 대비 커버리지 / 거리 |
| `04_render_obs` | **Open3D 대체 렌더러** — depth 정확도 검증 |
| `04c_light_explorer` · `04d_tonemap_sweep` | 조명·톤맵 스윕 |
| `04e_sample_compare` · `04f_light_gallery` | 씬 여러 개를 한 페이지에서 비교 |
| `05c_compare_vlnce` | vln_ce GT와 우리 렌더 토글 비교 |

`usd_study/`도 같은 성격이다(USD composition 실습).

다시 안 돌려도 되지만 **지우면 "왜 이 값인가"의 근거가 사라진다**. 예: `--light ambient_only
--film_iso 70`의 근거는 `04c`/`04d`가 만든 리포트에만 있다.

커맨드는 `$GS/checks/<이름>.py` 로 부른다. 라이브러리는 상위 `gs_vlnpe/` 것을 쓴다.

다시 안 돌려도 되지만 **지우면 "왜 이 값인가"의 근거가 사라진다**. 예: `--light ambient_only
--film_iso 70`의 근거는 `04c`/`04d`가 만든 리포트에만 있다.

**`apply_real/` — 노은역 전용**

진짜 다른 것만 둔다(9개): `00`·`01`·`02`(usdz ROI·`--geometry usd`), `04_render_obs*`(NuRec GS),
`04c`, `05_export_vlnce`(새 경로 저장), `camera_profiles`(vln_ce rig 5종), `03e`.
`03`·`03b`·`03c`·`03d`와 공용 라이브러리는 **상위 `$GS/` 것을 쓴다** — 사본을 두면 드리프트가
생기기 때문이다(실제로 `esdf_utils` 버그 수정과 `--path_profile`이 노은역에 전달되지 않았었다).

---

## 7. 자주 걸리는 것

| 증상 | 원인 |
|---|---|
| `/World/RenderCamera` 충돌로 죽음 | 한 프로세스에서 `build_renderer` 2회. **씬당 프로세스 1개** |
| 정상 종료인데 세그폴트 / 20분 멈춤 | Isaac `simulation_app.close()`. 이미지는 이미 저장됨 — **산출물로 판정** |
| `--rtx_ambient`가 안 먹음 | **씬 로드 후**(`build_renderer` 내부)에 걸어야 한다 |
| 10 m / 30 m 데이터셋 rgb가 다름 | far plane이 rgb도 바꾼다(실측). **혼용 금지** |
| r_b 스윕했는데 값이 안 변함 | `r_b`·`h_nav_ratio`는 ESDF에 박혀 있다. **`02`부터 다시** |
| 노은역 씬 폴더명이 잘림 | `loader.py`가 `key.split('_')`를 쓴다. `_` 없이 `noeunstationmid` |
| `ModuleNotFoundError: skimage` | `pip install scikit-image` (SSIM용) |
