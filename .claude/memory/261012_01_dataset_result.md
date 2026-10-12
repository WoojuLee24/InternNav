# 작업 B 결과 — `01_prepare_scene.py` 에 `--dataset` 추가 (2026-10-12)

## 무엇이 달라졌나

`01` 이 `vln_n1` parquet(`action` / `observation.camera_extrinsic` / `observation.camera_intrinsic`)
을 직접 읽던 것을, **`--dataset` 이 `vln_n1` 이 아닐 때만** `dataset_utils.load_gt_episode` 의 공통
계약으로 위임하도록 바꿨다. 기본 경로(`vln_n1`)는 코드·출력 모두 종전 그대로다.

고친 파일: `scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py` **하나뿐** (02/03/esdf_utils/
dataset_utils 는 손대지 않음).

### 새 인자
| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--dataset` | `vln_n1` | `{vln_n1, vln_pe, vln_ce}`. 생략 시 종전 동작 |
| `--data_root` | `None` | 생략 시 `dataset_utils.default_data_root(--dataset)` |
| `--rig` | `None` | `vln_ce` 전용 카메라 rig. 생략 시 `125cm_30deg` (02/03 과 같은 이름) |

### 구조 (guard clause + 위임)
- `run_alignment_gate()` 맨 앞 2줄 guard → `run_alignment_gate_other()` 로 위임. **vln_n1 본문은 무수정.**
- `gather_gt_reference()` 맨 앞 guard → `gather_gt_reference_other()`. vln_n1 분기는 main 에 있던
  기존 4줄을 그대로 옮긴 것.
- `load_depth_frame()` / `run_gt_extent_check()` 는 02 와 같은 **exhaustive if/elif/else + `assert False`**.
- `check_gt_extent_no_ply()` 는 기존 `check_gt_extent()` 를 호출한 뒤 장애물 항목만 덜어낸다(재사용).
- 로그 경로에 02/03 과 같은 데이터셋 태그(`dataset_tag`)를 붙였다 — `vln_n1` 은 `''` 라 경로 불변.

### 왜 `--dataset` 하나로 안 끝났나 (추가로 손봐야 했던 것)
1. **검증 ③** 이 쓰는 `geometry_utils.check_against_scene_mesh()` 가 parquet 의 `action` /
   `observation.camera_intrinsic` 을 **직접** 읽는다. 그 컬럼이 없어 그대로는 터진다 →
   같은 측정을 `dataset_utils` 계약 위에서 다시 하는 `run_alignment_gate_other()` 를 01 안에 뒀다
   (geometry_utils 는 건드리지 않음).
2. **검증 ③ 허용치**(`ALIGN_TOL_M`)를 데이터셋별로 나눴다. depth 의 **출처가 다르다**:
   - `vln_n1` 1 mm — depth 가 이 obj mesh 를 그대로 렌더한 것(실측 median 0.00003 m)
   - `vln_ce` 1 mm — habitat 이 같은 mesh 를 렌더(실측 median **0.0004 m**, 17DRP·2n8k ep0/1)
   - `vln_pe` **0.15 m** — depth 는 **Isaac 이 USD 를 렌더**한 것 + pose 도 시뮬 보고값이라 obj mesh 와
     cm 단위로 어긋난다. 실측: 올바른 해석 **0.030~0.069 m** / 틀린 pose 해석 **0.24~18 m** →
     0.15 m 가 둘을 확실히 가른다. (게이트의 목적은 "같은 world 프레임인가"지 렌더러 일치가 아니다.)
3. **`meta/pointcloud.ply` 는 vln_n1 에만 있다** → 검증 ④ 의 장애물 항목을 02 와 같은 방식으로 건너뛴다.
4. **vln_ce depth 디렉터리만 rig 접미사**(`observation.images.depth.<rig>`)라 `dataset_utils` 로는 못
   읽는다 → 이미 있는 `vlnce_pixel_goal.depth_loader` 를 쓴다.

## 명령어

```
# 회귀(기본 경로) — 기존과 바이트 동일해야 한다
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py --scene 17DRP5sb8fy --out_dir /tmp/b_check --log_dir /tmp/b_check_logs

# 15씬 01
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py --dataset vln_pe --scene <SCENE> --out_dir /tmp/b_out --log_dir /tmp/b_logs

# 15씬 02
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --dataset vln_pe --scene <SCENE> --r_b 0.20 --h_nav_ratio 0.12 --out_dir /tmp/b_out --scene_meta_dir /tmp/b_out --log_dir /tmp/b_logs

# 15씬 03 (프롬프트 2026-10-12 갱신판 설정)
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --dataset vln_pe --mode reproduce --scene <SCENE> --r_b 0.20 --plan_r_b 0.28 --h_nav_ratio 0.12 --refine_radius 0.20 --post_refine --post_refine_margin 0.05 --num_episodes 999 --esdf_dir /tmp/b_out/esdf --out_dir /tmp/b_paths --log_dir /tmp/b_logs
```
인자 뜻 — `--r_b 0.20` 로봇 반경(검사 기준) · `--plan_r_b 0.28` 계획 여유(스무딩이 깎아먹는 몫 포함) ·
`--h_nav_ratio 0.12` 지면 여유고 = 0.12·h_b · `--refine_radius 0.20` greedy refine 반경 ·
`--post_refine --post_refine_margin 0.05` 위반 구간 수리 후 재스무딩(목표를 r_b+0.05 로 잡음) ·
`--num_episodes 999` 전수.

## 검증 결과

### 1. 회귀 — 통과
- `17DRP5sb8fy`, `s8pcmisQ38h` 두 씬 모두 `cmp` **바이트 동일**. (로그 태그 추가 후 재확인도 동일)
- negative test `--pose_convention cam2world` → `FAIL (alignment=False)`, exit 1. 게이트는 여전히 떨어진다.
- `--dataset vln_ce --scene 17DRP5sb8fy` 스모크: median 0.000377 m, PASS.

### 2. 15씬 `01` — **15/15 PASS**
- mesh 표면거리 median **0.023~0.069 m** (허용 0.15) — 전 씬이 같은 world 프레임.
- `h_b` **1.654~1.694 m** (H1 카메라 높이 1.66 가정과 일치), 검증 ④ 의 0.2~2.5 m 범위 안.
- `floor_z` 는 **episode 0 기준**이라 다층 씬에서는 그 층의 바닥이 된다
  (`e9zR4mvMWw7 +2.75`, `Uxmj2M2itWa +0.70`, `kEZ7cmS4wCh -0.45` 등). 03 이 `--floor_tol_m 0.3` 으로
  에피소드마다 보정하므로 경로에는 영향 없다.

### 3. 15씬 `02` — **14/15 PASS** (`qoiz87JEwZ2` 만 FAIL, 아래 참고)

### 4. 15씬 `03` — **14/15 에서 경로 생성** (`HxpKQynjfin` 만 0건, 아래 참고)
합계 **계획 성공 387 에피소드 / 같은 루트 322 (83.2%) / 궤적 clearance 위반 0건**

| scene | planned | same_route | chamfer med | Fréchet med |
|---|---|---|---|---|
| rPc6DW4iMge | 53 | 50 | 0.146 | 0.448 |
| PX4nDJXEHrG | 50 | 39 | 0.219 | 0.615 |
| kEZ7cmS4wCh | 42 | 33 | 0.199 | 0.537 |
| aayBHfsNo7d | 37 | 24 | 0.224 | 0.551 |
| qoiz87JEwZ2 | 34 | 29 | 0.137 | 0.394 |
| 5q7pvUzZiYa | 31 | 25 | 0.168 | 0.414 |
| V2XKFyX4ASd | 29 | 27 | 0.152 | 0.448 |
| e9zR4mvMWw7 | 28 | 26 | 0.156 | 0.461 |
| Uxmj2M2itWa | 27 | 22 | 0.160 | 0.425 |
| b8cTxDM8gDG | 24 | 21 | 0.147 | 0.359 |
| VVfe2KiqLaN | 21 | 17 | 0.221 | 0.615 |
| Pm6F8kyY3z2 | 6 | 4 | 0.200 | 0.639 |
| GdvgFV5R1Z5 | 4 | 4 | 0.187 | 0.410 |
| gZ6f7yhEvPG | 1 | 1 | 0.140 | 0.369 |
| HxpKQynjfin | **0** | - | - | - |

## ⚠️ 보고할 것 두 가지 (코드 문제 아님 — 데이터/파라미터)

### (1) `qoiz87JEwZ2` — 02 clearance 게이트 FAIL. 원인은 **깨진 에피소드 0**
`episode_000000.parquet` 이 **프레임 2개**짜리에 이동거리 3 cm x 9 cm 인 사실상 빈 에피소드다.
게다가 이 씬은 다층인데 ep0 만 위층(`floor_z +0.06`)이고 실제 에피소드 대부분은 `-3.25` 다.
- 02 의 clearance 게이트는 항상 ep 0..N-1 을 보므로 ep0 의 `zero_frac=100%` 가 게이트를 떨어뜨린다
  (ep1~4 는 median 0.35~0.65 m 로 멀쩡). **ESDF npz 자체는 정상 생성**됐고 03 도 34/46 성공했다.
- 01 도 ep0 에서 `floor_z` 를 읽으므로 scene_meta 의 `floor_z=+0.059` 는 "위층 바닥"이다.
  03 은 `--floor_tol_m` 으로 에피소드마다 보정하니 경로에는 영향 없고, 02 의 **참조 레이어 시각화**
  에만 영향 있다.
- 판단 필요: 이대로 쓸지(권장 — 수치는 정상), 02/01 에 "참조 에피소드 선택" 옵션을 넣을지.
  옵션 추가는 02 수정이 필요해 이번 작업 범위(01 만) 밖이라 하지 않았다.

### (2) `HxpKQynjfin` — `--plan_r_b 0.28` 에서 5/5 전부 `start_or_goal_not_navigable`
좁은 씬이다. GT 궤적의 clearance 가 median 0.215~0.408 m, **min 0.150 m** 라 start/goal 셀이
0.28 m 기준을 못 넘는다.
- 같은 씬을 `--plan_r_b 0.25` 로 돌리면 **2/5 성공**(같은 루트 2/2, chamfer 0.089, Fréchet 0.286).
- 즉 0.28 로 올린 대가로 이 씬이 통째로 빠진다. 작업 A 와 설정을 맞추려고 **0.28 결과를 정본으로
  두고** 보고만 한다 — 0.25 로 내릴지는 전체 트레이드오프(관통/r_b 미달)와 같이 볼 문제다.

## 산출물 위치
- `scene_meta/` 15개 → `scripts/dataset_converters/gs_vlnpe/logs/scene_meta/` 에 **없던 것만** 복사
  (총 46 → **61**). 기존 46개는 건드리지 않음.
- `esdf/<scene>_vlnpe.npz|json` 15개 → `logs/esdf/` 에 없던 것만 복사.
- `paths/<scene>_vlnpe.json` 14개 → `logs/paths/` 에 없던 것만 복사 (`--update=none`).
- 원본·로그: `/tmp/b_out`, `/tmp/b_paths`, `/tmp/b_logs`.
