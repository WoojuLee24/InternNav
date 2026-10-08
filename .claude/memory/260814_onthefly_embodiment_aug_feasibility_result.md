# On-the-fly embodiment augmentation — 구현 가능성 판정 & 실행 계획

작성 2026-08-14. 근거 문서: `/ws/src/wiki/VLN/paper writing/VLN collision.md`
§"기존 방법의 문제점과 해결 방법v0.2" (L136–176). **경로 주의: `paper writing` (언더스코어 아님)**

## 목적

해결 방법 v0.2의 세 안이 InternNav에서 **학습 dataloader의 on-the-fly augmentation**으로
구현 가능한지, GT와 비교 가능한지 확정한다.

- **해결 3** — 기존 데이터 재활용. instruction 고정, 시야각 내 RGB/depth로 장애물 생성 + `e` 변화 → BEV에서 GT path 수정
- **해결 3-2** — 3D map에서 재생성 (3D occ→2D occ→A*→렌더)
- **paired counterfactual** — 같은 장면 × 다른 `e`를 한 batch에

**전제(사용자 확정)**: augmentation은 **학습 pipeline의 dataloader에서 on-the-fly**여야 한다.
3-2처럼 3D map을 쓰는 경우에도 **map 저장만 offline이고 map→관측 변환은 on-the-fly**여야 한다.
`scripts/dataset_converters/gs_vlnpe/`는 offline 데이터 **생성** 도구이므로 augmentation 본체가 아니다 —
이 계획에서는 **(a) offline occupancy 빌더, (b) 검증 oracle** 두 역할로만 쓴다.

**환경 제약**: 현 docker에서 habitat 평가 불가 (`habitat_sim`/`habitat` 미설치, magnum 미빌드.
`/ws/src/habitat-sim`·`/ws/src/habitat-lab`에 소스만 존재). **Isaac 평가만 가능.**
→ habitat 관련 항목은 **새 docker에서 별도 수행**(§7).

---

## 1. 학습 dataloader 실측

`internnav/dataset/internvla_n1_lerobot_dataset.py` / `NavPixelGoalDataset`

### 1-1. 학습은 `vln_ce`만 읽는다
- `scripts/train_eval/qwenvl_train/default_config.py:119-120`
  `vln_datasets="r2r_125cm_0_30%30,r2r_60cm_15_15%30"`, `data_root=".../InternData-N1-v0.5-mini/vln_ce"`
- `data_dict` 프리셋(`:60-150`)의 `(height, pitch_1, pitch_2)` = **vln_ce에 미리 렌더된 5개 리그**
  (`125cm_{0,30,45}deg`, `60cm_{15,30}deg`) → 현재 `e`는 **데이터셋 선택 축**이고,
  이미 두 리그를 30%/30%로 섞어 학습 중이다.
- `vln_n1`(`observation.images.rgb/`), `vln_pe`(`.npy` 배열)는 **현 dataloader가 못 읽는다**(경로·포맷 상이).

### 1-2. `__getitem__`이 손에 쥔 것 (`:1067-1286`)

| 항목 | 값 | 위치 |
|---|---|---|
| `scene_id` | `video` = `{data_root}/traj_data/r2r/{scene_id}/videos/chunk-000` → **경로에서 복원 가능** | `:837` |
| world pose | `pose.{height}cm_{pitch_2}deg` 4×4 시퀀스 (윈도우 구간) | `:796, 838` |
| RGB | `traj_images [N,224,224,3]` float[0,1] (lookdown, pitch_2) | `:1224,1238` |
| depth | `traj_depths [N,224,224]` **metric m** (`/1000`, 5.0m clip) | `:1052-1065,1239` |
| GT path | `traj_poses [N, predict_step_num, 3]` | `:1229-1240` |
| `e` 일부 | `traj_cam_heights`(m), `traj_cam_pitch_1/2`(deg) | `:1241-1243` |
| intrinsics | **없음** — 하드코딩 `fx=fy=388.19, cx=319.5, cy=239.5` | `:1027`, `internvla_n1_unified_provider.py:56-59` |

- GT path는 **planning 결과가 아니라 기록된 extrinsic replay**
  (`get_trajectory_relative_to_frame:632` → `interpolate_and_resample_trajectory:611` → cubic spline).
- BEV는 이 depth에서 **하류에서** 계산된다(`internvla_n1_unified_provider.py:203-255`)
  → **depth만 바꾸면 BEV가 자동 반영**되고 provider를 건드릴 필요가 없다.
- **현 BEV 경로에 robot radius dilation이 전혀 없다**
  (`internnav/model/utils/depth_rgb_to_bev_torch.py:319` `depth_to_bev_occ_ros2`,
  해상도 ≈4.46 cm/px, 높이 밴드 `[z_min,z_max]`만) → **논문 C1이 메울 구멍**.
- 기존 augmentation은 S2 RGB photometric(ColorJitter 등)뿐 (`internnav/trainer/internvla_n1_trainer.py:141-153`).
  depth/BEV/GT path는 건드리지 않는다.

### 1-3. 예산
`dataloader_num_workers=4` (`default_config.py:67`), CPU worker, GPU·시뮬레이터 없음.
현재도 `__getitem__`마다 jpg/png 수십 장을 읽는 IO-bound 구조.

---

## 2. On-the-fly 가능/불가능 판정 ★핵심

| 연산 | on-the-fly? | 근거 |
|---|---|---|
| 3D occupancy 로드 | **offline 저장 → worker RAM 상주** | 씬당 0.1–0.5 MB(`260803` 실측) × 61씬 ≈ **20 MB** |
| slab `(h_nav, h_b]` + 장애물 raster + `r_b` dilation → 2D navigable/ESDF | **✓ ms** | `esdf_utils.py` `derive_obstacle_2d:263` / `compute_esdf_2d:279` / `truncate_navigable:286` — 순수 numpy/scipy |
| A* + refine + spline → 새 GT path | **✓ ms** | `astar:339`, `greedy_refine:405`, `thin_waypoints:487`, `smooth_cubic_spline:521`. **2D bool 격자만 받아 world 좌표 의존이 없음 → 그대로 재사용** |
| **BEV occupancy 관측 생성** | **✓ ms — 이미지 렌더러 불필요** | 2D navigable을 로봇 pose로 crop/rotate. FOV 제한은 `depth_rgb_to_bev_torch.py:280` `_raycast_free_torch` 재사용 |
| depth에 장애물 합성 | ✓ | 박스 투영 후 z-buffer min |
| RGB에 장애물 합성 | △ | flat-shading 수준. 사실성이 한계 |
| `cam_height`/`pitch` **이산** 변경 | **✓ 무료** | vln_ce에 5개 리그가 이미 저장 — 파일 경로 문자열만 교체 |
| `cam_height`/`pitch` **연속** 변경 (FPV RGB 재렌더) | **✗** | Isaac `SimulationApp`은 worker당 불가; Open3D는 씬 mesh(수백 MB)를 worker마다 상주시켜야 하고 씬이 랜덤이라 캐시가 깨짐 |

### 결론 3줄
1. **S1이 BEV를 쓰면 on-the-fly augmentation이 완전히 성립한다** — 렌더러 없이 `e`가 관측(BEV)과
   GT path를 동시에 바꾼다. 논문 §제안 C1·차별점(나)가 그대로 구현된다.
   지원 config 이미 존재: `image_base/s1.bev.rgb.{concat,replace}_s2.fpv.py`, `image/s1.bev.occ.*`
2. **S2 FPV RGB의 연속 시점 변화만 불가능** → 이산 5리그 + 장애물 합성으로 제한. 논문 한계 절에 명시.
3. **정보 누수 주의**: cached 3D occ에서 만든 BEV는 전역 관측이다. 논문 L316 경고대로
   **FOV/raycast로 부분 관측으로 잘라야** 입력(부분)과 supervision(전역)의 구분이 유지된다.

### 데이터셋별

| | `vln_ce` | `vln_n1` | `vln_pe` |
|---|---|---|---|
| 현 dataloader가 읽는가 | **✓ (유일)** | ✗ | ✗ |
| on-the-fly aug 부착 | **✓ 즉시** | dataloader 신규 필요 | dataloader 신규 필요 |
| metric depth | ✓ 16bit png/1000 | ✓ | △ 0–1 정규화(×10) |
| intrinsics | ✗ 하드코딩 — **검증 게이트 G1** | ✓ per-step 3×3 | ✗ (USD 유도 fx=128) |
| rgb/depth 정합 | 미검증 | ✓ | **✗ 매 프레임 ±4px 비정렬**(실측, 사후보정 불가 — `260807` 메모리) |
| `e` 다양성 | 이산 5리그, `r_b` 축 없음 | `h_b` 0.251–1.493 연속, pitch 0–30° | H1 1종 |
| 씬 mesh | `mp3d_ce/<s>/<s>.glb` (동일 scan이 `mp3d_n1` `.obj`, `mp3d_pe` `.usd`로도 존재) | `mp3d_n1` `.obj` + subset 내 `meta/pointcloud_obstacle.npy` | `mp3d_pe` `.usd` |

→ **on-the-fly augmentation의 대상은 `vln_ce`다.** `vln_pe`는 rgb/depth 비정렬로 2D 합성에 부적합.

> 논문의 "(N) 저장 비용" 논변이 vln_ce에서 수치화된다: 5개 리그를 미리 렌더해 저장하느라 **346 GB**,
> 리그 1개 추가 ≈ +30 GB, 그런데 **`r_b`는 아예 축이 없다.**

---

## 3. 시뮬레이션 직접 조작 (평가 측, Isaac)

| 조작 | 가능? | 방법 | GT 비교 |
|---|---|---|---|
| **장애물 spawn** | **✓ config만으로** | `TaskCfg.objects` → `internutopia/core/task/task.py:104` `init_objects()`가 **에피소드마다** 실행. 타입 `UsdObjCfg/DynamicCubeCfg/VisualCubeCfg`. 주입: 전역 `task.task_settings['objects']` 또는 에피소드별 `internnav/env/utils/episode_loader/generate_episode.py:86-114` | **GT 재생성 필요** |
| **`r_b` 변화** | **✓** | `task.robot_platform_size` (`internnav/configs/evaluator/vln_default_config.py:247-248`), 기본 0.3 | **GT 재생성 필요** |
| **`cam_height`/`pitch`** | **✓** | `task.camera_translation` / `camera_orientation` (`sensors/vln_camera.py:47-62`). 선례 `scripts/eval/configs/h1_custom_cam_cfg.py:24-91` | **GT 불변 → 원본과 직접 비교 가능** |
| `h_b` | ✗ | 충돌 검사에 높이 밴드 개념 없음 | — |
| 물리 body scale | ✗ (flash 모드는 텔레포트라 사실상 무관) | `robot_settings['scale']` 미노출 | — |
| FOV | ✗ | USD 카메라 prim 고정 | — |

- **에셋 로컬 보유**: `scene_data/n1_eval_scenes/internscenes_commercial/models/object/` **30,958개 USD**
  (58+19 카테고리), `internscenes_home/` **33,330개**. 추가 다운로드 불필요.
- **충돌 검사가 spawn한 물체를 자동으로 본다**: flash collision은 PhysX contact가 아니라
  **top-down depth 기반 free map**
  (`internnav/env/utils/internutopia_extension/controllers/vln_move_by_flash_with_collision_controller.py:126-225`)
  이라, 렌더에 보이는 prim은 그대로 반영된다.
- **지표**: `internnav/evaluator/utils/result_logger.py:236-342` `finalize_all_results` →
  `TL/NE/FR/StR/OS/SR/SPL/nDTW` + **`CR`(:329, 총충돌/총스텝), `CFSR`(:330)**.
  `flash_collision ∈ {'stop','reset'}`일 때만 충돌 계수 동작.
- ⚠️ **함정**: 장애물을 넣거나 `r_b`를 키우면 `raw_data`의 `reference_path`가 **거짓 GT**가 된다
  (장애물 관통 경로일 수 있음). SR/SPL/nDTW를 원본 GT 대비로 재면 비교가 무효.
  → **GT도 같은 (장애물, `e`)로 재생성해야 성립.** 파이프라인은 `gs_vlnpe` 02/03에 이미 있고,
  빠진 조각은 **"장애물 footprint를 occupancy에 OR"** 하나뿐이다.

---

## 4. 실행 계획

### Phase 1 — 게이트 검증 (신규 코드 최소, 시뮬 불필요)

**G1. `vln_ce` pose ↔ mesh 좌표 정합 + intrinsics 검증** ← *가장 중요. 실패하면 전체 계획이 무너진다*

`pose.125cm_0deg` + depth로 point cloud를 복원해 씬 mesh 표면거리를 잰다.
`gs_vlnpe/geometry_utils.py`의 `check_against_scene_mesh` / `unproject_to_world_frame`을 **그대로 재사용**
(신규 스크립트 `scripts/debugging/verify_vln_ce_pose_mesh.py` 1개).

- 통과: median 표면거리 `< 0.01 m` → 하드코딩 intrinsics·pose 규약 유효, cached occupancy 인덱싱 가능
- 실패: 축 규약 후보 3종(`cam2world_gl` / `cam2world` / `world2cam`)을 순회해 재판정
  (`understanding_gs_vlnpe_fpv_bev_geometry.md` §1)

**G2. `e`가 GT path를 실제로 바꾸는가** — 같은 씬·같은 start/goal에서 `r_b` sweep

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --dataset vln_n1 --scene 17DRP5sb8fy --mode reproduce --r_b 0.35 --h_nav_ratio 0.12 --out_dir logs/260814_rb_sweep/rb035
```
| argument | 의미 |
|---|---|
| `--dataset vln_n1` | GT 궤적 로더 선택 (`dataset_utils.py` registry) |
| `--scene` | MP3D scan id |
| `--mode reproduce` | GT의 start/goal·`h_b`·pitch를 그대로 재현 (random 아님) |
| `--r_b` | 로봇 반경 — navigable dilation 반경 |
| `--h_nav_ratio` | `h_nav = h_b × ratio` (밟고 넘는 지면 높이) |
| `--out_dir` | 경로 json + report.html 출력 |

`--r_b` ∈ {0.20, 0.25, 0.35, 0.50} × `--h_nav_ratio` ∈ {0.08, 0.12}.
지표는 03이 이미 계산하는 chamfer / clearance / homotopy.
**`r_b`를 키워도 경로가 안 바뀌면 논문 전제가 이 씬에서 성립하지 않는다.**

**G3. on-the-fly 예산 실측** — cached occupancy에서 `slab+dilation+ESDF+A*+spline` 1회 시간.
목표 **< 20 ms/sample**(worker 4개 기준, 현재 IO 대비 무시 가능). 넘으면 A* 창을 로컬 윈도우로 제한.

### Phase 2 — offline occupancy 빌드 (한 번, `vln_ce` 61씬)

`02_build_freemap_esdf.py`에 `--dataset vln_ce` 항목 **추가**
(`dataset_utils.py`의 기존 registry에 1항목 — 기존 로더 무수정).

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --dataset vln_ce --scene 17DRP5sb8fy --geometry obj --out_dir data/occ_cache
```
| argument | 의미 |
|---|---|
| `--geometry obj` | mesh 소스 (`obj`/`usd`/`both`). vln_ce는 동일 scan의 `mp3d_n1` `.obj` 사용 |
| `--out_dir` | `<scene>.npz`(3D occupancy + origin + voxel + floor_z) 출력 |

산출물 총 ≈ 20 MB.

### Phase 3 — on-the-fly augmentation 구현

**guideline §1·§3 준수: 신규 파일만 추가, 기존 파일은 guard clause 1줄씩만.**

1. **`internnav/dataset/embodiment_augment.py`** (신규)
   - `class EmbodimentAugmenter`: `__init__(occ_cache_dir, cfg)`에서 npz lazy-load + LRU 캐시(worker 내 상주)
   - `sample_e()` → `(r_b, h_b, h_nav)` **연속** 샘플링 (범위는 config)
   - `augment(sample, scene_id, world_poses, e)` → `traj_depths` / `traj_poses` (+선택 `traj_images`) 갱신
   - 내부는 `esdf_utils`의 순수 함수 재사용 — **재구현 금지**
   - 가상 장애물: occupancy에 박스 raster OR → 같은 경로로 GT 재계획 + depth 합성
   - **FOV 제한 필수**(정보 누수 방지): `_raycast_free_torch` 방식으로 관측 BEV를 부분 관측으로 자름
2. **`internvla_n1_lerobot_dataset.py`** — `__getitem__` 말미(`:1272` 부근) guard clause 1줄
   `if self.augmenter is not None: sample = self.augmenter.augment(...)`,
   `__init__`에 `self.augmenter = EmbodimentAugmenter(...) if getattr(data_args,'embodiment_aug',False) else None`
3. **`default_config.py`** — `Params`에 기본값 off 필드 추가
   (`embodiment_aug: bool = False`, `occ_cache_dir`, `e_ranges`, `obstacle_prob`) + `train_argv()` 전달
4. **신규 config** `scripts/train_eval/qwenvl_train/image_base/s1.bev.occ_aug_s2.fpv.py`
   (기존 `s1.bev.*`를 `replace` 후 `embodiment_aug=True`)
5. **paired counterfactual**: 같은 `(ep_id, start_frame_id)`를 `e`만 달리해 K개 만들어 batch에 함께 넣는 sampler.
   초기엔 **pairing만**(consistency loss 없음) — 논문 §제안 C3 권고

### Phase 4 — Isaac 평가 (장애물 spawn + `e` sweep)

신규 config 2개 (기존 파일 무수정):
- `scripts/eval/configs/h1_internvla_n1_obstacle_cfg.py` — `task_settings['objects']`에 `UsdObjCfg` 리스트
- `scripts/eval/configs/h1_internvla_n1_rb_sweep_cfg.py` — `robot_platform_size`만 변경

```
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/s1.bev.occ_aug_s2.fpv.py --machine h1 --no-train --debug-dir logs/260814_obstacle_rb --model-path <ckpt> --flash-collision stop
```
| argument | 의미 |
|---|---|
| `--machine h1` | Isaac 단일 프로세스 eval (`_isaac_sim/python.sh scripts/eval/eval.py`) |
| `--no-train` | 학습 건너뛰고 평가만 |
| `--debug-dir` | `<dir>/eval_isaac`에 디버그 이미지·로그 |
| `--model-path` | 체크포인트 (없으면 `find_best_checkpoint`) |
| `--flash-collision stop` | 충돌 시 텔레포트 중단 → `CR`/`CFSR` 집계 활성화 |

---

## 5. 검증

| 대상 | 방법 | 통과 기준 |
|---|---|---|
| G1 정합 | mesh 표면거리 median | `< 0.01 m` |
| G2 `e` 효과 | 03 리포트 HTML | `r_b` 0.20→0.50에서 homotopy가 바뀌는 에피소드 ≥ 1건 |
| G3 예산 | 1000회 평균 | `< 20 ms/sample` |
| Phase 3 기본경로 무변경 | `embodiment_aug=False`로 1스텝 학습, loss가 기존과 동일 | 동일 |
| Phase 3 신규기능 | `--debug-dir`로 BEV/depth/GT path 오버레이 저장, 같은 프레임 × `r_b` 4값 비교 | 경로가 눈으로 달라야 함 |
| Phase 4 | `result_h1.json`의 `CR`/`CFSR` | 장애물 추가 시 CR 상승, `robot_platform_size` 상승 시 CR 단조 증가 |
| 회귀 | `python scripts/dataset_converters/gs_vlnpe/esdf_utils.py`, `geometry_utils.py` self-check + `pytest scripts/debugging/unit_test/test_unified_image_provider.py` | 전부 통과 |

---

## 6. 한계 (논문에 명시할 것)

1. **FPV RGB의 연속 시점 변화는 on-the-fly 불가** — 이산 5리그 + 장애물 합성으로 제한.
   `e`가 관측을 바꾼다는 주장은 **BEV 경로에서만 완전**하다.
2. **`h_b` 축은 GT 생성(오프라인 occupancy slab)에서만** — Isaac 충돌 검사에 높이 밴드가 없다.
3. **RGB 합성 장애물의 사실성이 상한** — Phase 4의 실제 spawn과 대조해 sim-to-sim gap을 보고.
4. **cached occupancy는 전역 관측** — FOV 제한을 안 걸면 정보 누수 비판을 받는다.

---

## 7. 별도 docker 작업 — habitat

현 컨테이너에서 확인 불가로 **새 docker에서 수행**(사용자 결정, 2026-08-14).

- 미설치 실측: `python3 -c "import habitat_sim"` → `ModuleNotFoundError`.
  강제 `PYTHONPATH=/ws/src/habitat-sim/build/lib.linux-x86_64-cpython-312` → `No module named 'magnum'`
  (magnum이 vendored 빌드만 되고 site-packages에 설치 안 됨)
- 새 docker에서 확인할 것:
  1. `runner.py --machine {5090,h200} --no-train`으로 habitat 평가가 도는지
  2. embodiment 조절 키 — `scripts/eval/configs/objectnav_hm3d.yaml:27,34,36-37`이 예시:
     `sim_sensors.*.position: [0,1.25,0]`, agent `height`, `radius`, `habitat_sim_v0.allow_sliding`
     (`vln_r2r_mini.yaml`에는 agent height/radius가 없어 habitat 기본값 사용 중)
  3. **`allow_sliding: False`로 두어야 CR이 0이 아니게 나온다** (논문 L307)
  4. `CollisionsMeasurementConfig()`가 `habitat_vln_evaluator.py:297`에 이미 등록돼 있으나
     eval 루프(`:818-849`)가 `collisions`를 **읽지 않는다** → 로깅 추가 필요
  5. habitat 장애물 spawn(rigid object manager)은 이 repo에서 전혀 사용되지 않음 — 신규 구현 대상

---

## 참고 파일

- 문제/해결 원문: `/ws/src/wiki/VLN/paper writing/VLN collision.md` L136–176 (§제안 방법 L191–222)
- 자매 계획: `/ws/src/wiki/VLN/plan/gs-to-vlnpe-dataset-plan.md`, `/ws/src/wiki/VLN/plan/execution-staged.md`
- 좌표 규약: `.claude/memory/understanding_gs_vlnpe_fpv_bev_geometry.md`
- occupancy 파이프라인: `.claude/memory/260803_gs_vlnpe_02_esdf_result.md`
- vln_pe 포맷/비정렬 실측: `.claude/memory/260807_gs_vlnpe_vlnpe_dataset_result.md`
