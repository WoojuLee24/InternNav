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
| 시뮬레이터 규약 | Habitat | Isaac Sim (H1 휴머노이드) | — (GT 없음) |
| 기존 GT 궤적 | 있음 | 있음 | **없음** |
| 이미지 | 640×480 | 256×256 | 480×270 |
| action | 이산 0.25 m / 15° | 이산(물리 롤아웃) | 이산화해서 생성 |
| 쓰는 모델 | InternVLA-N1 | CMA / Seq2Seq / RDP | (vln_ce로 내보냄) |
| 스크립트 위치 | `$GS/` | `$GS/` | `$GS/apply_real/` |

`vln_n1`은 네 번째 대상이 아니라 **`$GS`가 읽는 원본 데이터셋**이다(`--dataset vln_n1`, 기본값).

**노은역만 다른 점** — GT가 없어서:
- `02`가 geometry만으로 지도를 만든다 (`--geometry usd` 필수, `--ref_h_b`로 로봇 키를 직접 준다)
- `03`은 `--mode random`만 가능하다 (`reproduce`는 재현할 GT가 없다)
- `04`는 `--camera` 프로파일을 반드시 지정한다 (기본 `dataset`은 exit 2)
- 스캔에 구멍이 있어 검증 단계에 `--scan_coverage`를 켠다

---

## 2. 파이프라인 — 경로 생성부터 렌더링까지

```
01 → 02 → 03 → 04 → 05 → 05b
```

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 01 | `01_prepare_scene.py` | 씬 mesh/USD + GT | `<out>/scene_meta/<scene>.json` (floor_z 등) |
| 02 | `02_build_freemap_esdf.py` | 씬 mesh/USD | `<out>/esdf/<scene>.npz` (occupancy + ESDF) |
| 03 | `03_sample_gt_paths.py` | ESDF (+ GT의 start/goal) | `<out>/paths/<scene>.json` (경로 + GT 대조 지표) |
| 04 | `04_render_obs_isaac.py` | 경로 + 씬 | `<out>/obs/<scene>_<mode>_isaac_<cam>/` (rgb/depth) |
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

### 61씬 전수로 경로만 (CPU, Isaac 안 띄움)

`01` → `02` → `03`을 필요한 것만 돌려 **같은 루트 비율**을 잰다.

```
timeout --signal=KILL 21600 $ISAAC_PY $GS/03f_reproduce_batch.py --dataset vln_pe --n_scenes 0 --path_profile reproduce_v1 --work_dir $GS/logs --log_dir logs/gs-vlnpe/03f_v1
```

실측 45씬 2,274 ep: 같은 루트 **78.8 %**, 하드 충돌 **0**, 쓸 수 있는 ep **1,700**.

> ⚠️ 이 경로들을 **렌더해서 저장하는 `07_render_paths_to_vlnpe.py`는 아직 없다**(Phase C).
> action 이산화(`docs/gs-vlnpe-l1-l2a.md` §5.2)와 조립이 필요하다. 라이터 `vlnpe_writer.py`는 있다.

---

## 3. 주요 인자

| 인자 | 어디 | 뜻 |
|---|---|---|
| `--dataset {vln_n1,vln_pe}` | 02·03·04 | GT를 읽는 방식. 기본 `vln_n1` |
| `--mode {reproduce,random}` | 03 | `reproduce`=GT의 start/goal로 재계획 · `random`=임의 start/goal |
| `--mode {gt_replay,reproduce,random}` | 04 | `gt_replay`=GT pose 그대로(검증용, §4) |
| `--path_profile` | 03·03f | `legacy` / **`reproduce_v1`**(현재). 생략 시 개별 기본값 |
| `--preset measured` / `--render_profile` | 04 | 확정 렌더 설정. **생략보다 지정이 안전** |
| `--camera {dataset,d455_nominal,d455_30m}` | 04 | 노은역은 반드시 지정 |
| `--r_b` / `--h_nav_ratio` | 02·03 | 로봇 반경 / 네비 높이비 |
| `--geometry {obj,usd,both}` | 02 | 노은역은 `usd` 필수(obj 없음) |
| `--scan_coverage` | 03b·03c | 점유격자를 실제 스캔 영역으로 제한. **노은역만** |
| `--out_dir` / `--log_dir` | 전부 | 산출물 / 리포트 위치 |

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

**전수 실측** (`logs/gs-vlnpe/06b_verify_vlnpe_render/verify.json`): V1~V5 **61/61 PASS** ·
V7 SSIM 씬 중앙값 **0.8295**(최악 0.6669, ≥0.75가 56/61) · negative **5/5 검출**.
에피소드별 값은 `scenes.<씬>.V7.per_episode`에 있다.

### 5.2 눈으로 비교 — `04e_sample_compare.py`

```
timeout --signal=KILL 3600 $ISAAC_PY $GS/04e_sample_compare.py --dataset vln_pe --n_scenes 6 --both_settings --n_frames 4 --log_dir logs/gs-vlnpe
```
`--both_settings` = 기존·확정 설정을 둘 다 렌더해 같은 프레임에서 비교.

### 5.3 경로 진단 (선택)

| 스크립트 | 무엇을 묻나 | 인자 |
|---|---|---|
| `03b_verify_reproduction.py` | 03이 낸 거리값이 **좋은 값인지** | 노은역은 `--mode random --scan_coverage` |
| `03c_compare_refine.py` | refine 해석이 `argmax`인가 `min_move`인가 | 〃 |
| `03d_grid_search_params.py` | 8개 파라미터 중 어디를 더 만질 수 있나 | 노은역은 `--mode random` |
| `03g_compare_to_reference.py` | 계획 GT(`reference_path`) 대비 거리 | — |

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

**파이프라인 — 돌리는 것**

`01_prepare_scene` · `02_build_freemap_esdf` · `03_sample_gt_paths` · `03f_reproduce_batch` ·
`04_render_obs_isaac` · `05_export_vlnce` / `05_export_vlnpe` · `05b_verify_vlnce` / `05b_verify_vlnpe`

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

**근거 — 이미 답이 나온 일회성 실험**

`00_inspect_vln_n1` · `03b` · `03c` · `03d` · `03e` · `03g` · `04_render_obs`(Open3D) ·
`04c_light_explorer` · `04d_tonemap_sweep` · `04e_sample_compare` · `04f_light_gallery` ·
`05c_compare_vlnce` · `usd_study/`

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
