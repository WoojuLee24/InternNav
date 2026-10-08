# 노은역 중간층 실제 데이터 적용 파이프라인

이 폴더는 원본 `gs_vlnpe` 단계 구조를 유지하면서 노은역
`data/noeun_station/noeun_station_collision.usdz`에 적용한 코드만 분리한 것이다.

- 논리 scene ID: `noeun_station_mid`
- 중간층 기준 바닥 높이: `floor_z = -0.05 m`
- GT path: random 20 episodes
- 관측 카메라: D455 nominal, depth scale `0.001 m/unit`
- LiDAR: Isaac Sim 공식 `OS1_REV6_32ch10hz1024res`
- Conda는 사용하지 않는다.

`noeun_station_mid`는 잘라서 새로 만든 USDZ가 아니다. 원본 USDZ에서 `floor_z=-0.05 m`
주변의 상향 수평면을 선택한 logical scene ID다. 바닥 탐색에서 승강장·중간층·상층에 대응하는
세 높이대가 확인됐고, 사용자 요청으로 중간층부터 데이터셋을 생성했다. 높이 선정 과정과
face/ROI 계산 근거는 맨 아래의 상세 재현 문서에 기록되어 있다.

## 코드와 생성물의 분리

이 폴더의 Python/Markdown 파일은 Git에 올리는 소스다. 다음 경로는 실행 시 만들어지는
산출물이므로 `.gitignore`에서 제외한다.

```text
target_schema.json
scene_meta/
esdf/
paths/
obs/
verify/
compare/
gridsearch/
format_validation/*.json
format_validation/*.html
format_validation/*_assets/
```

로그와 단계별 `report.html`은 `logs/gs-vlnpe/apply_real/`에 생성되며 저장소에 올리지 않는다.
원본 USDZ도 저장소의 `data/` ignore 정책을 따른다.

## 실행 환경

저장소 root에서 실행한다. 모든 단계는 Isaac Sim에 포함된 Python을 사용한다.

```bash
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
PIPE=scripts/dataset_converters/gs_vlnpe/apply_real
LOG=logs/gs-vlnpe/apply_real
SCENE=noeun_station_mid
USDZ=data/noeun_station/noeun_station_collision.usdz
```

위 변수는 설명을 짧게 하기 위한 shell 변수이며 Conda 환경을 만들거나 활성화하지 않는다.

입력 파일을 먼저 확인한다.

```bash
test -f $USDZ
```

각 단계는 바로 앞 단계의 canonical JSON/NPZ를 입력으로 사용한다. 중간 단계가 FAIL인데 다음
단계를 억지로 실행하지 않는다.

```text
USDZ
 └─ 00 target_schema.json
 └─ 01 scene_meta/noeun_station_mid.json
     └─ 02 esdf/noeun_station_mid.{npz,json}
         └─ 03 paths/noeun_station_mid_random.json
             └─ 04 obs/noeun_station_mid_random_isaac_d455_nominal/
```

## 00. USDZ와 목표 schema 검사

```bash
$ISAAC_PY $PIPE/00_inspect_vln_n1.py \
  --scene $SCENE --usd_path $USDZ --floor_z -0.05 \
  --out_dir $PIPE --log_dir $LOG
```

확인:

- `$PIPE/target_schema.json`
- `$LOG/00_inspect_vln_n1/noeun_station_mid/report.html`
- 6개 gate PASS
- 기준 실행: triangles 665,812, 중간층 후보 faces 14,618

00은 USDZ를 변환하거나 GT path를 생성하지 않는다. 이후 단계가 사용할 geometry·좌표·저장
schema 계약을 검사한다.

## 01. 중간층 logical scene 준비

```bash
$ISAAC_PY $PIPE/01_prepare_scene.py \
  --scene $SCENE --usd_path $USDZ --floor_z -0.05 \
  --out_dir $PIPE --log_dir $LOG
```

`noeun_station_mid`는 별도의 cropped USDZ가 아니다. 원본 USDZ와 `floor_z=-0.05` 주변의
상향 수평면 ROI를 연결하는 논리 scene ID다.

확인:

- `$PIPE/scene_meta/noeun_station_mid.json`
- `$LOG/01_prepare_scene/noeun_station_mid/report.html`
- `passed=true`, selected face 14,618, surface area 약 798.1859 m²
- `mesh_path=null`은 mesh가 없다는 뜻이 아니라 원본 `usd_path`를 직접 사용한다는 뜻

## 02. free map과 ESDF 생성

```bash
$ISAAC_PY $PIPE/02_build_freemap_esdf.py \
  --scene $SCENE --geometry usd --ref_h_b 0.875 \
  --scene_meta_dir $PIPE --out_dir $PIPE --log_dir $LOG
```

확인:

- `$PIPE/esdf/noeun_station_mid.npz`
- `$PIPE/esdf/noeun_station_mid.json`
- `$LOG/02_build_freemap_esdf/noeun_station_mid/report.html`
- grid `[1345, 1308, 40]`, voxel 0.05 m, geometry sanity PASS
- 기준 실행: occupancy fraction 0.01013146, navigable fraction 0.879639

`ref_h_b=0.875`는 obstacle height band의 reference이며 D455 depth scale이 아니다.

## 03. random GT path 20 episodes 생성

```bash
$ISAAC_PY $PIPE/03_sample_gt_paths.py \
  --scene $SCENE --mode random --num_episodes 20 \
  --esdf_dir $PIPE/esdf --out_dir $PIPE --log_dir $LOG
```

확인:

- `$PIPE/paths/noeun_station_mid_random.json`
- `$LOG/03_sample_gt_paths/noeun_station_mid_random/report.html`
- 20/20 episodes 성공, hard collision 0, 최종 `passed=true`
- 기준 실행: path length median 14.4067 m, minimum clearance 0.1581 m

`distribution_ok=false`는 Matterport/R2R 분포와의 advisory 비교이며 노은역 collision/path 성공
gate의 FAIL이 아니다.

03의 검증·비교 단계도 보존되어 있다. GT 생성의 필수 본 단계는 03이며, 아래 `03b~03e`는
재현성 확인과 파라미터 비교를 위한 후속 진단이다.

```bash
$ISAAC_PY $PIPE/03b_verify_reproduction.py \
  --scene $SCENE --mode random --num_episodes 20 --out_dir $PIPE --log_dir $LOG

$ISAAC_PY $PIPE/03c_compare_refine.py \
  --scene $SCENE --mode random --num_episodes 20 --out_dir $PIPE --log_dir $LOG

$ISAAC_PY $PIPE/03d_grid_search_params.py \
  --scenes $SCENE --mode random --num_episodes 20 --out_dir $PIPE --log_dir $LOG

$ISAAC_PY $PIPE/03e_compare_reference_path.py \
  --scene $SCENE --dataset noeun_generated --out_dir $PIPE --log_dir $LOG
```

진단 결과의 기준은 다음과 같다.

- 03b: deterministic reproduction 20/20 PASS
- 03c: argmax refine가 기준 path와 Chamfer 0이며 radius gate PASS
- 03d: 모든 parameter candidate와 regression gate PASS
- 03e: 03 route와 04 저장 pose 20/20 일치

## 04. Isaac Sim D455 관측 생성

```bash
$ISAAC_PY $PIPE/04_render_obs_isaac.py \
  --scene $SCENE --mode random --num_episodes 20 \
  --camera d455_nominal --out_dir $PIPE --log_dir $LOG
```

확인:

- `$PIPE/obs/noeun_station_mid_random_isaac_d455_nominal/camera_config.json`
- `episode_000000`부터 `episode_000019`까지 존재
- 각 frame의 `rgb/*.jpg`, `depth/*.png`, `extrinsic/*.npy` key와 개수가 일치
- 20/20 episodes PASS, 총 8,378 frames
- depth: 270×480 `uint16`, scale 0.001 m/raw, 저장 범위 0.1–10.0 m
- pose step violation 0, 전체 mesh-anchor median 약 4.62 mm
- `$LOG/04_render_obs_isaac/noeun_station_mid_random_d455_nominal/report.html`

카메라 저장 규약 비교:

| 항목 | 기존 VLN-N1 D435i dataset | 노은역 D455 nominal |
|---|---:|---:|
| depth scale | 0.0001 m/raw | 0.001 m/raw |
| uint16 유효 표현 상한 | 6.5534 m | 65.534 m |
| pipeline 저장 cutoff | 3.0 m | 10.0 m |
| image size | 480×270 | 480×270 |

65.534 m는 uint16 encoding의 수치적 상한이지 D455의 측정 성능이 아니다. 현재 04의 유효
depth cutoff는 10 m다.

기본 Open3D 경로도 비교용으로 유지하지만 기준 dataset은 위 Isaac Sim 결과다. Open3D 결과는
USDZ 재질 표현이 제한적이므로 RGB 품질 기준으로 사용하지 않는다.

```bash
EGL_PLATFORM=surfaceless $ISAAC_PY $PIPE/04_render_obs.py \
  --scene $SCENE --mode random --num_episodes 20 \
  --usd_path $USDZ --camera d455_nominal --out_dir $PIPE --log_dir $LOG
```

## 05. OS1-32 RTX LiDAR

05는 04 결과를 읽는 독립 확장이다. OS1-32 한 frame 생성, 16-gate 검증, 3-frame 통합 HTML의
복사 가능한 명령은 `lidar/README.md`에 분리했다. 현재 전체 20 episodes batch는 아직 미완료다.

## 공통 helper

`dataset_utils.py`, `esdf_utils.py`, `geometry_utils.py`, `viz_utils.py`,
`camera_profiles.py`, `usdz_scene_utils.py`는 00~05가 직접 import하므로 이 폴더에 함께 둔다.
단계 파일만 복사하고 helper를 빼면 동일한 코드로 재현할 수 없다.

## 상세 문서

박사님이 다시 실행할 때는 이 README를 먼저 사용하고, 수치의 근거나 실패 원인을 확인할 때
다음 상세 문서를 사용한다.

- 00~04 전체 과정, 바닥 높이·중간층 선정 및 실제 결과:
  `.claude/memory/codex/codex_noeun_usdz_mid_00_to_04_complete_reproduction_guide.md`
- 05 OS1-32, robot 좌표계 및 RGB-D/LiDAR calibration:
  `.claude/memory/codex/codex_noeun_rtx_lidar_complete_reproduction_guide.md`
