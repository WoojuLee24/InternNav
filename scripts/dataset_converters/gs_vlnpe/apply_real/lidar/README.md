# 노은역 OS1-32 Isaac Sim RTX LiDAR

이 디렉터리는 `apply_real`의 03 GT path와 04 Isaac D455 관측 pose를 읽어 RTX LiDAR를
추가한다. 기존 RGB/depth/extrinsic은 수정하지 않는다.

## 현재 확정된 계약

- Isaac Sim 공식 model/variant: `OS1` / `OS1_REV6_32ch10hz1024res`
- 32 channels, 10 Hz, horizontal resolution 1024
- range: 0.3–120 m
- 주 저장 좌표계: frame별 `robot/base_link`
- 보조 저장 좌표계: LiDAR sensor, world
- LiDAR는 D455 optical center 위치에서 world-up 기준 수평인 virtual mounting v1
- RGB overlay는 영상 FOV 안의 전체 OS1 return을 표시
- RGB-D/LiDAR residual은 유효 D455 depth 범위에서만 계산
- Conda/Mamba는 사용하지 않음

## 파일 책임

- `05_render_lidar_isaac.py`: RTX scan 생성 및 NPZ/JSON 저장
- `validate_lidar_dataset.py`: 생성기와 독립적인 16개 schema·좌표·mesh·calibration gate
- `generate_lidar_report.py`: 단일 frame self-contained HTML
- `generate_lidar_multiframe_report.py`: 여러 frame의 그림·수치 추세 통합 HTML

## 재실행

저장소 root에서 Isaac Sim 포함 Python으로 실행한다.

```bash
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh
LIDAR=scripts/dataset_converters/gs_vlnpe/apply_real/lidar
```

### 1. 한 frame RTX scan 생성

04를 먼저 실행해 해당 frame의 RGB/depth/extrinsic이 존재해야 한다.

```bash
PYTHONUNBUFFERED=1 $ISAAC_PY $LIDAR/05_render_lidar_isaac.py \
  --smoke_test --episode 0 --frame 0 --warmup_frames 30
```

GPU 0 단일 실행으로 고정되어 있다. 4-GPU 자동 경로에서 실제 semaphore timeout이 발생했기
때문이다. `warmup_frames=30` 안에 32개 emitter와 약 360도 azimuth를 모두 가진 완전 scan이
나오지 않으면 저장하지 않고 실패한다.

### 2. 독립 검증

```bash
$ISAAC_PY $LIDAR/validate_lidar_dataset.py
```

검증기는 NPZ를 만든 코드의 판정을 신뢰하지 않고 dtype/shape/range, OS1 channel/azimuth,
sensor→robot→world 변환, 원본 USDZ mesh 정합, RGB-D projection residual을 다시 계산한다.

### 3. 단일 frame HTML

```bash
$ISAAC_PY $LIDAR/generate_lidar_report.py
```

`report.html`은 실행 산출물이므로 Git에 올리지 않는다.

### 4. episode 0 시작·중간·끝 비교

각 frame을 생성하고 독립 검증한 뒤 실행한다.

```bash
PYTHONUNBUFFERED=1 $ISAAC_PY $LIDAR/05_render_lidar_isaac.py \
  --smoke_test --episode 0 --frame 0 --warmup_frames 30
$ISAAC_PY $LIDAR/validate_lidar_dataset.py \
  --npz scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/episode_000000_frame_0000.npz
PYTHONUNBUFFERED=1 $ISAAC_PY $LIDAR/05_render_lidar_isaac.py \
  --smoke_test --episode 0 --frame 142 --warmup_frames 30
$ISAAC_PY $LIDAR/validate_lidar_dataset.py \
  --npz scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/episode_000000_frame_0142.npz
PYTHONUNBUFFERED=1 $ISAAC_PY $LIDAR/05_render_lidar_isaac.py \
  --smoke_test --episode 0 --frame 284 --warmup_frames 30
$ISAAC_PY $LIDAR/validate_lidar_dataset.py \
  --npz scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/episode_000000_frame_0284.npz

$ISAAC_PY $LIDAR/generate_lidar_multiframe_report.py \
  --episode 0 --frames 0,142,284
```

각 NPZ에는 대응하는 `.validation.json`이 있어야 통합 report에 포함된다. 위 명령은 세 frame을
생성한 직후 각각 독립 검증하므로 마지막 통합 report 명령의 입력 조건을 충족한다.

## 현재 완료 범위

- 단일 OS1-32 frame: 16/16 gate PASS
- episode 0의 frame 0, 142, 284: 모두 독립 PASS
- 3-frame 통합 HTML: PASS
- episode 0 전체와 20 episodes/8,378 frames batch: 아직 미완료

따라서 현재 `05_render_lidar_isaac.py`는 `--smoke_test` 전용이다. 전체 dataset이 완성된 것처럼
해석하거나 `--smoke_test` 없이 실행하지 않는다.

## 상세 문서

센서 profile 근거, robot 좌표계, pose transformation, NPZ schema, 16개 gate 의미, 실제 수치,
RGB/depth/pointcloud overlay 해석 및 문제 해결 이력은 다음 단일 문서를 기준으로 한다.

```text
.claude/memory/codex/codex_noeun_rtx_lidar_complete_reproduction_guide.md
```
