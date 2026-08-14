# Codex 노은역 Isaac Sim RTX LiDAR 완전 재현·검증 가이드

작성: Codex
작성일: 2026-08-14
대상: 노은역 중간층 04 Isaac D455 nominal 관측 20 episodes, 8,378 frames

## 1. 이 문서의 목적

기존 `Example_Rotary` smoke test 이후 실제 데이터셋 기준을 Ouster OS1-32로 바꾸고, 박사님 요청인 다음 두 항목을 전체 파이프라인 계약으로 고정한다.

1. 저장 점군의 주 좌표계를 scene world 원점이 아니라 frame별 `robot/base_link`로 한다.
2. `result.html`에서 같은 frame의 RGB, depth, point cloud 및 RGB 위 LiDAR projection overlay를 함께 보여 calibration과 pose transformation이 맞는지 확인한다.

이 문서는 초기 smoke-test 중심의 `codex_noeun_rtx_lidar_complete_reproduction_guide.md`와
그 이후 작성한 OS1-32 설계·실행 문서를 하나로 합친 최신 canonical 문서다. 수치나 실행법이
서로 다를 때에는 뒤의 OS1-32 16-gate 및 3-frame 실제 결과를 기준으로 한다.

### 1.1 LiDAR와 다른 point cloud의 구분

- 정적 scene point cloud는 전체 mesh/map의 점 표현이며 pose별 sensor observation이 아니다.
- RGB-D point cloud는 depth image를 camera intrinsic으로 역투영한 것으로 camera raster와
  FOV를 따른다.
- InternNav의 `tp_pointcloud`는 Replicator camera 기반 64×64 point cloud이며 회전식
  LiDAR가 아니다.
- 이번 데이터는 Isaac Sim `LidarRtx`와 공식 OS1 beam profile이 scene에 직접 ray를 발사해
  얻은 return이다.

따라서 파일명에 `pointcloud`가 있다는 이유만으로 LiDAR라 판단하지 않는다. sensor API,
profile, prim, annotator, channel 및 azimuth pattern을 함께 확인한다. 조사 당시 VLN-CE와
VLN-PE의 기존 frame별 RTX rotary LiDAR 저장은 확인되지 않았으며, 노은역 PASS를 두
데이터셋의 PASS로 확대 해석하지 않는다.

### 1.2 초기 `Example_Rotary` smoke-test의 역할

최초 smoke test에서는 노은역 episode 0 frame 0에서 44,784개 return, finite 100%, 원본
mesh 표면거리 median 0.735 mm, p95 4.699 mm로 7개 gate를 통과했다. 이 결과가 증명한 것은
USDZ geometry/API/RTX visibility가 작동한다는 사실이다. 실제 dataset용 sensor model,
robot 좌표계, channel, 완전 scan 및 RGB-D calibration 계약을 증명한 것은 아니므로 최종
OS1-32 데이터와 섞지 않는다.

### 1.3 실제로 해결한 실행 문제

첫 자동 실행은 RTX 3090 네 장의 multi-GPU 경로에서 copy-queue semaphore timeout이
발생했다. LiDAR 한 개에는 다중 GPU가 필요하지 않아 다음 설정으로 GPU 0 하나에 고정했다.

```python
SimulationApp({'headless': True, 'multi_gpu': False, 'active_gpu': 0})
```

그 뒤 오류 없이 return이 0개인 문제가 발생했다. 노은역 collision mesh가 geometry에는
존재하지만 RTX sensor에서 숨겨진 것이 원인이었다. 04에서 검증한
`usdz_scene_utils.expose_collision_meshes_for_rendering(...)`을 재사용해 runtime stage에서만
mesh를 노출했다. 원본 USDZ를 덮어쓰거나 점을 mesh에 사후 투영하지 않았으며, 독립 validator가
RTX return과 원본 `mesh.ply`의 표면 정합을 다시 확인한다.

### 1.4 코드 책임의 분리

05는 기존 04의 RGB/depth/extrinsic을 수정하지 않고 읽기만 한다. LiDAR 생성, 독립 검증,
HTML 시각화를 각각 다음 파일로 분리한다.

- `05_render_lidar_isaac.py`: RTX return 생성과 NPZ/JSON 저장
- `validate_lidar_dataset.py`: 생성기와 독립적인 schema·좌표·mesh·calibration gate
- `generate_lidar_report.py`: 단일 frame HTML
- `generate_lidar_multiframe_report.py`: 여러 검증 frame의 통합 HTML

## 2. 센서 profile 확정

Isaac Sim 5.1 설치본에 포함된 다음 variant를 사용한다.

- model: `OS1`
- variant: `OS1_REV6_32ch10hz1024res`
- 32 channels
- rotary 10 Hz
- horizontal resolution 1024
- minimum range 0.3 m
- maximum range 120 m
- range resolution 0.001 m
- nominal range accuracy 0.03 m
- 최대 2 returns 설정

`Example_Rotary` 결과는 RTX LiDAR와 노은역 mesh의 작동성 검증 자료로만 보존한다. 전체 생성 결과에는 섞지 않는다.

Isaac Sim 5.1에서는 variant 이름을 과거처럼 config 하나로 넘기는 방식이 deprecated이다. 전체 코드는 공식 경고가 안내하는 `config="OS1", variant="OS1_REV6_32ch10hz1024res"` 방식으로 asset을 만들고, asset 하위의 실제 `OmniLidar` prim을 찾아 연결한다.

## 3. 실제 ScanBuffer 계약 조사 결과

노은역 episode 0 frame 0에서 `IsaacCreateRTXLidarScanBuffer`와 `auxOutputType=FULL`을 실제 실행했다.

- 유효 return: 32,217
- finite: 100%
- 방위각: 약 -180도~+180도
- emitter ID: 정확히 0~31
- intensity: return 수와 같은 길이
- distance: return 수와 같은 길이
- timestamp: 16,108개로 return 수와 1:1이 아님
- beam ID: 이 OS1 variant/annotator 조합에서는 빈 배열

따라서 다음 잘못된 가정을 하지 않는다.

- timestamp 하나가 point 하나와 항상 대응한다고 가정하지 않는다.
- 비어 있는 beam ID를 억지로 생성하거나 emitter ID와 같은 값이라고 기록하지 않는다.
- 현재 확인되지 않은 timestamp 확장 규칙을 추측해 point별 timestamp로 복제하지 않는다.

현 단계에서는 annotator 원본 timestamp를 `tick_timestamps_ns` 성격으로 그대로 보존하고, 정확한 return 매핑을 공식 근거 또는 GenericModelOutput 조사로 확인한 뒤 schema version을 확정한다. emitter ID가 OS1-32의 0~31 channel 식별자로 정상 출력되므로 channel coverage에는 이를 사용한다.

## 4. buffer lifetime 문제와 해결

처음 FULL auxiliary 출력을 읽었을 때 emitter ID의 중간 연속 구간에 31보다 큰 비정상 정수가 나타났다. annotator가 소유한 배열을 `np.asarray`로 참조한 채 timeline 정지와 후속 처리를 수행한 것이 원인이었다.

모든 ScanBuffer 배열을 취득 직후 `np.array(..., copy=True)`로 독립 복사하도록 수정했다. 재실행 결과 emitter ID는 0~31만 남았고 비정상값은 0개였다.

이 사례 때문에 전체 생성에서도 annotator가 반환한 view를 frame 경계 밖에서 직접 사용하지 않는다.

## 5. 기존 pose의 정확한 의미

`episode_xxxxxx/extrinsic/frame_xxxx.npy`는 파일 이름과 달리 바로 OpenCV c2w로 사용하면 안 된다. 04 생성 코드는 다음 변환을 사용한다.

```text
camera_pose_c2w_opencv = action_to_c2w(saved_action_pose, "cam2world_gl")
```

저장 행렬은 OpenGL/USD camera convention이고 RGB/depth 투영은 OpenCV convention이다. 기존 smoke 코드는 위치만 사용했기 때문에 mesh return은 얻었지만 yaw 계산에서 이 축 변환이 빠져 있었다. OS1 전체 코드에서는 변환된 OpenCV c2w의 +Z forward를 XY 평면에 투영하여 robot yaw를 구한다.

## 6. Robot 좌표계 정의

frame별 `robot/base_link`를 다음처럼 정의한다.

- 원점: 카메라 XY의 수직 아래에 있는 해당 층 바닥점
- world 위치: `[camera_x, camera_y, floor_z]`
- +X: 경로 진행 방향
- +Y: 좌측
- +Z: 위
- 회전: world Z축에 대한 yaw만 사용

episode별 `floor_z`, `h_b`, `pitch_deg`는 추정하지 않고 원본 03 경로 파일인 `paths/noeun_station_mid_random.json`에서 읽는다. 노은역 중간층의 `floor_z`는 약 -0.05 m이며 episode마다 `h_b`와 pitch가 다르다.

v1 LiDAR mounting은 다음과 같다.

- LiDAR 위치는 D455 optical center와 동일한 world 위치
- LiDAR는 카메라의 downward pitch를 따르지 않고 수평
- LiDAR yaw는 robot 진행 방향과 일치
- 따라서 `T_robot_lidar` translation은 기본적으로 `[0, 0, h_b]`

이는 실제 OS1이 D455와 같은 위치에 장착된 물리 calibration이라는 주장이 아니다. 실제 로봇 장착 치수를 아직 받지 않은 상태에서 기존 RGB-D pose와 합성 LiDAR를 재현성 있게 결합하기 위한 **virtual mounting v1**이다. 실제 장착 치수가 정해지면 `T_robot_lidar`를 새 calibration version으로 교체한다.

## 7. 좌표 변환과 저장 필드

다음 행렬을 모두 저장한다.

- `T_world_robot`: robot 점을 world로 변환
- `T_robot_lidar`: LiDAR 점을 robot으로 변환
- `T_world_lidar = T_world_robot @ T_robot_lidar`
- `camera_pose_c2w`: OpenCV camera 점을 world로 변환
- `source_camera_action_pose_gl`: 변환 전 원본 04 행렬

점군은 다음 세 좌표를 저장한다.

- `points_lidar_m`: ScanBuffer 원시 sensor 점
- `points_robot_m`: 학습과 사용의 주 좌표계
- `points_world`: mesh 정합 및 scene 전체 시각화용 보조 좌표계

실측 수치 검증:

- `||points_lidar_m||`와 annotator distance의 최대 차이: 약 3.8e-6 m
- `points_robot_m`을 `T_world_robot`으로 되돌린 값과 `points_world` 최대 차이: 약 2.3e-6 m
- episode 0에서 `points_robot_m - points_lidar_m`의 z 중앙값: 1.0462022 m
- 원본 episode 0의 `h_b`: 1.046202109 m

즉, robot frame 정의와 LiDAR 장착 높이가 수치상 일치한다.

## 8. RGB·Depth·LiDAR overlay 변환

같은 frame의 LiDAR world 점을 카메라 영상으로 투영한다.

```text
X_world = T_world_robot @ X_robot
X_camera = inverse(camera_pose_c2w) @ X_world
u = fx * X_camera.x / X_camera.z + cx
v = fy * X_camera.y / X_camera.z + cy
```

`camera_pose_c2w`는 반드시 OpenCV 축으로 변환된 행렬을 사용한다. D455 nominal intrinsic은 episode의 `intrinsic.npy`를 읽고, 이미지 크기와 K의 principal point가 맞는지도 검사한다.

projection 대상은 다음 조건을 만족해야 한다.

- camera Z가 양수
- u, v가 영상 범위 안
- finite 좌표
- LiDAR 유효 거리 범위 안

한 pixel에 여러 LiDAR 점이 들어오면 camera에 가장 가까운 Z를 사용해 z-buffer를 만든다. overlay 색상은 LiDAR camera-Z 또는 range로 표시한다.

## 9. Calibration의 수치 검증

눈으로만 정렬을 판단하지 않는다. 투영된 LiDAR 점의 camera-Z와 같은 pixel의 저장 depth를 비교한다.

- `depth_residual_m = lidar_camera_z - rgbd_depth_m`
- 중앙 절대 오차
- p95 절대 오차
- 허용 오차 이내 비율
- 영상 내부 투영 비율
- 유효 depth와 겹치는 비율

LiDAR와 RGB-D의 광선 샘플링이 다르고 JPG/RGB 및 depth 해상도가 480x270이므로 정확히 0이 될 필요는 없다. 또한 경계·가림 영역은 큰 잔차를 만들 수 있다. 따라서 중앙값/p95와 함께 residual image 및 overlay를 같이 보고 threshold를 smoke 실측으로 정한다.

## 10. result.html 구성

각 검증 단계의 HTML에는 최소한 다음을 넣는다.

1. 원본 RGB
2. metric depth color map
3. robot-frame LiDAR top view
4. world-frame LiDAR/mesh 정합
5. RGB 위 LiDAR projection overlay
6. depth와 LiDAR camera-Z residual 시각화
7. intensity 분포
8. azimuth coverage
9. emitter/channel 0~31 coverage
10. 좌표 변환 chain과 수치 오차 표
11. RGB/depth/extrinsic/LiDAR frame key 대응표
12. PASS/FAIL 및 실패 이유

대표 frame만 예쁘게 보이는 오류를 피하기 위해 episode 시작/중간/끝과 방향 전환, 근거리 구조, 장거리 개방 공간을 포함해 표본을 뽑는다.

## 11. 전체 실행 순서

1. OS1 한 pose 계약 smoke: 완료
2. robot 좌표 변환 수치 검사: 완료
3. timestamp/return 매핑의 정확한 의미 확정: 진행 중
4. RGB-depth-LiDAR overlay와 calibration 수치 검사 구현
5. episode 0의 3 frames 생성 및 HTML
6. episode 0 전체 생성 및 HTML
7. 공간적으로 떨어진 3 episodes 생성 및 HTML
8. 20 episodes, 8,378 frames 전체 생성
9. 전체 frame key/schema/좌표/mesh/calibration 통계 검증
10. 최종 `validation.json`, `report.html`, MD 결과 기록

전체 생성은 profile과 schema가 다른 결과를 섞지 않도록 `lidar_rtx/os1_rev6_32ch_10hz_1024_v1/` 아래에 둔다. 기존 `lidar_rtx/smoke_test/`는 이전 검사 이력으로 유지한다.

## 12. 현재 주의점

- ScanBuffer timestamp 수가 return 수의 절반이므로 아직 point별 timestamp라고 부르면 안 된다.
- OS1 beam ID는 비어 있고 emitter ID가 0~31로 정상이다.
- 한 render tick을 완전 scan으로 간주하면 안 된다. 방위각 coverage와 channel coverage를 완료 gate로 사용한다.
- Isaac Sim의 MotionBVH 경고는 움직이는 센서의 motion effect가 부정확하다는 뜻이다. v1은 pose를 고정한 stop-and-scan이므로 이동 중 motion effect를 의도하지 않는다.
- 실제 OS1-D455 물리 extrinsic이 없으므로 현재 overlay는 virtual mounting calibration 검증이다. 이를 실제 하드웨어 calibration이라고 표현하지 않는다.

## 13. 2026-08-14 구현·실행 결과 업데이트

OS1 asset은 루트 Xform과 내부 `/sensor` OmniLidar로 구성된다. 내부 sensor만 옮기거나 이미 WORLD인 ScanBuffer data를 sensor 점으로 간주해 다시 변환하면 다음과 같이 실패했다.

- 잘못 이중 변환한 mesh 거리 median: 약 0.778 m
- 잘못 이중 변환한 RGB-D/LiDAR 잔차 median: 약 2.74 m

transform 후보를 동일 raw point에 적용해 mesh로 판별했다.

- ScanBuffer raw WORLD data: mesh median 약 1.305 mm, p95 약 8.686 mm
- raw data에 world transform을 다시 적용: mesh median 약 0.778 m

따라서 최종 코드는 sensor prim의 `outputFrameOfReference`를 명시적으로 `SENSOR`로 설정하고 raw sensor point에 `T_robot_lidar`, `T_world_robot`을 정확히 한 번 적용한다. 문자열 설정이나 변수 이름을 믿지 않고 mesh 거리로 좌표 의미를 판별했다.

annotator buffer는 return을 발견한 render loop 안에서 즉시 deep copy한다. timeline 정지 뒤 복사하면 emitter ID 일부가 큰 비정상 정수로 변하는 현상이 재현됐기 때문이다.

최종 episode 0 frame 0 독립 검증 결과는 다음과 같다.

- status: PASS
- gate: 16/16 PASS
- OS1 returns: 32,198
- finite: 100%
- emitter/channel: 정확히 0~31
- azimuth span: 359.998352도
- range 재계산 최대 오차: 수 µm 수준
- transform chain error: 0
- lidar→robot 최대 오차: 1.19e-7 m
- robot→world 최대 오차: 2.24e-6 m
- mesh 표면거리 median: 1.321 mm
- mesh 표면거리 p95: 8.887 mm
- RGB-D와 겹친 projected returns: 5,138
- RGB-D/LiDAR 절대 depth 잔차 median: 2.507 cm
- RGB-D/LiDAR 절대 depth 잔차 p95: 15.355 cm

생성된 HTML:

`scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/report.html`

이 HTML에는 RGB, metric depth, robot-frame/world-frame point cloud, RGB projection overlay, calibration residual, intensity, 32채널 coverage 및 모든 PASS/FAIL gate가 포함된다.

다음 구현 단계는 이 단일-frame 함수를 한 번의 SimulationApp에서 pose만 바꾸며 반복하도록 구조화하고, accumulator/buffer reset과 완전 scan gate를 frame마다 적용하는 것이다. 그 뒤 3 frames → episode 0 전체 → 분산 3 episodes → 20 episodes 8,378 frames 순서로 실행한다.

## 14. 10 m 표시 제한과 LiDAR 중심 공백 분석

사용자가 최초 HTML에서 point cloud도 10 m로 잘린 것처럼 보인다고 지적했다. 실제 NPZ를 다시 검사한 결과 LiDAR 생성에는 10 m cutoff가 없었다.

- OS1 profile 최대 거리: 120 m
- smoke frame 실제 최대 return: 약 42.01 m
- 전체 32,198점 중 10 m 이하: 25,375점
- 10 m 초과: 6,823점, 약 21.2%
- 20 m 초과: 1,127점

10 m는 D455 nominal depth 저장 cutoff이며 LiDAR cutoff가 아니다. 기존 HTML의 RGB overlay 코드가 calibration 비교와 표시를 모두 `camera_z <= 10 m`로 묶어 멀리 있는 LiDAR 점을 숨긴 것이 원인이었다.

HTML을 다음처럼 수정했다.

- RGB projection overlay: 영상 FOV 안에 투영되는 모든 OS1 return 표시
- calibration residual: 유효 D455 depth가 존재하는 10 m 범위만 비교
- robot/world top view: 처음부터 전체 LiDAR return 표시
- overlay color upper bound: 해당 frame의 range p99를 사용하고 값도 colorbar에 표시

`0.001 m/raw` uint16에서 사용할 수 있는 일반적인 최대 유효값은 raw 65534를 기준으로 **65.534 m**이다. raw 65535는 invalid/sentinel로 남기는 것이 안전하다. 그러나 이 수는 D455의 실제 유효 측정거리 사양이 아니라 저장 encoding의 수치적 상한이다. 04 depth를 65.534 m로 바꾸려면 `depth_max_m`뿐 아니라 `render_far_m`도 65.534 m보다 크게 바꾸고 20 episodes RGB-D를 다시 렌더해야 한다. 현재 LiDAR 전체 시각화에는 이 변경이 필요하지 않으므로 수행하지 않았다.

OS1 점군의 robot 원점 주변이 넓게 비어 보이는 것은 거리 cutoff보다 수직 FOV와 장착 높이의 영향이다.

- OS1-32 수직 각도: -22.1도~+22.1도
- episode 0 장착 높이: 약 1.0462 m
- 가장 아래쪽 -22.1도 ray가 평평한 바닥과 만나는 수평거리: 대략 `1.0462 / tan(22.1°) ≈ 2.58 m`
- smoke frame 실제 최소 return: 약 2.70 m

즉, 가까운 평평한 바닥이 원형으로 비어 보이는 것은 OS1-32의 물리적인 수직 시야와 장착 높이에 부합한다. 0.3 m 최소 거리 때문에 생긴 2.7 m blind radius가 아니다. 가까운 벽·사람·기둥처럼 ray 높이를 가로지르는 물체는 이 반경 안에서도 감지될 수 있다.

이 공백을 줄이려면 OS1-32 profile을 임의 왜곡하기보다 실제 장착 높이를 낮추거나 센서를 아래로 기울이거나, 더 넓은 수직 FOV 센서를 별도 dataset version으로 선택해야 한다. 현재 OS1-32 고정 요구에서는 이 현상을 정상 센서 특성으로 보존한다.

## 15. 현재 검증된 LiDAR 실행 방법

### 15.1 실행 환경 원칙

- 저장소 루트 `/ws/src/InternNav`에서 실행한다.
- Conda 환경은 생성하거나 활성화하지 않는다.
- 일반 `python`이나 `python3` 대신 설치된 Isaac Sim 전용 Python launcher를 사용한다.
- RTX LiDAR 생성기는 내부적으로 GPU 0 한 장만 사용한다. 이 장비의 4-GPU 자동 경로에서 semaphore timeout이 실제 발생했기 때문에 `multi_gpu=False`, `active_gpu=0`으로 고정되어 있다.
- 아래 명령은 2026-08-14 현재 실제 실행해 PASS까지 확인한 명령이다.

먼저 저장소 루트로 이동한다.

```bash
cd /ws/src/InternNav
```

### 15.2 OS1-32 한 frame 생성

기본값은 노은역 중간층 episode 0, frame 0이다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/05_render_lidar_isaac.py \
  --smoke_test \
  --episode 0 \
  --frame 0 \
  --warmup_frames 30
```

기본 profile은 코드에 다음과 같이 고정되어 있다.

```text
model   = OS1
variant = OS1_REV6_32ch10hz1024res
```

다른 episode/frame을 한 개 확인하려면 숫자만 바꾼다. 예를 들어 episode 7, frame 120은 다음과 같다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/05_render_lidar_isaac.py \
  --smoke_test \
  --episode 7 \
  --frame 120 \
  --warmup_frames 30
```

입력 pose가 없으면 생성기는 즉시 `FileNotFoundError`로 중단한다. frame 번호를 추측하지 말고 해당 episode의 `extrinsic/frame_xxxx.npy` 존재 여부를 확인해야 한다.

생성 파일은 현재 smoke 경로에 저장된다.

```text
scripts/dataset_converters/gs_vlnpe/apply_real/obs/
└── noeun_station_mid_random_isaac_d455_nominal/
    └── lidar_rtx/
        └── smoke_test/
            ├── episode_000000_frame_0000.npz
            └── episode_000000_frame_0000.json
```

NPZ에는 최소한 다음 정보가 포함된다.

- `points_lidar_m`: OS1 sensor 좌표 점군
- `points_robot_m`: 주 데이터 좌표인 robot/base_link 점군
- `points_world`: mesh 검증용 world 점군
- `ranges_m`, `intensity`, `azimuth`, `elevation`, `emitter_id`
- `T_robot_lidar`, `T_world_robot`, `T_world_lidar`
- OpenCV `camera_pose_c2w`
- 원본 OpenGL/USD `source_camera_action_pose_gl`
- episode/frame 번호와 장착 높이

### 15.3 독립 검증 실행

생성 코드와 별개의 검증기로 NPZ를 검사한다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/validate_lidar_dataset.py
```

기본적으로 episode 0 frame 0 smoke NPZ를 읽는다. 특정 NPZ를 검사하려면 `--npz`를 명시한다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/validate_lidar_dataset.py \
  --npz scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/episode_000007_frame_0120.npz
```

검증 결과는 NPZ 옆의 `.validation.json`에 저장된다.

```text
episode_000000_frame_0000.validation.json
```

현재 gate는 다음을 포함한다.

- shape/dtype/finite
- 거리 재계산 일치
- `T_world_robot @ T_robot_lidar == T_world_lidar`
- LiDAR→robot 및 robot→world 점 변환 일치
- OS1 emitter 0~31 전체 coverage
- 약 360도 azimuth coverage
- source mesh 표면거리 median/p95
- RGB-D와 LiDAR projection overlap
- RGB-D/LiDAR depth 잔차 median/p95

프로그램 종료코드는 PASS이면 0, FAIL이면 1이다. JSON의 일부 수치가 좋아 보여도 최종 `status`가 `PASS`인지 반드시 확인한다.

### 15.4 HTML report 생성

독립 검증이 먼저 실행되어 `.validation.json`이 있어야 한다. 그 다음 HTML을 만든다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/generate_lidar_report.py
```

특정 NPZ에 대해서는 생성과 검증 때 사용한 동일 경로를 넘긴다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/generate_lidar_report.py \
  --npz scripts/dataset_converters/gs_vlnpe/apply_real/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/episode_000007_frame_0120.npz
```

기본 HTML 위치:

```text
scripts/dataset_converters/gs_vlnpe/apply_real/obs/
└── noeun_station_mid_random_isaac_d455_nominal/
    └── lidar_rtx/smoke_test/report.html
```

HTML에는 다음이 함께 들어간다.

- 같은 frame의 D455 RGB와 metric depth
- robot/base_link 점군
- world 점군 상면·측면
- 전체 거리 OS1 점을 RGB에 투영한 overlay
- 10 m 유효 depth 범위의 calibration residual
- mesh 표면거리 분포
- range와 intensity 분포
- OS1 32채널 coverage
- 모든 수치 gate의 PASS/FAIL 표

RGB overlay는 10 m 밖 LiDAR 점도 표시한다. 10 m 제한은 D455 depth와 수치 calibration을 비교할 때만 적용한다.

### 15.5 처음부터 report까지 실행하는 순서

각 명령이 성공한 것을 확인하면서 아래 순서로 실행한다.

```bash
cd /ws/src/InternNav

/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/05_render_lidar_isaac.py \
  --smoke_test --episode 0 --frame 0 --warmup_frames 30

/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/validate_lidar_dataset.py

/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/generate_lidar_report.py
```

### 15.6 현재 아직 실행하면 안 되는 명령

현재 `05_render_lidar_isaac.py`는 검증된 단일-frame `--smoke_test`만 허용한다. `--episodes 0-19` 또는 전체 8,378 frames 옵션은 아직 구현 중이다. 옵션 없이 실행하면 전체 생성을 시작하는 것이 아니라 의도적으로 오류를 발생시킨다.

따라서 다음과 같은 명령을 현재 존재한다고 가정해서 사용하면 안 된다.

```text
--episodes 0-19
--all
--resume
```

다중 frame 구현이 완료되면 이 문서에 실제 `--help`와 일치하는 명령, 재개 규칙, 출력 디렉터리, 단계별 검증 명령을 추가하고 이 경고를 제거한다.

## 16. 결과 수치의 정확한 의미와 해석

이 절은 터미널 JSON과 `report.html`에 표시되는 숫자가 무엇을 계산한 값인지 설명한다. 서로 다른 단위와 좌표계의 숫자를 한 종류의 정확도로 해석하면 안 된다.

### 16.1 `point_count`

한 scan에서 Isaac Sim ScanBuffer가 반환한 Cartesian point의 개수다.

```text
point_count = points_lidar_m.shape[0]
```

OS1-32는 수직 32채널과 수평 1024 resolution을 사용하지만 항상 정확히 `32 × 1024`점이 저장된다는 뜻은 아니다. geometry hit 여부, return 정책, invalid point 제거 및 scan 시작 위상에 따라 개수가 달라질 수 있다. 따라서 frame 간 point 수가 조금 다른 것은 정상이다.

현재 대표 frame: 약 32,227점.

### 16.2 `valid_point_count`와 `finite_fraction`

`valid_point_count`는 NaN/Inf가 없고 양의 거리를 갖는 점의 수다. `finite_fraction`은 전체 점 중 XYZ 세 좌표가 모두 finite인 비율이다.

```text
finite_fraction = finite XYZ point 수 / 전체 point 수
```

현재 대표 frame은 100%다. 100%가 아니면 point buffer 또는 sensor 출력에 문제가 있으므로 전체 생성에서 허용하지 않는다.

### 16.3 `range_m`

OS1 sensor 원점에서 각 return까지의 Euclidean 거리다.

```text
range = sqrt(x_lidar² + y_lidar² + z_lidar²)
```

- `min`: 가장 가까운 return
- `median`: return 거리 중앙값
- `p95`: 전체 return의 95%가 이 거리 이하
- `max`: 가장 먼 return

현재 대표 frame의 최대값은 약 42.03 m다. 이는 OS1 profile 최대 거리 120 m와 다르다. 120 m는 센서 설정 상한이고, 42.03 m는 해당 pose에서 실제 mesh에 맞아 돌아온 가장 먼 점이다.

### 16.4 `range_recomputation`

ScanBuffer가 제공한 distance와 `||points_lidar_m||`를 독립적으로 다시 계산해 비교한다. 좌표가 실제 SENSOR frame인지 검증하는 gate이기도 하다.

허용오차는 절대·상대 `1e-5` 수준이다. 이 gate가 실패하면 sensor/world 좌표를 혼동했거나 배열 alignment가 깨졌을 가능성이 있다.

### 16.5 `azimuth_span_deg`

저장된 return의 최대 방위각과 최소 방위각 차이다.

```text
azimuth_span = max(azimuth) - min(azimuth)
```

현재 대표 frame: 약 359.9986도. 359도 이상이어야 360도 회전 scan으로 인정한다. point 수가 많더라도 이 값이 작으면 부분 scan일 수 있으므로 저장하지 않는다.

이 값은 모든 각도에서 같은 수의 hit가 있다는 뜻은 아니다. 벽이나 mesh가 없는 방향에서는 return이 적을 수 있다.

### 16.6 `emitter_ids`와 OS1 32채널 coverage

OS1-32의 수직 발광 채널 번호다. 정상 scan은 고유값이 정확히 다음과 같아야 한다.

```text
[0, 1, 2, ..., 31]
```

32개 중 하나라도 빠지거나 31보다 큰 비정상값이 있으면 FAIL이다. 실제로 annotator buffer lifetime 문제로 큰 비정상 정수가 섞이는 현상을 발견했으며, 생성 단계에서 해당 candidate scan을 폐기하고 자동 재시도하도록 수정했다.

채널이 모두 존재한다고 해서 수직 공간이 연속적으로 채워지는 것은 아니다. OS1-32는 -22.1도~+22.1도 사이의 32개 고정 각도를 측정하므로 줄무늬 구조는 정상이다.

### 16.7 좌표변환 오차

세 가지 좌표계 변환을 독립적으로 재계산한다.

```text
T_world_lidar = T_world_robot @ T_robot_lidar
points_robot = T_robot_lidar @ points_lidar
points_world = T_world_robot @ points_robot
```

HTML/JSON의 값은 두 방식으로 계산한 좌표 사이의 XYZ 최대 절대 차이다.

- `chain`: 두 행렬 합성 방식의 최대 차이
- `lidar_to_robot`: 저장된 robot 점과 LiDAR 점 변환 결과의 최대 차이
- `robot_to_world`: 저장된 world 점과 robot 점 변환 결과의 최대 차이

현재 대표 frame:

- chain: 약 `5.96e-8`
- LiDAR→robot: 약 `1.19e-7 m`
- robot→world: 약 `2.28e-6 m`

이는 수치 연산·float32 저장에 따른 마이크로미터 수준 차이다. 센서 측정 정확도가 마이크로미터라는 뜻이 아니다. 단지 저장된 좌표변환이 서로 일관된다는 뜻이다.

### 16.8 `mount_position_error_m`

요청한 가상 장착 위치와 Isaac Sim 내부 OmniLidar의 실제 world 위치 차이다.

```text
mount_position_error = ||requested_position - actual_sensor_position||
```

현재 값은 0 m다. OS1 asset의 root와 내부 sensor prim을 혼동하면 이 값이 달라질 수 있으므로 전체 frame마다 검사한다.

### 16.9 mesh 표면거리 median/p95

`points_world`에서 일정 간격으로 표본을 뽑아 원본 노은역 collision mesh의 가장 가까운 표면까지 거리를 계산한다.

- median: 표본 오차의 중앙값
- p95: 표본의 95%가 이 값 이하

현재 대표 frame:

- median: 약 1.270 mm
- p95: 약 9.778 mm

현재 gate:

- median ≤ 2 mm
- p95 ≤ 10 mm

이 값은 RTX ray hit가 원본 mesh 표면에 놓이는지 확인한다. 좌표변환이 중복 적용됐던 실패 코드에서는 median이 약 0.778 m까지 증가했으므로, pose/좌표계 오류를 매우 민감하게 잡는다.

mesh 오차는 실제 OS1 하드웨어의 거리 정확도를 측정한 값이 아니다. 합성 ray와 합성 source mesh의 내부 정합을 측정한 값이다.

### 16.10 `projected_count`

LiDAR world 점을 해당 frame의 OpenCV camera 좌표로 변환한 뒤 RGB 영상 범위 안에 들어온 점의 수다.

조건:

- camera Z > 0
- u/v가 480×270 영상 내부
- 유효 LiDAR 점

OS1은 360도이지만 RGB 카메라는 전방 약 90도만 보므로 전체 LiDAR 점 중 일부만 projected된다. projected되지 않은 점이 LiDAR에서 인식되지 않았다는 뜻은 아니다.

### 16.11 `valid_depth_overlap_count`

RGB 영상 안으로 projected된 LiDAR 점 중 같은 pixel에 유효한 D455 depth가 존재하는 수다. 현재 D455 nominal depth 저장 cutoff가 10 m이므로 calibration 수치 비교도 이 유효 depth 범위에 한정한다.

현재 대표 frame: 약 5,172점.

RGB overlay 그림 자체는 10 m보다 먼 OS1 return도 표시한다. 10 m 제한은 depth residual 계산에만 적용한다.

### 16.12 RGB-D/LiDAR 절대 depth 잔차 median/p95

LiDAR 점을 카메라 영상에 투영한 뒤 LiDAR 점의 camera-Z와 같은 pixel의 D455 depth 차이를 계산한다.

```text
absolute residual = |lidar_camera_z - rgbd_depth|
```

현재 대표 frame:

- median: 약 2.47 cm
- p95: 약 14.28 cm

현재 gate:

- median ≤ 5 cm
- p95 ≤ 25 cm

p95가 median보다 큰 이유는 물체 경계, 얇은 구조, LiDAR와 카메라의 서로 다른 ray sampling, 480×270 pixel 양자화 및 가림 관계 때문이다. 이 값은 실제 OS1-D455 factory calibration 결과가 아니라 현재 virtual mounting과 합성 RGB-D/LiDAR의 정렬 정도다.

### 16.13 intensity

각 return의 RTX LiDAR intensity 값이다. 현재는 원본 float32를 변형하지 않고 저장한다. intensity histogram은 값이 모두 0이거나 비정상적으로 한 값에 몰리는 문제를 확인하는 진단 그림이다.

실제 장비의 calibrated reflectivity와 동일하다고 가정하지 않는다. 재질, RTX sensor model 및 Isaac Sim 설정의 영향을 받는 합성 intensity다.

### 16.14 timestamp

현재 ScanBuffer timestamp 배열은 point 수와 1:1이 아니다. 대표 scan에서는 약 32,227 points에 약 16,113 timestamps가 나온다. 따라서 아직 `point_timestamps_ns`라고 부르거나 임의로 두 번 복제하지 않는다.

현재는 annotator 원본 배열을 보존하고 단조 증가 여부만 확인한다. 정확한 tick/echo/return 매핑을 확정하기 전에는 학습 입력의 point별 시간으로 사용하지 않는다.

## 17. 다중-frame `report.html` 설계

현재 `lidar_rtx/smoke_test/report.html`은 episode 0 frame 0 한 개를 확인하는 단일-frame report다. 다중 frame 생성 후에는 단일 HTML을 덮어쓰지 않고 단계별 aggregate report를 별도로 만든다.

예정 구조:

```text
lidar_rtx/os1_rev6_32ch_10hz_1024_v1/
├── episode_000000/
│   ├── lidar/frame_0000.npz
│   ├── lidar/...
│   ├── validation.json
│   └── report.html
├── validation.json
└── report.html
```

### 17.1 3-frame 단계 report

episode 0의 시작·중간·끝 frame을 한 페이지에서 비교한다.

각 frame 카드에 다음을 넣는다.

- frame 번호와 원본 pose 경로
- RGB, depth, RGB+LiDAR overlay
- robot-frame 3D/top view
- world-frame point cloud
- range/intensity/channel plot
- mesh median/p95
- RGB-D/LiDAR median/p95
- 360도·32채널·좌표변환 PASS/FAIL

페이지 상단에는 3개 frame의 전체 PASS 여부와 min/median/max 통계를 둔다. frame 하나라도 필수 gate에 실패하면 단계 전체를 FAIL로 표시한다.

### 17.2 episode 전체 report

모든 frame의 큰 이미지를 한 HTML에 전부 넣으면 파일이 지나치게 커지므로 다음처럼 구성한다.

- 모든 frame: 수치 행과 PASS/FAIL 제공
- 대표 frame: 시작·25%·50%·75%·끝 및 최악 수치 frame의 상세 그림 제공
- frame별 point count/range/mesh/calibration을 시계열 plot으로 제공
- 실패 frame을 표 상단에 별도로 모음

### 17.3 20-episode 최종 report

최종 root `report.html`에는 다음을 넣는다.

- 20 episodes/8,378 frames 완성도
- RGB/depth/extrinsic/LiDAR key 일치율
- episode별 frame 수와 PASS/FAIL
- 전체 point count 및 거리 분포
- mesh 정합 전체 분포
- RGB-D/LiDAR calibration 전체 분포
- 각 episode 대표 overlay
- 최악 frame 상세 분석과 경로 링크
- 중단·재개 및 skip된 frame 통계

즉, 다중 frame도 HTML에서 확인한다. 단일 frame 그림을 단순 반복하는 것이 아니라 전체 통계와 이상 frame을 찾을 수 있는 검증 report로 만든다.

## 18. 실제 3-frame 실행 결과

2026-08-14에 episode 0의 시작·중간·끝을 선택해 실제 RTX LiDAR 생성, 독립 검증, 통합 HTML 생성을 완료했다.

선택 기준은 총 285 frames의 index 범위 `0..284`에서 다음과 같다.

- 시작: frame 0
- 정중앙: frame 142
- 끝: frame 284

연속된 인접 frame만 검사하면 한 장소의 우연한 성공만 확인하게 된다. 시작·중간·끝은 경로상 서로 떨어진 pose에서 센서, 좌표변환, mesh 정합 및 RGB-D 투영이 유지되는지를 빠르게 검사하기 위한 최소 대표 집합이다.

### 18.1 생성된 통합 report

```text
scripts/dataset_converters/gs_vlnpe/apply_real/obs/
└── noeun_station_mid_random_isaac_d455_nominal/
    └── lidar_rtx/smoke_test/
        ├── three_frame_report.html
        ├── three_frame_validation.json
        └── three_frame_report_assets/
            ├── three_frame_metrics.png
            ├── frame_0000_summary.png
            ├── frame_0142_summary.png
            └── frame_0284_summary.png
```

`three_frame_report.html`은 PNG를 base64 data URI로 포함한 약 3.0 MiB의 self-contained HTML이다. 따라서 HTML 하나만 열어도 모든 그림을 볼 수 있다. `three_frame_report_assets`는 원본 해상도 PNG를 별도로 확인하거나 이후 문서에서 재사용하기 위해 함께 보존한다.

### 18.2 report 재생성 명령

Conda는 사용하지 않는다. Isaac Sim에 포함된 Python으로 다음을 실행한다.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/dataset_converters/gs_vlnpe/apply_real/lidar/generate_lidar_multiframe_report.py \
  --episode 0 \
  --frames 0,142,284
```

이 명령은 LiDAR를 다시 ray-cast하지 않는다. 이미 생성된 세 NPZ와 각 `.validation.json`, 기존 04의 RGB/depth/intrinsic을 읽어 통합 report를 재구성한다. 입력 NPZ가 없거나 해당 독립 검증 JSON이 없으면 report를 만들지 않고 오류로 중단한다. 검증되지 않은 데이터를 그림만 그려 성공처럼 보이게 하지 않기 위한 조건이다.

다른 episode 또는 frame을 비교하려면 `--episode`와 쉼표로 구분한 `--frames`를 바꾼다. NPZ 이름은 다음 규칙을 따른다.

```text
episode_{episode:06d}_frame_{frame:04d}.npz
```

### 18.3 실제 측정값

| frame | 독립 검증 | returns | mesh median | mesh p95 | RGB-D/LiDAR median | RGB-D/LiDAR p95 |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | PASS | 32,227 | 1.270 mm | 9.778 mm | 2.467 cm | 14.281 cm |
| 142 | PASS | 31,659 | 1.138 mm | 7.938 mm | 1.059 cm | 9.660 cm |
| 284 | PASS | 32,317 | 1.360 mm | 7.700 mm | 1.025 cm | 6.850 cm |

세 frame 모두 각자의 16개 validation gate를 통과했고, 통합 판정도 `PASS`다.

### 18.4 통합 report 그림을 읽는 법

각 frame 카드는 네 개 패널로 구성된다.

1. `RGB + all in-FOV OS1 returns`
   - 360도 LiDAR 중 D455 RGB 시야 안으로 투영되는 모든 return을 표시한다.
   - 점 색은 OS1 원점에서 return까지의 거리다.
   - 10 m로 자른 점군이 아니다. 그 frame의 in-FOV LiDAR 점 전체를 사용한다.
   - RGB 구조와 점의 윤곽이 겹치면 camera/LiDAR pose transformation을 시각적으로 확인할 수 있다.
2. `D455 metric depth`
   - 기존 04에서 생성한 uint16 depth를 `depth_raw × 0.001 m`로 복원한 그림이다.
   - 현재 nominal profile의 검증 유효 범위는 10 m다.
   - 이 10 m는 RGB-D/LiDAR 수치 비교 범위이며 OS1의 최대 거리 120 m와 별개다.
3. `robot/base_link point cloud`
   - 저장 계약의 `points_robot_m`을 robot 좌표계에서 위에서 내려다본 XY top view로 표시한다.
   - 원점의 별은 robot/base_link 원점이다.
   - X는 전방, Y는 좌측이며, 줄무늬는 OS1-32의 32개 수직 beam과 1024개 수평 sampling에서 생기는 정상적인 scan pattern이다.
4. `Full OS1 range distribution`
   - RGB 시야나 D455 10 m 범위로 제한하지 않은 전체 유효 return 거리 histogram이다.
   - 점선은 거리 중앙값이다.

상단 `frame별 수치 추세` 그림은 다음을 비교한다.

- `return count`: frame마다 유효하게 돌아온 LiDAR 점 개수. 기하 구조와 가림에 따라 달라지므로 세 값이 완전히 같을 필요는 없다.
- `mesh alignment [mm]`: world 점과 원본 collision mesh 표면의 거리. median 2 mm, p95 10 mm gate 아래여야 한다.
- `RGB-D/LiDAR residual [cm]`: 동일 영상 pixel에서 LiDAR camera-Z와 D455 depth의 절대 차이. median 5 cm, p95 25 cm gate 아래여야 한다.

### 18.5 현재 단계의 정확한 의미와 한계

이번 결과로 확인한 것은 한 frame의 smoke test를 넘어 episode 0의 서로 떨어진 세 pose에서도 다음 계약이 유지된다는 점이다.

- OS1-32의 32개 channel과 약 360도 azimuth scan
- RTX return과 노은역 collision mesh의 millimeter 수준 정합
- sensor/world/robot 좌표변환 왕복 일치
- robot/base_link 기준 점군 저장
- RGB 투영과 D455 depth를 이용한 calibration residual gate 통과

하지만 이것은 아직 episode 0 전체 285 frames 또는 전체 20 episodes/8,378 frames 완료를 뜻하지 않는다. 현재 세 NPZ는 각각 독립적인 `--smoke_test` Isaac Sim 실행으로 생성됐다. 다음 구현 단계에서는 Isaac Sim을 한 번 띄운 상태로 pose를 순회하는 batch mode, 중단 후 재개, frame 원자 저장, episode aggregate validation을 추가한 뒤 같은 report 구조를 episode 전체와 20 episodes 전체로 확장한다.
