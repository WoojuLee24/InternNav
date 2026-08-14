# Codex LiDAR 조사 01 — 저장소 내부 증거

작성일: 2026-08-14 (UTC)

## 이번 단계의 질문

InternNav에서 `pointcloud`라고 부르는 값이 실제 회전식/RTX LiDAR 측정인지, 카메라 depth에서 만든 점군인지 코드로 구분한다.

## 확인한 코드 경로

- `internnav/configs/evaluator/vln_default_config.py`
- `internnav/env/utils/internutopia_extension/controllers/h1_vln_move_by_speed_controller.py`
- `internnav/env/utils/internutopia_extension/tasks/vln_eval_task.py`
- `internnav/env/utils/internutopia_extension/sensors/vln_camera.py`
- `tests/function_test/test_evaluator.py`

## 확인 결과

### 1. VLN-PE의 주 관측 센서는 RGB/depth 카메라다

평가 설정은 H1의 `torso_link/h1_pano_camera_0`을 `VLNCamera`로 등록한다. `VLNCamera.get_data()`는 Isaac Sim 카메라에서 `rgba`와 `get_distance_to_image_plane()` depth를 가져온다.

### 2. `tp_pointcloud`는 LiDAR API가 아니다

H1 설정에 다음 센서가 추가된다.

```text
RepCameraCfg(
    name='tp_pointcloud',
    prim_path='logo_link/Camera_pointcloud',
    rgba=False,
    pointcloud=True,
    resolution=[64, 64],
)
```

따라서 이 값은 카메라 투영 모델과 64×64 raster를 기반으로 Replicator가 생성하는 포인트클라우드다. RTX LiDAR prim, LiDAR profile, 수직 채널 수, 회전 주파수, azimuth/elevation firing pattern을 설정하는 코드는 현재 InternNav 평가 경로에서 발견되지 않았다.

### 3. 이 점군의 사용 목적은 학습 관측 저장이 아니라 H1 보행 지형 샘플링이다

`h1_vln_move_by_speed_controller.py`는 `tp_pointcloud.get_data()['pointcloud']`를 받아 동적 height sample을 갱신한다. 즉 locomotion controller가 발 주변 높이를 추정하는 내부 보조 센서다. VLN 에이전트에 전달되는 관측은 `vln_eval_task.py`의 RGB, depth, camera pose 중심이다.

### 4. `h1_vln_pointcloud.usd`라는 파일명은 센서 종류의 증거가 아니다

테스트는 이 USD를 로봇 asset으로 지정하지만, 실제 센서 종류와 출력은 Python의 `RepCameraCfg`가 결정한다. 파일명만 보고 LiDAR가 이미 있다고 판단하면 안 된다.

## 현재 판정

- **VLN-PE에서 점군 취득 가능:** 예. 현재 구현은 카메라 기반 pointcloud다.
- **VLN-PE에서 Isaac Sim LiDAR 취득 가능:** scene mesh가 Isaac Sim USD로 로드되므로 기술적으로 가능성이 높지만, 현재 파이프라인에 구현되어 있지는 않다. 설치된 Isaac Sim LiDAR API와 asset 충돌/재질 조건을 별도 실험해야 확정한다.
- **노은역 04 출력:** 현재 RGB, depth, intrinsic/extrinsic을 저장한다. depth 역투영 점군은 만들 수 있지만 그것도 LiDAR 측정과 동일하지 않다.

## 다음 단계

1. VLN-CE/Habitat가 제공하는 센서와 episode 포맷을 코드로 확인한다.
2. 실제 VLN-N1/노은역 산출물의 dtype, shape, 단위, 좌표계, 파일 대응 개수를 자동 검사한다.
3. 설치된 Isaac Sim의 LiDAR API로 최소 scene smoke test를 설계한다.
