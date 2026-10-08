# Codex LiDAR 조사 02 — 설치된 Isaac Sim API 기준 방향

작성일: 2026-08-14 (UTC)

## 설치 상태

로컬 Isaac Sim에 `isaacsim.sensors.rtx` extension이 설치되어 있고, 공식 구현·테스트·LiDAR profile이 함께 존재한다.

- 생성 command: `IsaacSensorCreateRtxLidar`
- 고수준 wrapper: `isaacsim.sensors.rtx.LidarRtx`
- 원시/확장 출력: `GenericModelOutput`, `RtxSensorMetadata`
- profile 위치: `exts/isaacsim.sensors.rtx/data/lidar_configs`

## 버전상 중요한 선택

설치된 구현은 Isaac Sim 5.0부터 Camera prim 기반 RTX LiDAR 생성을 deprecated로 표시하고 **OmniLidar prim** 사용을 요구한다. 따라서 과거 예제의 Camera prim 코드를 복사하지 않고, 현재 설치본의 `LidarRtx` + OmniLidar 경로를 사용해야 한다.

## 왜 바로 04에 코드를 붙이지 않는가

현재 04는 RGB-D 카메라 재생/검증 책임을 가진다. 여기에 센서 생성, 회전 누적, timestamp, point decode, 좌표계 변환, 저장 schema를 한 블록으로 넣으면 사용자가 지적한 “돌아가게만 만든 거친 코드”가 반복된다. 먼저 최소 smoke test로 다음을 확정한 뒤 가장 작은 통합 지점을 정한다.

1. 노은역 USDZ가 RTX LiDAR ray hit를 반환하는가.
2. MP3D VLN-PE USD가 동일 profile에서 반환하는가.
3. VLN-CE GLB를 Isaac Sim용 USD로 옮겼을 때 scale/축/충돌 geometry가 보존되는가.
4. 한 frame의 정의가 instantaneous firing인지 한 rotation 누적인지 확정한다.
5. sensor frame과 world frame 중 무엇을 canonical 저장값으로 할지 확정한다.

## 다음 실험의 합격 기준

- point 수가 0이 아니며 NaN/Inf 비율과 min/max range가 profile 범위 안이다.
- 고정 pose 반복 측정의 geometry가 재현된다.
- 알려진 벽/바닥까지의 range를 mesh raycast 또는 camera depth와 독립 교차검증한다.
- 동일 profile과 저장 schema를 세 scene source에 적용한다.
- RGB/depth/pose/LiDAR의 frame index와 simulation time이 일치한다.
