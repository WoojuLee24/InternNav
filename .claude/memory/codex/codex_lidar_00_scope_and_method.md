# Codex LiDAR 00 — 조사 범위와 판정 방법

## 요청 해석

이 조사의 중심은 “원본 VLN 파일에 LiDAR가 이미 있는가”만이 아니다. VLN-CE, VLN-PE, 노은역 씬을 각각 시뮬레이터에 로드하고 Isaac Sim의 동일 LiDAR profile로 관측 GT를 합성할 수 있는지, 그리고 동일 schema로 저장할 수 있는지를 검증한다.

## 반드시 분리할 개념

1. 원본 센서 관측: 데이터셋에 이미 프레임별 LiDAR가 저장되어 있음
2. 정적 scene pointcloud: 전체 씬 map/obstacle GT이며 프레임 센서 관측이 아님
3. depth 파생 pointcloud: 카메라 depth를 역투영한 것으로 LiDAR가 아님
4. Isaac Sim LiDAR 합성: 센서 ray pattern, range, pose와 timestamp를 정의해 새로 생성

이 네 가지를 섞지 않는다.

## 세 환경별 질문

- VLN-CE: Habitat/MP3D 자산을 Isaac Sim에서 사용할 수 있는가, 또는 원래 Habitat sensor로만 합성해야 하는가
- VLN-PE: InternUtopia/Isaac 씬과 H1 robot에 기존 LiDAR 또는 pointcloud sensor prim/config가 있는가
- 노은역: 현재 USDZ collision/render prim에서 동일 Isaac LiDAR ray가 정상 hit하는가

## 호환성 판정 항목

- 센서 종류: RTX LiDAR / PhysX raycast / camera-depth 파생
- 출력 표현: point XYZ, XYZI, range image, 2D LaserScan
- dtype/shape, 단위, invalid 표현
- sensor frame과 world/base frame 변환
- pose convention 및 mount extrinsic
- frame/timestamp 동기화
- episode/frame naming과 serialization
- train/eval 입력 경로의 실제 사용 여부

## 진행 원칙

- 먼저 읽기 전용으로 코드, config, USD prim, 실제 데이터 샘플을 조사한다.
- “가능할 것”과 “실제 실행 검증됨”을 구분한다.
- 구현 전 공통 schema와 sensor profile을 확정한다.
- Conda/mamba는 사용하지 않는다.
- 이후 단계별 분석과 실행 결과는 별도 `codex_lidar_*.md`로 기록한다.
