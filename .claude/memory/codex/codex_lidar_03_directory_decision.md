# Codex LiDAR 조사 03 — 작업 디렉터리 분리 결정

작성일: 2026-08-14 (UTC)

## 결정

새 LiDAR 작업은 다음 디렉터리에서 진행한다.

`scripts/dataset_converters/gs_vlnpe/lidar/`

## 이유

1. 기존 00~04는 원본 GS→VLN-PE RGB-D 파이프라인의 재현 경로이므로 안정된 상태로 보존한다.
2. LiDAR는 원본 저장소에 없던 센서와 schema를 추가하는 기능이므로 기존 파일에 큰 코드 블록을 삽입하면 변경 책임이 불명확해진다.
3. LiDAR smoke test, profile, 저장, 검증을 한 디렉터리에 두면 현재 복잡해진 상위 스크립트 목록을 더 늘리지 않는다.
4. 04의 pose와 결과를 입력으로 읽기만 하므로 기존 20개 노은역 episode를 손상시키지 않고 반복 실험할 수 있다.

## 파일 구성 원칙

- 실행 진입점은 `05_render_lidar_isaac.py` 하나로 유지한다.
- 센서 profile과 출력 schema가 커질 때만 `lidar_profiles.py`로 분리한다.
- 데이터 검증은 생성 코드와 독립적인 `validate_lidar_dataset.py`에서 수행한다.
- 임시 파일, `.bak`, 버전 번호가 붙은 복제 스크립트는 만들지 않는다.
- 분석 과정과 실행 결과는 `.claude/memory/codex_lidar_*.md`에 기록한다.

## 기준 노은역 입력

`scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex/obs/noeun_station_mid_random_isaac_d455_nominal`

이 경로의 20개 episode와 총 8,378개 RGB/depth/pose frame을 LiDAR frame 정렬 기준으로 사용한다.
