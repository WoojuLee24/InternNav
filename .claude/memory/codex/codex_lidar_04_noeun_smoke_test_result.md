# Codex LiDAR 조사 04 — 노은역 RTX LiDAR smoke test 결과

작성일: 2026-08-14 (UTC)

## 대상

- scene: `data/noeun_station/noeun_station_collision.usdz`
- pose: 노은역 중간층 04 Isaac D455 결과의 `episode_000000/frame_0000`
- API: Isaac Sim 5.1 `LidarRtx` / OmniLidar
- 임시 profile: `Example_Rotary`
- 출력 좌표계: world frame
- 임시 mount: 04 카메라 위치 + 카메라 yaw, 하향 pitch는 제거해 수평 배치

`Example_Rotary`와 이 mount는 geometry/API 검증용이며 최종 실제 LiDAR 사양이 아니다.

## 구현 파일

- `scripts/dataset_converters/gs_vlnpe/lidar/05_render_lidar_isaac.py`
- `scripts/dataset_converters/gs_vlnpe/lidar/validate_lidar_dataset.py`

## 발견한 문제와 수정

1. 첫 실행에서 RTX 3090 네 장을 자동 사용해 render semaphore timeout이 발생했다. 센서 하나에는 불필요하므로 GPU 0 단일 실행으로 고정했다.
2. 단일 GPU 첫 실행은 return이 0개였다. 노은역 collision capture mesh가 RTX sensor에서 숨겨져 있었고, RGB-D 04에서 검증한 `expose_collision_meshes_for_rendering()`을 재사용해 runtime mesh 1개를 노출하자 정상화됐다.
3. `SimulationApp.close()`가 예외 출력을 가릴 수 있어 종료 전에 traceback을 명시적으로 출력하도록 했다.

## 생성 결과

- point/valid count: 44,784 / 44,784
- finite fraction: 1.0
- return 생성까지 warmup: 2 frames
- range min / median / p95 / max: 3.2765 / 6.8311 / 21.6469 / 27.0826 m

산출물은 기준 observation root 아래 `lidar_rtx/smoke_test/episode_000000_frame_0000.{npz,json,validation.json}`에 있다.

## 독립 geometry 검증

생성 코드와 분리한 validator가 USDZ 내부 원본 `mesh.ply`를 읽고 return의 1/20인 2,240점을 검사했다.

- 표면거리 median: **0.000735 m (0.735 mm)**
- 표면거리 p95: **0.004699 m (4.699 mm)**
- 표면거리 max: 0.013343 m
- gate: median ≤ 2 mm, p95 ≤ 10 mm
- 판정: **PASS**

shape, dtype, finite, 저장 range 재계산, valid point, mesh median/p95의 7개 gate가 모두 통과했다.

## 현재 해석

노은역 중간층 04 pose에서 Isaac Sim RTX LiDAR 취득은 가능하고, 점군은 실제 노은역 mesh 표면과 정합한다. 다만 전체 20 episode 전에 실제 LiDAR 모델/profile, 장착 높이·회전, 최대 거리, 한 frame의 scan 누적 정의를 확정해야 한다. 이 선택 없이 `Example_Rotary`로 전량 생성하면 기술적으로 실행돼도 목표 센서를 반영한 데이터셋이라고 할 수 없다.
