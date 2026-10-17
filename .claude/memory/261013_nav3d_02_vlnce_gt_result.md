# nav3d 02 — vln_ce GT 직접 비교 결과 (2026-10-13)

스크립트 `scripts/dataset_converters/nav3d/02_vlnce_gt_compare.py`, 리포트 `nav3d/logs/02_vlnce_gt_compare/report.html`.
61씬 10,684 ep (평지 9,092 · 층 이동 1,592), 매칭 실패 0. `--recompute` 포함, 4 프로세스 병렬.

## 결과
- A 좌표: navmesh 에서 10 cm 넘게 떨어진 GT 점이 없는 ep 10,668 (층 이동 1,588/1,592). 높이 차 p99 0.0002 m.
- A2 시작점 오차 0. B `find_path` 거리 = `info.geodesic_distance` 1 cm 안 93.6 % (중앙 0, p99 0.48 m, 씬 몇 개에 몰림).
- C reference_path 추종기 ↔ GT 위치: ep 중앙의 중앙 0.074 m (평지 0.088, 층 이동 0.026). action 완전 일치 24 ep 뿐 —
  GT 는 R F R F 지그재그, GreedyGeodesicFollower 는 회전을 몰아서. 도달 반경 0.1/0.5/1.0 모두 일치 0 → 생성기 구현 차이.
- C2 경유점끼리 find_path 로 이은 경로 ↔ GT: 위치 오차 중앙 0.083 m (평지 0.092 · 층 이동 0.047), 같은 루트 87.2 % (평지 85.1 · 층 이동 99.6).
  → 경유점을 주면 우리 계획도 GT 를 재현. TODO(사용자): start→goal 에 중요 경유점을 찍는 경로 생성.
- C 이상치 175 (1.6 %): 86 % route split(경유점 사이 장애물 반대편), 14 % GT 가 경유점 1 m+ 벗어남(데이터 불일치).
- D start→goal `find_path` 같은 루트 73.3 % (평지 71.7 · 층 이동 82.2), 길이비 0.92, 높이 범위 일치 10,677.
  → 층 이동도 find_path 하나로 계단을 지나 GT 와 같은 길. 2D occupancy 같은 루트 73.6 % 와 비슷.
- E glb 재빌드(기본 설정) 면적비 0.997~1.010, 같은 루트 7,830 vs 공식 7,827 → 우리 빌더 = 공식본.

## 함정
- 집계에서 `x or 9` 가 0.0 을 None 처럼 처리 → 높이 범위 일치 41 % 로 오보. 명시적 None 검사로 고침.
- jh4fc5c5qoQ ep85: GT 가 raw episode 와 안 맞음(추종기와 5 m, 끝 15 m) — 유일한 A 이상치.
- 삼각형 포함 판정만으로 "navmesh 위" 를 재면 경계 위 점이 빠짐 → snap 거리 0.10 m 로 판정.

## 노은역 (같은 날)
- 대합실 남쪽 y=-15.5 유리 칸막이(사용자 확인) → `SCENES['noeun_station']['blockers']` 상자(x −4.3~0.8, 바닥~0.95 m),
  문 x 0.8~1.95 유지. 여유 200/200, 경로가 x=1.17 문을 지남.
- 3D html 색 = 층(파랑 S0 · 주황 S1 · 초록 S2), 빨강 = 계단. top-down 그림은 색 = 섬(뜻이 다름) — 범례 추가.

## 다음
03 경로 다듬기(벽 밀어내기·스무딩, z = navmesh 표면) → 04 판정·paths json → 05 렌더.
