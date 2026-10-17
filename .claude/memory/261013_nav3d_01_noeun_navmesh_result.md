# nav3d 01 — 노은역 navmesh 생성·검증 결과 (2026-10-13)

새 폴더 `scripts/dataset_converters/nav3d/` (2D occupancy + 계단 연결은 다른 세션, 분리 진행).
README: `nav3d/README.md`. 리포트: `nav3d/logs/01_build_navmesh/noeun_station/report.html`.

## 명령
`/usr/bin/python scripts/dataset_converters/nav3d/01_build_navmesh.py --scene noeun_station` (habitat 컨테이너)
설정: agent_radius 0.25 · agent_height 0.875 (노은역 2D 와 같은 로봇) · max_climb 0.2 · max_slope 45 · cell_height 0.05.

## 결과
- navmesh 빌드 1.6 s, 2,083 m², 섬 20개. 좌표 V1 PASS(꼭짓점↔mesh 중앙 0.039 m, 틀린 규약 6.13 m).
- 층 3개: S0 B2 승강장 −5.6~−5.06 · S1 B1 대합실 −0.5~0.0 · S2 지상 +6.9~+8.2. 바닥이 끝에서 끝까지 ~0.5 m 기울어
  평면 6개 → 층 3개로 묶음(`--storey_gap 1.5`).
- 계단 3개 자동 검출: C0·C1 (B2↔B1, 승강장 2개) · C2 (B1↔지상 출구). stepness 0.136~0.163 m (같은 층 안 경사 ≤ 0.023).
  세 층이 한 섬(1,489 m²). 에스컬레이터·엘리베이터 연결부 없음.
- 층간 경로 200/200 성공, 경로 점 100 % navmesh 위(수직 투영). 승강장→지상 약 100 m.
- mesh 3D 여유 ≥ 0.20: 117/200. 미달 83개 전부 (−2.5, −15.5) — 천장에서 바닥 위 ~0.95 m 까지 내려온 구조물 아래 통과.
  투명 유리 칸막이면 실제로는 벽(3DGS 가 유리 못 살림). 현장 확인 필요.
- 2D 지도 대조(대합실): IoU 0.18, mesh 바닥 있는 곳만 0.85. 2D 만 통행 3,140 m² 중 3,105 m² 는 바닥 없음(벽 밖).

## 측정 함정 (고침)
- `build_navmesh_vertex_indices(i)` 섬별 이어 붙임 ≠ `(-1)` 전체 순서 → 라벨 어긋남(계단이 끊긴 것처럼 보였음).
  `navmesh_triangles_islands` 로 섬별로 뽑고 `get_island` 표본 검사.
- `find_path` 는 2D 로 당긴 꺾은선 — 꼭짓점 사이 직선 z 가 계단 구간에서 허공(최대 ~4 m). `pf.snap_point` 는 3D 최근접이라
  계단 발치에서 옆 경사로 끌어감. → `NavSurface.track`(xy 수직 투영 + 직전 높이 연속성).
- 좌표: trimesh glb 를 habitat 에 올리면 habitat(x,y,z)=우리(x,z,−y). 미리 회전하면 틀림.
- 컨테이너에 한글 폰트 없음 → 그림 라벨은 영어.

## 다음
vln_ce 02(공식 navmesh + GT 재현 검사) → 03 계획(밀어내기·스무딩) → 04 판정·내보내기 → 05 렌더.
확인 필요: 로봇 실제 키·턱, (−2.5,−15.5) 구조물.
