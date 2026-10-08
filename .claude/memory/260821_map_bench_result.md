# 학습(batch)용 지도 벤치마크 — navmesh vs 우리 맵 (속도·제약)

2026-08-21 · 계기: "batch 단위 학습에 navmesh로도 가능한지? 속도는? 우리 맵은?"
측정: 세션 스크래치 벤치(REP 20/10, 워밍업 제외) · `scene_grid`/`navmesh_grid.load_mask`/
`recast_like.recast_mask`/`follow_waypoints` 그대로 사용.

## 샘플당 지도 생성 비용 (r_b=0.10, 에피소드 floor 기준)

| 지도 | 17DRP | s8 | 연속 r_b | habitat | 비고 |
|---|---|---|---|---|---|
| ① 밴드 (occ, 현 기본) | **1.7 ms** | **4.2 ms** | ✅ | 불필요 | 사실상 공짜 |
| ② navmesh 래스터화 (현 default) | 37.3 ms | 72.5 ms | ❌ **이산만** | **필요** | 콜드(.navmesh 로드+첫 래스터) +70~245 ms/씬 |
| ③ recast v2 (csum 캐시 후) | **17.0 ms** | **40.8 ms** | ✅ | 불필요 | 캐시 전 44.6/234.5 ms — cumsum(96 MB)이 병목이었음 |

`follow_waypoints` 전체(지도+A*+refine+spline, r_b=0.20): occ **4.8~6.3 ms** vs navmesh **36~68 ms**/에피소드.
참고 스케일: 학습 샘플 전체 비용은 depth 렌더가 지배(R2 실측 ~349 ms/sample) — **지도 비용은 어느 쪽이든
렌더 대비 소액**(최악 s8 navmesh 72 ms ≈ 렌더의 21%, recast 41 ms ≈ 12%).

## 질문에 대한 답

**1. navmesh로 batch 학습 가능한가?** — **가능. 단 제약 둘.**
- **이산 r_b만**: `.navmesh` 캐시가 r_b별 파일(21 KB)이라 **연속 r_b 샘플링 불가** —
  r_b를 격자화(예: 0.05 간격)해 #03으로 사전 생성해야 한다. 학습 중 `recompute_navmesh`는
  Simulator+glb가 필요해(수 초) 불가.
- **worker에서 habitat_sim import 필요**: `PathFinder`는 GL 컨텍스트가 없어 **fork 안전**
  (헤드리스 확인됨). import 비용은 worker당 1회.
- 속도: 웜 37~73 ms/호출. 병목은 `get_topdown_view` 래스터가 **호출마다** 도는 것 —
  (scene, r_b, floor_y)로 마스크를 캐시하면 epoch 반복 시 ~0으로 상각 가능(미구현).

**2. 우리 맵은?**
- 밴드: **1.7~4.2 ms** — 지금도 학습 기본값이고 속도 문제 없음. esdf가 r_b와 무관하므로
  (scene, floor_y) 캐시 시 r_b당 threshold ~0.1 ms.
- recast v2: csum을 `grid['_recast_csum']`에 씬당 1회 캐시하도록 수정(이번에 반영, selfcheck 13/13
  유지) → **17~41 ms**. 연속 r_b OK, habitat 불필요 — **학습 투입 가능한 속도**.

## 학습 투입 시 판단 기준 (속도 아님)

세 지도 모두 렌더 대비 충분히 싸므로 **선택 기준은 속도가 아니라 정확도·제약**:
- habitat 씬 + 이산 r_b면 → navmesh 래스터(정확도 100%)
- 연속 r_b 또는 navmesh 없는 씬(Isaac) → **recast v2**(habitat 일치 92~96%)
- 밴드는 다층 씬에서 거짓 승인(허공)이 크므로([[260820_04_rewrite_result]]) 단층 씬 외 비권장

recast v2를 학습 경로 생성에 실제로 쓰려면 `embodiment_augment`에 `map_source='recast'` 연결이 필요
(작음: `leg_grids` 분기 + `navmesh_grid.esdf_from_mask` 트릭 재사용 — mask가 이미 r_b 깎임) — **미구현**.

## 잡은 것

- EGL 충돌: Open3D 렌더러를 habitat import **후에** 만들면 core dump — 반드시 렌더러 먼저
  (기존 05의 순서가 정답이었음). 벤치에서 재확인.
- recast v2 cumsum을 grid dict에 캐시 (`recast_like.py` 수정, s8 5.8배 빨라짐).
