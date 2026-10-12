# vln_ce navmesh 검증 — 20 씬 결과 (2026-10-12)

프롬프트: [261012_habitat_navmesh_prompt.md](261012_habitat_navmesh_prompt.md) · 리포트: https://claude.ai/artifact/RfV2JGkfG2PLAogrGQWtkD

## 결론
- **GT 는 navmesh 기준으로 깨끗하다.** 계단 제외 356 ep 에서 GT 깊은 침범(> 2 셀 = 0.112 m)이
  occupancy **13** → navmesh **0**. navmesh 위반 37 건은 전부 ≤ 0.112 m(22 건 ≤ 1 셀) — r_b 로 깎인
  configuration space 경계에 GT 가 붙어 지나가서 생기는 래스터 반올림이다.
  → occupancy 의 GT 충돌(39/400)은 **지도 오차**: 계단 24 · 바닥 높이 어긋남(아래 버그 3) · 밴드 투영.
- **우리 경로는 navmesh 로 재면 실제로 깊게 침범한다.** 계단 제외 341 ep: occupancy 14(전부 1 셀) →
  navmesh 94, 그중 깊은 침범 **30 (8.8 %)**, > 0.3 m 24. occupancy 충돌 21 건은 전부 navmesh 에서도 충돌,
  navmesh 에서만 충돌 92 건. 원인은 occupancy 가 놓친 통행 불가 영역(바닥 없는 공간, 막힌 틈).
  JF19kD82Mey 가 대표(14/20, 깊이 2.4~2.7 m — 집 가운데 "바닥 없는" 영역을 직진) = 그 씬 같은루트 7/20 의 원인.

## 수치 (r_b 0.10, rig 125cm_30deg, --downsample_mode majority)

| | occupancy | navmesh |
|---|---|---|
| GT 실행 충돌 (전체 400) | 39 | 74 (계단 37) |
| GT 실행 충돌 (계단 제외 356) / 깊음 | 15 / 13 | 37 / **0** |
| 우리 경로 충돌 (367) | **21** (= 기준값, 03 과 일치) | 113 |
| 우리 경로 (계단 제외 341) / 깊음 | 14 / 0 | 94 / **30** |

03g (매칭 346 ep) — 기준값과 대조:

| | 이번 | 기준값 |
|---|---|---|
| 우리 ↔ 계획GT | chamfer 0.218 · Fréchet 0.618 · 같은루트 68.8 % | 0.217 · 0.584 · 73.7 % |
| 우리 ↔ GT실행 | 0.188 · 0.471 · 72.8 % | 0.192 · 0.487 · 73.1 % |
| GT실행 ↔ 계획GT | 0.141 · 0.451 · 78.9 % | 0.135 · 0.433 · 80.8 % |

매끄러움 (367 ep): m 당 회전 중앙값 우리 30.0° vs GT 26.7°(비 1.11), 곡률 p95 1.51 vs 4.27(90 % ep 에서 우리가 낮음).
17DRP5sb8fy 는 기준값과 동일(33.6/25.7, 1.63/3.77).

navmesh 캐시(03_verify_navmesh_gt): 20 씬 전부 게이트 A·B·C·D 통과, waypoint navigable 100 %,
GT 의 J2 밖 점은 전부 깊이 정확히 1 셀 — 2 씬 결과가 20 씬에서도 유지. 계단 ep 44 제외.

## 새로 발견한 버그 (고치지 않음 — 결정 필요)
1. **03 paths json 의 `episodes[].floor_z` 는 씬 대표 바닥**이 기록된다. 다층 씬에서 실제 계획 바닥과 다르다
   (1LXtFkjw3qL ep0: 기록 −3.06, 실제 +3.48). 이걸로 지도를 다시 만들면 다른 층을 잰다(처음 131/367 로 나온 원인).
   소비자는 03 main 루프 규칙(씬 바닥, `|ep−scene| > floor_tol_m` 이면 ep 바닥)을 다시 적용해야 한다.
2. **GT occupancy 기준값 29/400 은 재현 안 됨**(39/400). 우리 경로 21/367 은 정확히 재현되므로 충돌 정의는 같다.
   기준값을 낸 코드가 저장소에 없다.
3. **03 `floor_tol_m 0.3` > `h_nav 0.10`.** 바닥이 0.1~0.3 m 어긋난 ep 은 씬 바닥을 쓰고, 그러면 실제 바닥면이
   `(h_nav, h_b]` 밴드에 들어가 바닥 전체가 점박이 장애물이 된다. PuKPg4mmafe(+0.22 m) 성공 12/20 · GT occ 8/20 의 원인.
4. 02 게이트 FAIL 8 씬(`clearance=False`, GT zero_frac 6.8~32.7 %) — exit 1 이지만 esdf 는 저장되므로 03 진행 가능.
   `xargs ... 02 && 03` 으로 묶으면 03 이 건너뛰어진다.

## 실행 환경 · 커맨드
- habitat 컨테이너: `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` (habitat_sim 0.3.3).
  ⚠️ `...-gdsim`/`torch2.9.1-...-habitat-jazzy-5090` 이미지는 `/usr/bin/python` 에 habitat_sim 이 **없다**.
  이번엔 `internnav-habitat-navmesh` 컨테이너를 같은 마운트로 띄워 실행했다.
- scene_meta 는 호스트 `$GS/logs/scene_meta` 에 20 씬 모두 있어 01 은 생략. 02→03 은 씬당 ~7 s.
- 신규: `scripts/dataset_converters/gs_vlnpe/checks/03n_navmesh_collision.py`
  ```
  /usr/bin/python scripts/dataset_converters/gs_vlnpe/checks/03n_navmesh_collision.py --scenes <콤마구분> --label vln_ce_20scenes
  ```
  `--r_b 0.10`(navmesh 캐시 r_b) · `--grid_out_dir data/embodiment_aug/navmesh_grid`(npz 내보내기, 빈 문자열이면 생략) ·
  `--esdf_dir/--paths_dir`(02/03 산출물). 출력 `logs/gs-vlnpe/03n_navmesh_collision/<label>/summary.json` + 충돌 ep PNG.
  `summary.ours_occ_matches_03` 이 True 여야 occupancy 재계산이 03 과 같다는 뜻.
- **npz 격자**: `data/embodiment_aug/navmesh_grid/<scene>_rb0.10.npz` (20 씬, 3.0 MB).
  `mask`/`esdf` (K,Ny,Nx) · `floor_y` (K,) · `origin` · `cell` · `r_b`. K = 이 20 ep 들이 쓴 바닥 높이별 절단
  (다른 바닥이 필요하면 habitat 으로 다시 뽑아야 함). esdf 는 `esdf_from_mask` 규약 — r_b 를 다시 적용하지 말 것.
  격자 = 02 vln_ce esdf 의 origin/cell. **habitat 없이 400 ep 재판정 → 불일치 0** 확인.
