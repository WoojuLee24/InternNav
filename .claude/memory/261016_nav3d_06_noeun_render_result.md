# nav3d 04 · 06 · 06b · 06v — 노은역 vln_ce 렌더 + vln_pe GT 로 방식 검증 (2026-10-16)

## 04 경로 선택 (https://claude.ai/artifact/EJvqTfBEoGTj9dGq7RfHrb)
층 이동 100 (03 전부) + 평지 S0 50 · S1 50, 모두 U1~U5 통과 (U5 제외 106). 길이 중앙 61.0 / 16.0 m, mesh 여유 최소 0.287 m (> r_b 0.25).

## 06 렌더 (Isaac, 2026-10-16) — 200 ep · 45,726 프레임 · 9,783 s, pixel goal 중앙 0.949
`06_render_noeun_vlnce.py` -> `vln_ce_noeun_nav3d/traj_data/r2r/noeunstation/`. rig 칸 `60cm_15deg` 에 D455 값 (0.584 m · 14.76° · 전방 0.365 m).
자세 계산은 `episode_poses()` 로 뗌 (06v·06p 공용) — 떼기 전후 200 ep 자세·바닥 z 차 0 확인.

## 06b 검사 (https://claude.ai/artifact/MbHmZctMqYj1Qs1dqrpYYc) — R1 R2 R3 R5 R6 PASS · **R4 FAIL**
R4 평지 카메라 mesh 바닥 위 높이: 중앙 0.616 (목표 0.584), ±6 cm 안 89.4 %, 벗어난 952 프레임 전부 **위로**.
원인: 몸 바닥 z = navmesh 표면 — 노은역 navmesh 는 mesh 보다 중앙 3.1 cm (5~95 % 0.8~7.3 cm) 위 (04b 실측). GS 바닥은 mesh 보다 4~5 cm 더 아래(00).
게시: strip 668 장은 버전당 파일 511 개 한도 초과 → 400 프레임 묶음·절반 크기로 482 장, 4 번에 나눠 게시.

## 06v vln_pe GT 검증 (https://claude.ai/artifact/9sEPsQfeSSCfLTwhZya2me) — 사용자 요청 "노은역 Isaac 방식을 vln_pe 로 검증"
MP3D 7 씬 46 ep (계단 3 — vln_pe 2,813 ep 중 로봇 z 범위 > 0.5 m 는 3 개뿐). GT 로봇 xy + navmesh z ('gt'), nav3d 03 wp ('wp') 를 06 함수로 자세화, GT 와 짝 프레임 비교 + 같은 세션 Isaac 렌더.
- **핵심: H1 카메라는 mesh 바닥 위 1.656 ± 0.004 m 로 모든 ep 에서 일정, navmesh 위 높이는 ep 마다 1.46~1.61 m** → 로봇은 mesh 를 딛고 navmesh 는 장소마다 −5~+20 cm 어긋남.
- V2 높이 |Δz| p95: navmesh z 0.079/0.102 m (FAIL) → mesh z (`floor_mesh.py`) 0.021/0.022 m (PASS). V3 방향 PASS (Δyaw 4.2°, Δpitch 0.9°).
- V1 위치 FAIL (짝 거리 중앙 0.158): GT 프레임 간격 0.144 m vs 우리 0.25 m + yaw 로 짝 → 앞뒤 차가 대부분, 좌우 |중앙| 0.048 m. 자 설계 문제.
- V5 이미지 FAIL (SSIM 0.42 vs 기준 0.82): 짝끼리 0.16 m·4° 어긋남에 SSIM 이 민감. 그림으로는 같은 시점 — 규약 오류 없음. 자 설계 문제.
- V4 계단 mesh p95 0.112 (기준 0.10 FAIL), navmesh 0.186. 표본 3.
- 함정: MP3D glb 파일은 이미 z-up (= Isaac·nav3d 좌표). hab_to_ours 를 또 하면 높이가 3~13 m 로 나온다. apply_real/04 기본 far clip 이라 06v 렌더 먼 곳이 검정.

## 수정 준비 (아직 다시 렌더 안 함)
- `04b_paths_mesh_floor.py` -> `noeun_station_paths_meshfloor.json` (평지 = mesh 바닥 광선, 계단 = navmesh 경사 + 양 끝 차이 보간).
- `06p_render_noeun_vlnpe.py`: 노은역 vln_pe 포맷 (사용자 결정 "vln_pe 도"). 256×256 · hfov 90 · D455 자세 · vlnpe_writer. action 변환 pe[i] = ce[i+1], 끝 0. progress = 경로 길이 비율 (릴리스는 0.2 계단값).
- 다시 렌더 순서: 06 (--paths_json meshfloor, 2.7 h) → 06p → 06b.

## 다시 렌더 (2026-10-17) — mesh 바닥
- vln_ce: `vln_ce_noeun_nav3d_meshfloor/` 200 ep · 45,726 f · 9,707 s · goal 중앙 0.948. **06b 6/6 PASS — R4 99.9 % (중앙 0.584 m, 이전 89.4 % · 0.616)**. log `logs/06_render_noeun_meshfloor`.
- vln_pe: `vln_pe_noeun_nav3d/` 200 ep · 45,726 f · 20,554 s (256² 인데 vln_ce 보다 느림, 2.2 f/s) · 20 GB. 스키마 = 릴리스 200/200, npy 모양·depth 범위 이상 0, stats 200 · tasks 600 행. 지시문은 placeholder.
- 이전 `vln_ce_noeun_nav3d/` (navmesh z) 는 지우지 않고 남김.
