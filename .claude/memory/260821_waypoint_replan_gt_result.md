# #09 — waypoint만으로 GT는 8~9 cm까지 재현되고, 남는 오차는 전부 route-split이다

2026-08-21 · `3dloader_vlnce/09_waypoint_replan.py`(신규 1파일, 기존 파일 수정 0) ·
`logs/embodiment_augment/waypoint_replan/` · Artifact https://claude.ai/code/artifact/b510cb8e-fb12-40a0-9869-4ba8c89042dd
계기: "navmesh 맵과 recast v2 맵이 주어짐. 각각 확인하고 정답 경로에 가까워지도록 경로 생성.
GT 입력 금지(waypoint만) · r_b=0.10 고정 · 양자화는 일단 제외."

## 무엇을 했나
GT 궤적을 planner에 주지 않고 사람 주석 waypoint(`reference_path`) 좌표만으로, 두 맵 위에서 **완전히
동일한 planner**로 경로를 생성해 GT 재현도를 쟀다. 맵은 bool 마스크만 공급 —
`navmesh_grid.esdf_from_mask`(+r_b 트릭 재사용) → `truncate_navigable`이 공용 변환이라 생성 코드에
map 분기가 없다. 변형 사다리(사전 고정): P0 waypoint 체인 A* → P1 +솎기 0.8 m·cubic spline →
P2(w) +clearance_weight {0.5,1,2} (esdf를 0.30 m로 캡 — 캡 없으면 개활지 비용이 1e-6으로 퇴화해 A*가
개활지로 우회한다).

## GT 생성 방법 (논문·코드로 확인 — 추정 아님)
VLN-CE(ECCV 2020, arXiv:2004.02857) §3.1 + repo `habitat_extensions/shortest_path_follower.py`:
R2R 노드를 navmesh에 스냅(2 m 하향 레이캐스트, 변위 ≤0.5 m) → waypoint 사이 navmesh 근사 최단경로
("A*-based heuristic search") → 구식 `ShortestPathFollowerCompat`(0.25 m/15°, 매 스텝 재계획)의 실행
흔적이 `gt.json.gz` locations/actions. 통과 기준 = 다음 waypoint 0.5 m 이내(77% 전이).
→ GT = "waypoint 경유 최단경로 + 이산 실행". GT 자신이 양자화 잔물결을 가지므로 매끈한 생성 경로의
gt_dist에는 **노이즈 플로어 2.6~2.7 cm**(GT vs smooth GT, 실측)가 있다.

## 결과 — 게이트 (17DRP 5/5 · s8 3/5, 기준은 실행 전 고정)

| 게이트 | 17DRP5sb8fy | s8pcmisQ38h |
|---|---|---|
| G1 GT 생존율 ≥99% (맵 확인) | navmesh 100% · recast 100% ✅ | navmesh 99.18% · recast 100% ✅ |
| G2 체인 A* 전 leg 연결 | ✅ | ✅ |
| G3 최선 gt_med ≤15 cm | **9.4 / 10.9 cm** ✅ | navmesh **10.9** ✅ · recast **17.5 ❌** |
| G4 두 맵 차 ≤5 cm | 1.5 cm ✅ | **6.6 cm ❌** |
| G5 len비 ≤1.05 & SPL 손실 ≤0.02 | ✅ | ✅ |

최선 변형: 17DRP 둘 다 P2(w=0.5), s8 navmesh P1 · recast P2(w=0.5). clearance_weight 이득은 최대
−0.7 cm로 미미, w=2부터 악화.

## 결과 — 오차 분해: split leg가 전부다

split(leg 대응거리 >0.20 m)을 빼면 **두 맵·두 씬 모두 leg median 8.1~9.2 cm로 균일** — planner는
지배 변수가 아니다.

| 최선 변형 | 비-split leg med | split | split leg med |
|---|---|---|---|
| 17DRP navmesh / recast | 8.1 / 8.7 cm | 7 / 6 (69 leg) | 22 / 23 cm |
| s8 navmesh / recast | 9.0 / 9.2 cm | **4 / 6** (47 leg) | **128 / 71 cm** |

- **navmesh split(ep9 전형)**: GT는 장애물 섬을 위로, 우리는 아래로 — 어느 쪽으로 돌지는 waypoint에
  없고 instruction에만 있는 정보. #04의 habitat follower도 같은 씬 4/9 leg가 갈라졌다 → **waypoint
  표현의 정보 한계이지 생성기 문제가 아니다.**
- **recast 추가 split +2(ep1 전형)**: recast가 중앙 구역을 잘못 열어(#04 잔여 거짓 승인 F=745) A*가
  waypoint 사이를 직진. **거짓 승인 = shortcut = GT 재현 직접 훼손.**

## 결론
1. 루트가 유일한 구간은 waypoint만으로 GT를 8~9 cm(플로어 2.7 cm)로 재현한다.
2. route-split은 waypoint-only의 구조적 한계 → **"corridor 중심선=GT"(#05 ladder)는 선택이 아니라
   필요조건**임이 실측 확정.
3. recast v2는 GT 생존 100%지만 거짓 승인이 split을 만든다 → **경로 생성 축에서도 default 맵은
   navmesh 래스터화**(#04 결론 재확인).

## 검증
- selfcheck 9/9(합성 마스크, 씬·habitat 불필요): 체인A*·하드충돌0·스무딩끝점·항등평가·평행이동·split
  검출·clearance밀기·플로어·생존율.
- `git status`: 기존 파일 수정 0 — 신규 `09_waypoint_replan.py` + `reports.md`/메모리 문서만.
- 시각화: `logs/embodiment_augment/waypoint_replan/<scene>/ep*_{navmesh,recast}.jpg` + `report.html`,
  ep1(recast shortcut)·ep9(route-split)를 육안 확인.

## 실행 명령
```bash
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/09_waypoint_replan.py --scene 17DRP5sb8fy
/usr/bin/python scripts/dataset_converters/3dloader_vlnce/09_waypoint_replan.py --scene s8pcmisQ38h
```
| 인자 | 의미 |
|---|---|
| `--maps navmesh,recast` | 비교할 맵 (마스크 공급자만 다름) |
| `--clearance_weights 0.5,1,2` | P2 스윕 값 (0.30 m 캡한 esdf에 적용) |
| `--episodes 14` / `--n_frames 40` | #05와 동일 로더·필터(계단 z_spread>0.30 제외) |
| `--selfcheck` | 합성 마스크 자기검사 (씬 불필요) |

## 남은 것
- 2씬 · 계단 제외 · r_b=0.10만. 에피소드 중복(17DRP ep0=ep10 등) 미제거 집계.
- 양자화(0.25 m/15° locations+actions) 출력 보류 — 생성 path를 새 GT로 쓸 때 필요(사용자가 추후 적용
  가능으로 판단).
- split을 waypoint 없이 푸는 방법(instruction 파싱 등)은 범위 밖.

관련: [[260820_04_rewrite_result]] [[260820_03_rewrite_result]] [[260821_map_comparison_report]]
