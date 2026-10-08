# 맵을 VLN-CE와 일치시키고 그 위에서 r_b를 흔든다 — S1/S2/S3 결과

2026-08-19 · `3dloader_vlnce/` · 상세 표와 Artifact 링크는 `reports.md` #0 / #0b / #6

## 문제

우리 voxel/ESDF 맵에서 **원본 GT가 `clearance ≥ r_b`를 위반**했다(r_b=0.2에서 4/7). 지도가 GT와
어긋난 상태로는 "이 embodiment로 통과 가능한가"를 판정할 수 없다. 그리고 ladder에 밑단이 없어
통과 가능한 leg도 전부 `nudge`를 거쳤고, 리포트가 `nudge 47/47`로 나와 "47개를 밀어냈다"로 오독됐다
(실제 이동량 median **0.000 m**).

## S1 — habitat navmesh를 기준으로 세웠다 (`03_verify_navmesh_gt.py`)

- 배포 `.navmesh` 설정 직독 = **habitat 기본값 그대로** (`agent_radius` 0.10 / `agent_height` 1.50 /
  `agent_max_climb` 0.20 / `cell_size` 0.05 / `edge_max_error` 1.3셀).
  → **"VLN-CE와 동일 조건" = `NavMeshSettings().set_defaults()`에서 `agent_radius`만 바꾸는 것.** 자유도 0.
- GT는 그 위에서 완전 정합: waypoint **474/474 · 531/531** navigable, `snap_point` 이동 **0.0000 m**,
  leg `find_path` ≤0.5 m **100%**(VLN-CE 논문 자신의 기준), 저장 per-frame GT **99.93%**(1516/1517).
- ~~**핵심 발견: GT는 navmesh 경계를 여유 0으로 스친다**~~ → **철회.** `clearance min`(최악의 한 점)을
  전형값으로 읽은 오류였다. 2026-08-20 #03 재작성에서 `clearance` 분석 자체를 삭제했다(navmesh는 이미
  반경만큼 깎인 지도라 그 위에서 여유를 또 요구하면 반경을 두 번 센다). 재측정: GT가 배포 navmesh를
  벗어나는 점은 17DRP **0**/2465 · s8 **14**/2450이고 그 깊이 최대 **3.7 cm < 한 칸(5 cm)**.
  → 자세히는 [[260820_03_rewrite_result]].
- `find_path`는 **경로는 재현하고 궤적은 재현하지 않는다** (대응점 거리 median 15.8 cm, 길이비 1.013).
  ⚠️ 2026-08-20 정정: 원인이 코너 절단만이 아니다 — s8에서는 **측지 최단선이 장애물의 반대편으로 돈다**
  (median 56.4 cm인데 **길이비 1.002**). **길이비 ≈ 1은 "같은 경로"의 증거가 아니다.**
  → **회랑 중심선은 GT 서브궤적이어야 한다**(선택이 아니라 필수).
- `recompute_navmesh`가 배포본 재현(52.03/52.04, 183.73/184.01) → r_b별 navmesh를
  `data/embodiment_aug/navmesh/`에 캐시(21 KB/개).
- `get_topdown_view` 규약 실측 확정: `tv[iz,ix]`, `ix=(x−lo0)/mpp`, `iz=(z−lo2)/mpp`. **단일 높이 절단**.

## S2 — 밴드 투영으로는 못 맞춘다 → navmesh 래스터화 채택 (`04_calibrate_map.py`)

⚠️ **2026-08-20에 #04를 재작성했다** → [[260820_04_rewrite_result]]. **아래 숫자는 지표가 다르다** —
여기 "88.7%"는 **GT 궤적 점**에 대한 일치율(GT 점은 두 지도 다 통행 가능이라 쉬운 문제만 채점)이고,
새 판의 "75.83%"는 **GT 주변 2 m 안 모든 셀**의 셀 단위 일치율이다. 직접 비교 금지.
새 판이 추가한 것: **불일치 원인 분해**(s8 거짓 승인의 66.6%가 "이 높이에 바닥 없음") ·
`& floor_exists` 수정(일치율 90.69%지만 거짓 기각 9배 → **채택 안 함**) · **pathfollower 재현**
(s8 median 56.4→8.8 cm이지만 4/9는 경로가 갈라져 50~82 cm). 결정은 그대로 **navmesh 래스터화**.

좌표 정합 게이트 **1.0000**. 후보별로 "GT가 이 맵의 `r_b` 마스크 밖으로 나가나"를 우리 맵과 habitat
**양쪽에** 물어 점 단위 일치율을 쟀다.

| 후보 | habitat 일치 | IoU@0.10 | identity @0.10 |
|---|---|---|---|
| 밴드 h_nav 0.10~0.25 × h_obs 1.25/1.50 (17DRP) | 96.9~97.1% | 0.920~0.923 | ✅ |
| 밴드 (s8) | **88.6~88.7%** | 0.634~0.652 | ✅ |
| **navmesh 래스터화 (채택)** | **100.00%** | **1.0000** | ⚠️ 66/8459 · **최대 깊이 0.050 m = 1셀** |

- **갈라지는 방향이 위험한 쪽이다**: 우리 밴드 맵이 일관되게 **더 관대**하다. s8 r_b=0.20에서
  habitat이 막았다는 **747점을 우리가 승인**(우리 178 vs habitat 925) = 거짓 승인.
- **선정 기준을 identity → habitat 일치율로 바꿨다.** identity 실패가 1셀이면 `nudge`가 2~3 cm로
  흡수하지만(샘플 손실 없음), 불일치는 "갈 수 없는 길"을 정답으로 가르친다(되돌릴 수 없음).
- `agent_max_climb=0.20`을 맞추면 문턱 문제가 풀린다는 가설은 **기각**(일치율 88.67→88.65%).
  → 2026-08-20 보강: **밴드 하단이 원인이 아니었던 게 맞다.** 진짜 원인은 밴드가 "이 높이에 바닥이
  있는가"를 아예 묻지 않는 것이었다(다층 씬에서 다른 층 공간이 새어 들어온다).

## S3 — `none` rung ladder + habitat 오라클 (`embodiment_augment.py`, `navmesh_grid.py`, `05_validate_waypoints.py`)

ladder 기본값 **`('none','nudge','corridor')`**. `none`은 GT를 그대로 쓴다(이탈 정의상 0).

| | 17DRP (69 leg) | s8 (42 leg) |
|---|---|---|
| **W5 identity @r_b=0.1** | `none` **69/69** · 보정 **0** · GT 거리 **0.00 cm** · **PASS** | 37/42 · 보정 2 · 0.58 cm · CHECK (`hard` ×5) |
| **W6 거짓 승인** | **0** / 276 leg | **0** / 188 leg |
| 보정이 걸린 곳 | r_b=0.2 `nudge` 2 · r_b=0.3 `corridor` 5 | r_b=0.2 `corridor` 3 · r_b=0.3 `corridor` 3 |
| 거짓 기각 | 0 → 12 → 19 → 27 | 8 → 16 → 29 → 5 |

s8의 W5 실패 5건은 **전부 `hard`** = S2가 정량화한 1셀 오차(66/8459점)와 동일 원인. 기준을 느슨하게
해서 100%로 만들지 않았다.

## 잡은 버그 2건 (자세히는 `reports.md` #6 troubleshooting)

1. **ladder 밑단 부재** → 통과 가능한 leg도 전부 `nudge`. "보정 필요 leg" 열이 없어 오독을 유발.
2. **`snap_to_grid`가 leg 시작점을 순간이동시키는데 끝점만 검사** → 결과가 **비단조**(17DRP ep0 leg0이
   r_b=0.20에서 blocked, **0.45에서 통과**). 시작점 이동 0.79 m → **1.68 m**. 마스크는 완전 단조임을
   먼저 확인해(위반 셀 0) 맵이 아니라 플래너 문제로 분리한 것이 진단의 핵심.
   조치: 스냅된 시작점이 `GOAL_TOL_M`(0.10 m)을 넘으면 기각.

## 정정한 이전 주장

- ~~"우리 맵이 recast보다 **과대 장애물**이라 GT가 위반한다"~~ → **방향이 반대**. 우리 맵이 더 관대하다.
- ~~"GT가 위반하는 이유는 **경계를 여유 0으로 스치기 때문**"~~ → **철회** (2026-08-20). GT는 데이터셋의
  지도에서 벽을 뚫지 않는다. 원 증상은 `Z_OFFSET_M = 0.20` 정렬 버그 + 데이터셋이 쓴 적 없는 `r_b = 0.2`
  에서 나온 숫자였고, 버그 정정 후 #05 identity는 69/69 · 47/47 PASS다. → [[260820_03_rewrite_result]]
- ~~`CLEARANCE_OFFSET_M = 0.10`(GT 판정 반경을 깎아주는 보정)~~ → 폐기. **관대한 맵을 더 관대하게**
  만드는 것이라 정반대로 위험했다. 맵 자체를 교체하는 것이 정공법이었다.

## self-check (씬·habitat 불필요, 총 47개 assert)

`embodiment_augment.py` 6+6 · `navmesh_grid.py` 7 · `04_calibrate_map.py` 8 · `waypoint_spine.py` 10 ·
`corridor_utils.py` 10 — 전부 통과.

## 남은 것

- **파이프라인이 habitat에 묶였다.** offline 캐시라 dataloader worker는 habitat 없이 돌지만
  Isaac/VLN-PE는 별도 맵이 필요하다.
- 거짓 기각(최대 29)의 두 원인 중 "첫 blocked에서 경로를 끊는다"는 정의 차이다. 줄이려면 leg를
  **이전 leg의 끝점에서** 이어 계획해야 한다(현재는 각 leg를 GT 구간에서 독립 계획).
- pixel-goal 국소 재계획(`07_validate_pixel_goal.py:176`)은 여전히 순수 A*다 — waypoint 앵커 미적용.
- `Z_OFFSET_M` 정정 이후 #7 modes / #8~#10 pixel-goal 리포트 재실행 대기.
