# 벽에 안 붙으면서 GT 따라가기 — GT 여유 매칭 refine (`corridor_gtc`) 실측

2026-08-20 · `gt_clearance_refine.py`(신규) · `05_validate_waypoints.py` 실험결과 2b ·
[17DRP](https://claude.ai/code/artifact/003a26b8-2335-4cdb-bde5-bc0ce6d3c057) ·
[s8](https://claude.ai/code/artifact/e6fd96c9-f60a-446d-b0a8-1f17db0ee1e6)

## 무엇을 만들었나

기존 refine 양극단(`refine_min_move` = r_b까지만 / `greedy_refine` = 최대(N1식, GT 이탈)) 사이의
**`refine_match_gt`**: 점마다 목표 여유 = clamp(**GT 최근접점의 esdf**, r_b, r_b+0.30)로 최소 이동.
좁은 문(GT도 여유 없음)은 안 밀고, 넓은 방은 GT 쪽으로 민다. A* 단계 push(`clearance_weight`)는
경로 자체가 바뀌어 이미 기각됐으므로(03d) refine 단계에 넣었다.
ladder rung `'corridor_gtc'`로 연결(기본 ladder 불변 — W5 identity 69/69·47/47 그대로).

## 전제부터 수정됐다 (측정이 가설을 이김)

**"우리 생성 경로가 벽에 붙는다"는 ladder에는 해당 없었다.** corridor 출력의 |경로 여유 − GT 여유|
median이 이미 **3.4 cm**(두 씬 공통), 여유<r_b+0.05 비율 0.3~1.6%. 벽에 붙는 것은
**navmesh 측지선/follower 경로**([[260820_gt_wall_clearance_result]])였다 — GT 중심 회랑 + A* +
min_move 설계가 이미 GT 여유를 거의 보존하고 있었다.

## 결과 — 씬에 따라 갈림 (사전 등록 예측 판정)

| 예측 | 17DRP | s8 |
|---|---|---|
| P1 \|Δ여유\| 개선 | **PASS** 3.4→**1.5 cm** | FAIL 3.4→4.8 cm |
| P2 GT 거리 악화 ≤1 cm | **PASS** 7.5→**6.0 cm** (개선!) | FAIL 17.0→19.5 cm (+2.6) |
| P3 새 blocked 없음 | **PASS** — blocked **5→0** | **PASS** — blocked **3→0** |
| P4 큰 r_b 일반화 | **PASS** (0.5 vs 2.4 / 0.0 vs 0.7) | FAIL (0 vs 0 — 신호 없음) |

ablation: `fixed+0.15`(GT 매칭 없는 고정 마진)와 `cap=0.15` 둘 다 full gtc보다 나쁨(17DRP GT거리
8.1/7.6 vs 6.0) → **GT 매칭이 실제 기여分**이라는 직접 증거. 고정 마진의 "좁은 문 blocked" 예측은
안 나타남(문이 그만큼 좁지 않았음).

**채택 규칙(P1~P3 전부)에 따라 기본 ladder 교체는 안 함** — s8에서 P1·P2 실패.

## 예상 밖의 발견 — blocked 해소 (양쪽 씬 공통)

강제 corridor 모드에서 blocked였던 leg **8개(5+3)가 corridor_gtc에서는 전부 통과**한다.
메커니즘: `refine_min_move`가 남긴 계단형 점을 스플라인이 코너컷하며 hard/body 위반 → `_path_ok` 기각.
match_gt는 점을 GT 여유 쪽으로 밀어 스플라인이 안전해짐. **커버리지 이득 후보** — 단, 기본 ladder에서는
baseline에서 `none`이 다 받으므로 효과는 큰 r_b에서만 나타날 것(gtc-ladder 스윕은 미실행, 후속).

## s8에서 왜 나빠지나 (분석, 부분 추정)

"여유가 높은 곳에 GT가 있다"는 휴리스틱은 단순 복도에서 성립하지만, s8처럼 장애물이 복잡하면
같은 여유값의 등고선이 **GT 반대쪽**에도 있어 최근접-셀 선택이 GT에서 멀어질 수 있다.
여유(스칼라)는 위치를 유일하게 결정하지 않는다 — 방향 항(GT 쪽 반공간 제한)을 넣으면 고칠 수 있을
것이나 **미구현**(후속 축).

## 파일

- 신규 `gt_clearance_refine.py` — `refine_match_gt(…, cap_m, fixed_m, corr)` + selfcheck 6/6
- `embodiment_augment.py` — `_plan_leg`에 `'corridor_gtc'` 분기 + `follow_waypoints`에
  `gtc_cap_m/gtc_fixed_m` optional 파라미터 (기본 동작 불변)
- `05_validate_waypoints.py` — `MODES` + `_clr_stats` + 실험결과 2b(여유 스타일 표 + P1~P4 판정)
- P3 판정식을 실행 중 정정: "동일" → "**새 blocked 없음(≤)**" — 사전 등록한 실패 조건이
  "새 blocked이 생기면"이었고 감소는 실패가 아니므로 명세 수정(기준 완화 아님)

## 남은 것

- **방향 항**: 목표 여유 매칭에 "GT 쪽 반공간" 제약 추가 — s8 역행의 유력 해법, 미구현
- **gtc-ladder r_b 스윕**: blocked 해소가 실제 커버리지(usable frames)로 이어지는지 미측정
- 여유 스타일 문제의 진짜 소재지: ladder 출력이 아니라 **follower/측지선 기반 재생성**을 쓸 경우다 —
  그 경로를 쓰기로 하면 이 refine을 후처리로 붙이는 것이 답
