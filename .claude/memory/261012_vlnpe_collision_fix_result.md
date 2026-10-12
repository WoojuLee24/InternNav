# vln_pe 충돌 제거 — 파라미터 + 새 방법 (2026-10-12)

## 결론

| | 관통 | r_b 미달 | 경로 생성 |
|---|---|---|---|
| 시작 (`plan_r_b` 없음 = 0.20) | 17 | 403 | 2274 |
| `plan_r_b 0.25` + `post_refine` | 6 | 166 | 2203 |
| **파라미터** `0.28` + `margin 0.05` | 6 | 85 | 2056 |
| **+ `--thin_safe`** (새 코드) | 1 | 31 | 2056 |
| **+ `--smooth bezier`** | **0** | **20** (18은 GT 끝점) | 2056 |

45씬 2352 에피소드 · `r_b 0.20`. 네 줄 모두 **그 자리에서 같은 조건으로 다시 돌린 값**이다
(rule 9-2). 관통 = 여유 ≤ 0, r_b 미달 = 여유 < 0.20 — 둘은 다른 사건이라 항상 따로 적는다 (rule 9-3).

## 명령

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --dataset vln_pe --scene <SCENE> --r_b 0.20 --plan_r_b 0.28 --h_nav_ratio 0.12 --refine_radius 0.20 --smooth bezier --thin_safe --post_refine --post_refine_margin 0.05 --num_episodes 999 --esdf_dir scripts/dataset_converters/gs_vlnpe/logs/esdf --out_dir <OUT> --log_dir <LOG>
```

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--r_b 0.20` | 로봇 몸 반지름 (검사 기준) | H1 휴머노이드 |
| `--plan_r_b 0.28` | **계획** 반경 | spline 이 여유를 중앙 0.05 · p90 0.15 깎는다. `r_b+0.05`(=0.25)로는 부족 |
| `--post_refine_margin 0.05` | 수리 목표를 `r_b` 보다 이만큼 위로 | 0 이면 수리 직후 재스무딩이 같은 양을 다시 파고든다 |
| `--thin_safe` | 솎기가 만든 관통 구간은 점을 되살림 | 아래 ①. **신규** |
| `--smooth bezier` | waypoint 를 통과하지 않고 convex hull 안에 머문다 | **부풀 수가 없다**. 아래 ③ |

## 진단 — 어디서 여유를 잃는가

`checks/03h_violation_breakdown.py`(신규)로 위반을 **출처별로** 가른다.

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/checks/03h_violation_breakdown.py --paths_dir <OUT>/paths --r_b 0.20 --plan_r_b 0.28
```

`plan_r_b 0.25` 일 때:
```
  r_b 미달 166건의 출처:
    waypoint 부터 나쁨 :  42   <- 계획 문제. 삽입으로 못 고침
    spline 만 나쁨     : 124   <- post_refine 이 고칠 수 있는 것
  spline 이 깎아먹은 양: 중앙 0.050 m · p90 0.150 · 최대 0.354
  post_refine 발동 294 · 통과 128 · 실패 166 {no_gain 132, turn 33}
```
`no_gain` 132 건이 **margin 0 의 증거**다 — 정확히 `r_b` 까지만 밀면 재스무딩이 도로 깎는다.
`margin 0.05` 로 바꾸자 5씬 기준 수리 성공이 **4/12 → 32/33** 으로 뒤집혔다.

## 새로 찾은 버그 — 파라미터로는 못 고치는 것

### ① 솎기가 안전 검사를 빠져나갔다  (`--thin_safe`)

`thin_waypoints` 는 호길이로만 0.8 m 마다 고른다. 모서리 점이 빠지면 남은 두 점의 직선이
벽을 가로지르는데, 파이프라인은 솎기 **전** `wp_refined` 만 검사했다. 정작 spline 이
통과하는 건 솎은 **후** 점들이다. `thin_waypoints` docstring 의

> `r_b` 제약은 솎기 전 `wp_refined` 에 걸리므로 안전 판정은 영향받지 않는다

가 틀렸다 — 같은 docstring 이 RDP 를 거부한 이유와 **똑같은 결함**이다.

실측 (`759xd9YjKW5` ep30):
```
  waypoints_astar     꼭짓점 0.250 · 선분 0.250 · 관통샘플 0
  waypoints_refined   꼭짓점 0.320 · 선분 0.320 · 관통샘플 0
  waypoints_thinned   꼭짓점 0.250 · 선분 0.050 · 관통샘플 0   <- 여기서 무너진다
  trajectory          꼭짓점 0.000 · 선분 0.000 · 관통샘플 32
```

### ②  되살린 점의 간격 — ① 의 첫 구현이 만든 회귀

처음엔 **인덱스 중점**으로 점을 되살렸다. 그랬더니 이웃 바로 옆에 점이 들어가 간격이
0 이 됐고, 그 불균등한 knot 간격이 cubic spline 을 **1.23 m** 부풀려 `JeFG25nYj2p` 에
새 관통 6건이 생겼다 (중복점 있는 에피소드 24 → 83).

고친 방식 — 분할점을 **두 조각의 여유가 최대가 되는 점**으로 고르고, 남는 과밀은
`min_spacing_m`(기본 `spacing_m/4`)로 풀되 **뺄 때마다 안전을 재확인**한다.
그 결과 관통 6 → 1, 중복점 83 → 원래 수준.

## 해보고 버린 것 (음성 결과)

### `--downsample_mode majority` 전역 적용 — **쓰면 안 된다**
vln_ce 에서는 `r_b 0.10` 이 작아서 효과가 있었지만, vln_pe 는 `r_b 0.20` 이 커서
coarse 격자가 과하게 막힌다. 5씬 실측 **에피소드 217 → 151**, r_b 미달 개선은 없음(1 → 1).

### `--post_refine_margin 0.10` — 0.05 보다 나쁘다
5씬 r_b 미달 **1 → 4**. 과하게 밀면 재스무딩이 다른 데서 튄다.

### `--refine_radius` 확대 — **오히려 나쁘다**
0.20 창이 좁아서 여유를 못 끌어올린다고 보고 0.30 / 0.40 을 시험했다. 같은 6씬:
r_b 미달 **15 -> 22 (0.30) / 17 (0.40)**, 회전도 31.7 -> 34.0 -> 36.2 로 거칠어진다.
`waypoint 부터 나쁨` 6건은 창 크기와 **무관**했다(끝점이라 refine 이 안 건드리기 때문 — 위 참조).

### A* 실패 에피소드만 majority 로 재계획 — **효과 0, 코드 되돌림**
"낙관적 `any` 다운샘플이 좁은 틈을 통과시킨다"는 가설로 per-episode fallback 을 구현했으나,
45씬 산출물이 fallback 없는 것과 **바이트 동일**했다. 계측해 보니 majority 재계획은
거의 항상 길을 아예 막고(`start_or_goal_not_navigable` / `astar_failed`), 성공할 때는
**같은 결과**를 낸다. 남은 19건은 coarse 격자 탓이 아니다.

### ③ cubic 대신 bezier — **이미 있던 옵션**

`smooth_bezier` docstring 이 "좁은 씬·큰 `r_b` 일 때 쓸 옵션"이라 적어 뒀는데, vln_pe 가
정확히 그 조건이다(`r_b 0.20`, vln_ce 의 2배). cubic 은 waypoint 를 **통과**하느라 그 사이에서
부풀지만, bezier 는 convex hull 안에 머물러 **애초에 부풀 수가 없다**.

45씬 (변수는 스무딩 하나):

| | 관통 | r_b 미달 | spline 이 깎는 양 p90 | chamfer | Fréchet | 같은루트 | 길이비 | 회전 |
|---|---|---|---|---|---|---|---|---|
| cubic | 1 | 31 | 0.114 | 0.170 | 0.489 | 1569 | 0.925 | 33.6 |
| **bezier** | **0** | **20** | **0.108** | 0.176 | 0.492 | 1565 | 0.898 | 25.6 |

GT 와의 거리(chamfer 0.170 vs 0.176)와 루트 일치(1569 vs 1565)는 사실상 같고,
길이비만 0.925 -> 0.898 로 모서리를 조금 더 질러간다. GT 회전은 100.5 도/m 라 어느 쪽도
GT 보다 훨씬 완만하다(33.6 / 25.6).

## 남은 20건 — 18건은 **GT 의 시작/목표점**이다

```
  waypoint 부터 나쁨 : 19
  spline 만 나쁨     :  1
```
19건의 최소 여유 지점이 어디인지 재보면:
```
  시작점 11 · 목표점 7 · 중간 1
```
`refine` 은 `fix_endpoints=True` 라 양 끝을 **움직이지 않는다** — 시작/목표는 GT 좌표
그대로다. 즉 이 18건은 우리 계획의 실패가 아니라 **GT 끝점을 우리 occupancy 로 쟀을 때
좁게 나오는 것**이고, rule 9-1 이 금지하는 비교에 해당한다(GT 끝점은 habitat navmesh 산물,
그쪽은 `agent_radius 0.10` 으로 이미 깎여 있다). 끝점을 옮기면 reproduce 계약이 깨지므로
옮겨서도 안 된다.

**진짜로 남은 것은 중간 1건(`s8pcmisQ38h` ep11) + spline 1건 = 2 / 2056 (0.1 %).**

## 검증

- `--thin_safe` 생략 시 산출물 **바이트 동일** (`cmp` PASS, 2가지 조합)
- `esdf_utils.py` self-check 전부 통과
- 대조군은 모두 같은 코드·같은 45씬으로 그 자리에서 재실행 (rule 9-2)
