# GT는 왜 우리 경로보다 벽에서 떨어져 가나 — 의도인가? (vln_ce vs vln_n1)

2026-08-20 · 계기: 사용자 "gt 노랑 path가 우리 path보다 벽에 더 떨어져 있다. habitat vlnce는 일부러
벽에서 떨어지게 만든 건가?" + "vln_n1은 ESDF로 벽 거리 최대화했다는데"

## 답 요약

| | vln_ce (지금 #03/#04의 노랑 GT) | vln_n1 (InternVLA-N1) |
|---|---|---|
| 생성 방식 | **이산 액션 follower** (실측: 이동 스텝 100%가 0.25 m · 회전 90~100%가 15° 배수) | **ESDF 기반 연속 경로** (스플라인 — gs_vlnpe가 cubic spline으로 재현) |
| 벽 거리 최대화 의도 | **문서화된 의도 없음** (논문: "shortest path via low-level actions" — 벽 거리 항 없음) | **있음** — refine 목적함수 자체가 "장애물에서 멀어지기" ([[260804_gs_vlnpe_reproducibility_limits]]) |
| 실측 벽 여유 | navmesh 추가 여유 median **0.36 / 0.24 m** (17DRP/s8) | 우리 ESDF 기준 median **0.46 / 0.67 m** (다른 지도·지표 — 직접 비교 주의) |

사용자의 말이 **N1에 대해서는 정확하다**. 지금 보는 노랑 GT는 vln_ce라 그 설명이 적용되지 않는데,
그럼에도 벽에서 떨어져 있고 — 그 이유는 **미해결**이다 (아래).

## 내 가설 2개가 실측으로 반증됨

**측정**: 배포 navmesh 위 `distance_to_closest_obstacle`(navmesh가 이미 r_b=0.10 깎였으므로 "추가" 여유),
경로를 5 cm로 채움. 세 경로: ⓐ 저장 GT / ⓑ 우리 headless follower(벽 회피 항 없음) / ⓒ find_path 측지선.

| 경로 | 17DRP median · 여유<0.10 | s8 median · 여유<0.10 |
|---|---|---|
| ⓐ 저장 GT | **0.360** · **3.4%** | **0.237** · **12.9%** |
| ⓑ 우리 follower (sliding ON) | 0.285 · 14.3% | 0.207 · 30.1% |
| ⓑ′ 우리 follower (sliding OFF) | 0.289 · 11.5% | 0.208 · 28.0% |
| ⓒ find_path 측지선 | 0.295 · 9.2% | 0.215 · 27.8% |

1. ~~"이산 액션의 부산물"~~ → **반증.** 같은 0.25 m/15° 이산 액션을 쓰는 우리 follower가 측지선만큼
   벽에 붙는다(여유<0.10이 GT의 2~4배). 이산화만으로는 GT의 여유가 안 나온다.
2. ~~"sliding 차이"~~ → **반증.** `try_step_no_sliding`으로 바꿔도 거의 그대로(30.1→28.0%).
   leg 도달은 여전히 100%.

**남는 결론**: vln_ce GT의 추가 여유는 의도된 목적도, 단순 이산화 부산물도 아니고,
**2020년 당시 habitat-lab `ShortestPathFollower` 구현의 특성**이다. 현대
`GreedyGeodesicFollower`(greedy 측지 하강)와 액션 선택 규칙이 달랐던 것으로 보이나
**정확한 메커니즘은 특정하지 못했다** — 확정하려면 당시 구현(v0.1.x)을 재현해야 한다.

## 파이프라인에 주는 함의

- **#05의 회랑 중심선 = GT 서브궤적** 선택이 다시 한 번 옳다 — GT가 벽에서 넉넉하므로 큰 `r_b`에서도
  중심선이 더 오래 유효하다. `find_path`나 우리 follower를 중심선으로 쓰면 벽에 붙는다.
- **우리 follower로 경로를 재생성하면 원본 GT보다 벽에 붙는 스타일 차이**가 생긴다(여유<0.10 기준 2~4배).
  r_b를 키워 재생성할 때 이 차이가 학습 분포에 들어간다 — 아직 정량화 안 한 리스크.
- 옛 #03 D2의 절반은 유효했다: ~~"GT가 여유 0으로 스친다"~~는 철회가 맞지만,
  "**find_path가 GT보다 벽에 붙는다**"는 관찰 자체는 정확했다 — 이번에 follower까지 포함해 재확인.

## 명령 (재현)

측정 스크립트는 일회성 프로브(세션 스크래치)로 돌렸다 — `04_calibrate_map.py`의 `follower_path` /
`03_verify_navmesh_gt.py`의 `chain_find_path`·`dense3` 재사용, `distance_to_closest_obstacle`로 여유 집계.
sliding OFF는 `agent.controls.move_filter_fn = pf.try_step_no_sliding` 한 줄 차이.
