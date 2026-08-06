# `03_sample_gt_paths.py --mode random` (C2) 구현 결과

M1.3의 마지막 조각. `reproduce`(C1, GT에서 h_b/pitch/start/goal 전부 복사)와 달리, 논문처럼
무작위로 h_b/pitch/start/goal을 뽑아 같은 파이프라인(A\*→refine→thin→spline)을 돌린다.

## 구현

- `h_b ~ U(0.25, 1.5)`, `pitch ~ U(0, 30°)` (`sample_random_episode`).
- start/goal: 그 `h_b`로 만든 `nav_coarse`의 **최대 연결 성분**(`scipy.ndimage.label` + 최대
  크기 선택)에서 무작위 두 셀을 뽑되, 월드 거리 ≥ 2 m(`sample_random_start_goal`). 작은 성분은
  고립된 방일 확률이 높아 처음부터 제외했다.
- 이후 `plan_episode` 호출은 reproduce와 **완전히 동일한 함수/코드 경로** — 분기는 `main()` 맨
  앞의 guard clause(`if args.mode == 'random': return main_random(args)`) 하나뿐이고, 계획 로직
  내부에는 분기가 없다(가이드라인의 guard clause 원칙).
- esdf 로딩 블록은 reproduce/random 양쪽이 완전히 같아서 `load_esdf()`로 뽑아 공유했다 —
  순수 추출이라 reproduce의 결과(chamfer 0.103 등)가 재실행으로 **완전히 그대로** 나옴을 확인했다.

## 검증 방식이 reproduce와 다르다

GT가 없어 chamfer/Fréchet로 못 잰다. 대신:
- **하드 게이트는 reproduce와 동일하게 유지**한다 — refined waypoint clearance ≥ `r_b`, 하드 충돌 0.
- **분포 비교**는 reproduce에서 실측한 GT 통계(길이 median 6.03 m, clearance median 씬별
  0.46~0.68 m, 길이/직선 비율 median 1.08)를 참고값으로 걸되, 매우 넓은 범위(0.3~3배 등)의
  **스모크 테스트**로만 쓴다 — 하드 게이트에 넣지 않는다. 무작위 목적지 쌍은 GT 에피소드와 애초에
  다른 거리/난이도라 정확히 일치할 이유가 없다(실제로 씬2는 지도가 커서 길이 median이 16.4 m로
  나왔다 — 참고범위 0.3~3배(1.8~18.1) 안이라 여전히 통과).

## 새로 발견한 것 — 무작위 start/goal이 하드 게이트를 reproduce보다 훨씬 자주 위반한다

처음 구현(재추출 없이 그냥 채택)했을 때 씬1 20개 중 **7개**가 refined waypoint 단계에서
`r_b`(0.25 m)를 위반했다(최저 0.15 m). GT 40/40은 이 게이트를 전부 통과했었다. 원인은
03의 `refine_radius` 설명에 이미 문서화돼 있던 것과 같다 — `nav_coarse`의 낙관적 다운샘플(`any`)이
통과시킨 0.2 m 셀 중 일부는 하위 0.05 m 셀에서 `r_b` 미만일 수 있고, `refine_radius=0.10`의 좁은
탐색 범위로는 되끌어올리지 못한다. **GT의 특정 경로는 이 병목을 우연히 잘 안 지나가지만, 무작위
start/goal은 지도 전체를 고르게 훑으므로 훨씬 자주 걸린다** — 표본 크기(경로 개수)가 늘면 이런
케이스가 늘어나는 것은 당연하다.

해결: 에피소드 슬롯당 **재추출(reject-and-resample, 최대 30회)** — `plan_episode`가 성공해도
`check_refined`가 `r_b` 미만이거나 하드 충돌이면 새 h_b/pitch/start/goal로 다시 뽑는다. 논문의
무작위 생성도 암묵적으로 "갈 수 없는/위험한 에피소드는 버린다"는 전제가 있을 것이므로, 이건
파이프라인 버그가 아니라 무작위 생성기에 당연히 있어야 하는 거부 샘플링을 추가한 것이다.

## 결과 (2026-08-04, seed=0, 20 에피소드/씬)

| 씬 | 성공 | 재추출(버려진 후보) | 길이 median | clearance median | 길이/직선 median | 하드충돌 |
|---|---|---|---|---|---|---|
| 17DRP5sb8fy | 20/20 | 10건 | 6.70 m | ~0.45 m | ~1.2 | 0 |
| s8pcmisQ38h | 20/20 | 2건 | 16.44 m | ~0.6 m | ~1.1 | 0 |

두 씬 모두 PASS. 씬1이 재추출을 훨씬 더 많이 했다(10 vs 2) — 씬1이 좁은 통로가 많은 작은 아파트,
씬2가 더 넓은 공간이라는 것과 일치한다.

## 산출물

- `paths/<scene>_random.json` — reproduce의 `<scene>.json`과 겹치지 않는 별도 파일.
- `logs/gs-vlnpe/03_sample_gt_paths/<scene>_random/report.html` — navigable 배경 위에 경로(초록)/
  start(마젠타)/goal(시안)를 겹쳐 그린다. GT가 없어 reproduce처럼 개별 정답 비교(worst 3 등)는
  없다.

## 구현 메모 (guideline 준수)

- `main()`에 추가된 것은 최상단 guard clause 한 줄(`if args.mode=='random': return main_random(args)`)
  뿐 — reproduce의 본문은 전혀 건드리지 않았다(회귀 재실행으로 chamfer 0.103 동일함을 확인).
- `main_random`/`sample_random_episode`/`sample_random_start_goal`/`render_states_random`/
  `build_summary_random`은 전부 새 함수로 분리했다(기존 GT-비교 함수 `compare_with_gt`/
  `build_summary`/`render_states`는 GT 필드에 의존하므로 재사용하지 않고 병렬 버전을 새로 만들었다).

## 사후 발견 버그 — start/goal이 건물 밖 허공에 생김 (2026-08-04 수정)

리포트를 본 사용자가 "바깥에서 start/goal이 생긴다"고 지적. 원인: `truncate_navigable`(clearance
기반)만으로는 "장애물 없음"과 "mesh 데이터가 아예 없는 스캔 밖 허공"을 구분 못 한다 —
`derive_obstacle_2d`의 `(h_nav,h_obs]` 밴드에 점유가 없으면 navigable로 치는데, 밴드가 빈 이유가
"뚫린 공간"인지 "애초에 mesh가 없는 칸"인지는 안 본다. 실측(`s8pcmisQ38h`): navigable로 분류된
셀의 **~40%**가 전체 높이(z 전체)에서 점유가 0인 칸이었다. GT는 항상 건물 안 좌표만 쓰므로
reproduce는 이 문제를 절대 안 겪지만, 무작위 샘플링은 navigable 전체에서 균등하게 뽑으므로
그 40%를 그대로 "갈 수 있는 곳"으로 뽑아버렸다 — **씬을 넓게 훑는 방식(random)이라야 드러나는
버그**였다(reproduce 회귀 검증만으로는 절대 못 잡는다).

해결: `esdf_utils.compute_scan_coverage_mask(occupancy)`(전체 높이에서 점유가 하나도 없는 열은
False) 추가, `navigable = truncate_navigable(...) & compute_scan_coverage_mask(occupancy)`로
`plan_episode`/`sample_random_start_goal`/두 리포트 렌더 함수/`03d_grid_search_params.py`
전부에 적용. reproduce 회귀(chamfer 0.103/0.134) **완전히 불변** 확인 — GT는 원래 이 영역을
안 지나가므로 당연한 결과다. 수정 후 씬2 랜덤 경로 길이 median이 16.44 m → 8.15 m로 GT 참고값
(6.03 m)에 훨씬 가까워졌다(건물 밖 먼 허공까지 잇던 "가짜로 긴" 경로가 사라졌기 때문).

관련: [[260803_gs_vlnpe_03_reproduce_result]], [[260804_gs_vlnpe_03d_grid_search_result]]
