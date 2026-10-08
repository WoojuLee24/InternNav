# Codex 03d — 노은역 8축 파라미터 그리드서치

## 원칙

박사님의 `03d_grid_search_params.py`가 가진 8축 독립 스윕, 표, line chart, 극값 blink와 `save_gallery`를 그대로 실행했다. 입력만 노은역 ESDF와 03 경로 GT로 연결했다.

## 실행 축

- A* grid
- refine radius
- connectivity
- clearance tie-break weight
- waypoint spacing
- cubic spline output step
- h_nav
- robot radius
- downsample mode

## 결과

- 모든 후보 계산 완료
- 8축 계산 회귀 상태 모두 OK
- 결과 JSON: `noeun_pipeline_codex/gridsearch/results.json`
- HTML: `logs/gs-vlnpe/codex_noeun/03d_grid_search_params/report.html`

노은역 random 모드에서는 Matterport 두 씬의 고정 chamfer 숫자를 강제하지 않는다. 표와 차트에는 노은역 20개 경로 GT 기준 실제 측정값이 기록된다.
