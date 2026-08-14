# Codex LiDAR 조사 05 — 사람이 확인하는 HTML 리포트

작성일: 2026-08-14 (UTC)

## 요청

JSON 수치만 보는 대신 기존 00~04의 `report.html`처럼 사진, 그래프, 구체적인 수치와 PASS gate를 결과 폴더에서 바로 확인할 수 있게 한다.

## 구현

`scripts/dataset_converters/gs_vlnpe/lidar/generate_lidar_report.py`를 추가했다. 기존 `viz_utils.save_gallery()`와 `blink_widget_html()`을 재사용하므로 00~04와 같은 self-contained HTML 스타일이며 외부 이미지 서버에 의존하지 않는다.

## report 구성

1. 최종 PASS/FAIL, return 수, finite 비율, mesh median/p95를 상단 stat card로 표시
2. 7개 독립 검증 gate를 표로 표시
3. 같은 frame의 04 Isaac RGB와 D455 depth를 blink 방식으로 전환
4. LiDAR world pointcloud 상면도와 측면도를 range 색상으로 표시
5. 전체 return 거리 histogram과 median 표시
6. 원본 USDZ mesh 표면거리 오차 histogram과 median/p95 표시

## 산출물

`scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex/obs/noeun_station_mid_random_isaac_d455_nominal/lidar_rtx/smoke_test/report.html`

HTML은 약 558 KB이며 RGB와 모든 그래프를 base64로 내장한다. 같은 폴더의 `report_assets/`에도 원본 PNG 다섯 장을 별도로 남긴다.

## 표시 수치 교차검증

- status: PASS
- RTX returns: 44,784
- mesh median: 0.735 mm
- mesh p95: 4.699 mm
- HTML의 7개 gate: 모두 PASS

이는 `episode_000000_frame_0000.validation.json`의 값과 일치한다.
