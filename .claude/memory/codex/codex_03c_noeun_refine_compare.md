# Codex 03c — 노은역 refine 방식 비교

## 원칙

박사님의 `03c_compare_refine.py`가 가진 argmax/min_move 비교, floorplan blink, 표와 `save_gallery`를 그대로 사용했다. 노은역에서는 parquet 대신 03의 20개 경로 GT를 공급한다.

## 결과

- 20개 전부 계획 성공
- argmax chamfer: 0.0000m
- min_move chamfer: 0.0944m
- 두 방식 모두 robot radius: 20/20 통과
- 실제 경로 GT 기준 우세: argmax
- 출력 JSON: `noeun_pipeline_codex/compare/noeun_station_mid.json`
- HTML: `logs/gs-vlnpe/codex_noeun/03c_compare_refine/noeun_station_mid/report.html`
