# Codex 03 — 노은역 중간층 20개 경로 GT

## 목적

02의 occupancy/ESDF에서 원본 `03_sample_gt_paths.py --mode random` 로직 그대로 안전한 start/goal과 경로 GT 20개를 만든다.

## 실행

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py --scene noeun_station_mid --mode random --num_episodes 20 --esdf_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex/esdf --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

## 실제 결과

- 판정: PASS, 20/20
- seed: 0
- 경로 길이 중앙값: 14.41m
- 길이 범위: 2.26~31.90m
- 하드 충돌/robot radius 위반 후보 1개는 원본 재추출 로직으로 폐기 후 대체됐다.
- 출력: `noeun_pipeline_codex/paths/noeun_station_mid_random.json`
- report: `logs/gs-vlnpe/codex_noeun/03_sample_gt_paths/noeun_station_mid_random/report.html`

리포트의 “분포 참고범위 밖”은 원본 Matterport 통계와 노은역 무작위 경로 분포가 다르다는 참고 경고이며, 안전성 게이트 실패가 아니다.
