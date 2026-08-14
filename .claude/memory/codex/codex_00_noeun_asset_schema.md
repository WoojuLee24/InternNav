# Codex 00 — 노은역 자산 및 생성 GT 규약 검사

## 목적

원본 `00_inspect_vln_n1.py`의 역할인 입력 규약 확정을 노은역 USDZ에 적용한다. 노은역에서는 USDZ/mesh가 공간 GT이고, 03이 경로 GT, 04가 RGB·depth·camera pose GT를 생성한다.

## 최소 변경

- 별도 실행 파일을 만들지 않고 원본 `00_inspect_vln_n1.py`에 `--usd_path`, `--floor_z` 분기만 추가했다.
- 기존 Matterport/N1 기본 인자와 본문은 유지했다.
- USDZ readable, Z-up, metersPerUnit=1, 유한 bounds, mesh 존재, 중간층 상향면 존재를 검사한다.
- 원본 gallery 형식의 요약 표, 중간층 후보 시각화와 `target_schema.json`을 만든다.

## 실행

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/00_inspect_vln_n1.py --scene noeun_station_mid --usd_path data/noeun_station/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

Conda/mamba는 사용하지 않았다.

## 실제 결과

- 판정: PASS
- schema: `noeun_pipeline_codex/target_schema.json`
- report: `logs/gs-vlnpe/codex_noeun/00_inspect_vln_n1/noeun_station_mid/report.html`
- 리포트에는 실제 중간층 후보 면 시각화가 포함된다.
