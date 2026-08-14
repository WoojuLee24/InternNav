# Codex 01 — 노은역 중간층 scene_meta 생성

## 목적

원본 `01_prepare_scene.py`의 씬 사용 가능 여부 판정과 `scene_meta` 계약을 노은역 USDZ 중간층에 적용한다.

## 중간층 선정

- 검증된 기준 높이: `floor_z=-0.05m`
- 법선이 위를 향하고 수평에서 15도 이내인 면을 사용한다.
- 중심 높이가 floor_z 주변 ±0.45m인 면만 고른다.
- 선택 면의 XY bounds에 1m padding, Z는 floor 아래 0.2m부터 위 1.6m까지 논리 ROI로 둔다.
- 원본 USDZ는 수정하지 않는다.

## 최소 변경

- 원본 파일에 `--usd_path` 분기를 추가했다.
- 거부된 별도 `01_prepare_usdz_scene.py`는 삭제했다.
- 단순 JSON 출력이 아니라 원본 gallery 형식의 판정 요약, 중간층 ROI 이미지, 상세 메타를 `report.html`로 만든다.

## 실행과 결과

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/01_prepare_scene.py --scene noeun_station_mid --usd_path data/noeun_station/noeun_station_collision.usdz --floor_z -0.05 --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

- 판정: PASS
- 상향 바닥 후보: 14,618 faces
- 선택 면적: 798.2m²
- 기존 분석값과 floor_z 및 영역이 일치했다.
- scene_meta: `noeun_pipeline_codex/scene_meta/noeun_station_mid.json`
- report: `logs/gs-vlnpe/codex_noeun/01_prepare_scene/noeun_station_mid/report.html`
