# Codex 02 — 노은역 중간층 occupancy/ESDF

## 목적

원본 `02_build_freemap_esdf.py`를 이용해 01의 논리 ROI 안에서만 노은역 USDZ를 voxelize하고 freemap/ESDF를 만든다.

## 발견한 문제와 수정

- 기존 수정은 USD 전체에 surface sample을 뿌린 후 ROI를 자르는 구조여서 거대한 원본 씬 크기에 따라 ROI 표본 밀도가 달라졌다.
- ROI와 triangle bounds가 겹치는 face를 먼저 submesh로 만든 뒤 샘플링하도록 `esdf_utils.voxelize_surface(..., bounds=...)`를 보완했다.
- 첫 실행은 `pxr` import 전에 SimulationApp이 초기화되지 않아 실패했다.
- 앱을 mesh 로드 직후 닫자 Isaac fast shutdown이 프로세스를 끝내 산출물이 없는 가짜 exit 0이 발생했다.
- 최종적으로 USD를 쓰는 02 프로세스 수명 동안 SimulationApp을 유지하도록 최소 수정했다.

## 실행

`/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/02_build_freemap_esdf.py --scene noeun_station_mid --geometry usd --ref_h_b 0.875 --scene_meta_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --out_dir scripts/dataset_converters/gs_vlnpe/noeun_pipeline_codex --log_dir logs/gs-vlnpe/codex_noeun`

## 실제 결과

- grid: 1345×1308×40, voxel 0.05m
- origin: [-33.4, -26.0, -0.4]
- occupancy: 1.013%
- obstacle cell zero ESDF, free cell positive ESDF, bounds sanity 모두 PASS
- navigable fraction: 0.87964
- ESDF: 3.0MB
- report: `logs/gs-vlnpe/codex_noeun/02_build_freemap_esdf/noeun_station_mid/report.html`

`UNVERIFIED` 표시는 실패가 아니다. 03 이전에는 외부 episode가 없어 원본의 GT-clearance 항목을 건너뛰었다는 뜻이며 geometry sanity는 통과했다.
