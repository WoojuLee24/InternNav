# gs_vlnpe 파이프라인 전체 리포트 링크 모음 (00~04)

각 스크립트가 생성한 `report.html`의 Artifact 링크 + 로컬 경로. 2026-08-06 기준 확인
(로컬 파일 존재 + Artifact 발행 목록 대조, `<title>` 태그로 정확히 매칭). 누락돼 있던 4개
(02/03/03b/03c의 s8pcmisQ38h 버전)는 이번에 새로 발행함.

## 00_inspect_vln_n1.py

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy (ep0) | https://claude.ai/code/artifact/de729add-5777-444e-8491-d4ae1dc966f4 | `logs/gs-vlnpe/00_inspect_vln_n1/17DRP5sb8fy/episode_000000/report.html` |

## 01_prepare_scene.py

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy | https://claude.ai/code/artifact/e7f57d22-bfe9-449f-bd82-ce416c245013 | `logs/gs-vlnpe/01_prepare_scene/17DRP5sb8fy/report.html` |
| s8pcmisQ38h | https://claude.ai/code/artifact/7cb35166-ff60-4530-a540-16ab31a3d316 | `logs/gs-vlnpe/01_prepare_scene/s8pcmisQ38h/report.html` |

## 02_build_freemap_esdf.py

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy | https://claude.ai/code/artifact/7c1bc45b-7fd0-4813-b666-6438fce6b9d7 | `logs/gs-vlnpe/02_build_freemap_esdf/17DRP5sb8fy/report.html` |
| s8pcmisQ38h | https://claude.ai/code/artifact/d84e872f-dacf-40f5-a93c-43e304faed70 | `logs/gs-vlnpe/02_build_freemap_esdf/s8pcmisQ38h/report.html` |

## 03_sample_gt_paths.py

| 씬 | 모드 | Artifact 링크 | 로컬 경로 |
|---|---|---|---|
| 17DRP5sb8fy | reproduce/cubic | https://claude.ai/code/artifact/1bea4e9a-997c-4f43-9fdf-2920f5596a8f | `logs/gs-vlnpe/03_sample_gt_paths/17DRP5sb8fy/report.html` |
| 17DRP5sb8fy | random/cubic | https://claude.ai/code/artifact/4739e7ea-7a72-4c8f-9278-567a97e936f4 | `logs/gs-vlnpe/03_sample_gt_paths/17DRP5sb8fy_random/report.html` |
| 17DRP5sb8fy | reproduce/bezier(구버전, cubic으로 대체됨) | https://claude.ai/code/artifact/f69a22ab-a400-400f-8d66-c7f5f9f9ad4d | (스크래치, 로컬 파일 없음) |
| s8pcmisQ38h | reproduce/cubic | https://claude.ai/code/artifact/5c986fb2-9a23-4287-84be-aed4b03ee38f | `logs/gs-vlnpe/03_sample_gt_paths/s8pcmisQ38h/report.html` |
| s8pcmisQ38h | random/cubic | https://claude.ai/code/artifact/94d32ebb-7d7c-4f97-9f04-12666b38c2ab | `logs/gs-vlnpe/03_sample_gt_paths/s8pcmisQ38h_random/report.html` |

## 03b_verify_reproduction.py

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy | https://claude.ai/code/artifact/22087e26-59f1-4b69-be00-911e921d98ff | `logs/gs-vlnpe/03b_verify_reproduction/17DRP5sb8fy/report.html` |
| s8pcmisQ38h | https://claude.ai/code/artifact/4db6b895-57c7-4cf2-8abd-1062e321092d | `logs/gs-vlnpe/03b_verify_reproduction/s8pcmisQ38h/report.html` |

## 03c_compare_refine.py

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy | https://claude.ai/code/artifact/e69d44bf-f3a1-4fdc-b8d8-afdc3440dd2c | `logs/gs-vlnpe/03c_compare_refine/17DRP5sb8fy/report.html` |
| s8pcmisQ38h | https://claude.ai/code/artifact/77f179d1-2378-4e60-894a-f6b12f90d1df | `logs/gs-vlnpe/03c_compare_refine/s8pcmisQ38h/report.html` |

## 03d_grid_search_params.py

| 내용 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 8축 그리드서치(17DRP5sb8fy) | https://claude.ai/code/artifact/3a07b6b1-51a2-45fe-aae3-f559535a85ee | `logs/gs-vlnpe/03d_grid_search_params/report.html` |

## 04_render_obs.py (Open3D 렌더러)

| 씬 | 모드 | Artifact 링크 | 로컬 경로 |
|---|---|---|---|
| 17DRP5sb8fy | gt_replay 2-A | https://claude.ai/code/artifact/0621a082-76fa-423f-a48d-8ae3e0309a15 | `logs/gs-vlnpe/04_render_obs/17DRP5sb8fy/report.html` |
| 17DRP5sb8fy | reproduce 2-B | https://claude.ai/code/artifact/d681642b-bc3b-420f-84dd-8796131a69b7 | `logs/gs-vlnpe/04_render_obs/17DRP5sb8fy_reproduce/report.html` |
| 17DRP5sb8fy | random 2-B | https://claude.ai/code/artifact/c586d7ae-23a8-41e8-ba89-9b08db8b0931 | `logs/gs-vlnpe/04_render_obs/17DRP5sb8fy_random/report.html` |
| s8pcmisQ38h | gt_replay 2-A | https://claude.ai/code/artifact/a5a5ba52-fc5e-4aa7-abc1-0d31620dc019 | `logs/gs-vlnpe/04_render_obs/s8pcmisQ38h/report.html` |
| s8pcmisQ38h | reproduce 2-B | https://claude.ai/code/artifact/c3663deb-1aab-48e2-9c74-69ec7f12dc72 | `logs/gs-vlnpe/04_render_obs/s8pcmisQ38h_reproduce/report.html` |
| s8pcmisQ38h | random 2-B | https://claude.ai/code/artifact/ef656f54-2be6-4e24-b0cd-08d3279fb8de | `logs/gs-vlnpe/04_render_obs/s8pcmisQ38h_random/report.html` |

## 04_render_obs_isaac.py (Isaac Sim 렌더러, M1.4 최종)

| 씬 | 모드 | Artifact 링크 | 로컬 경로 |
|---|---|---|---|
| 17DRP5sb8fy | gt_replay 2-A (SSIM 반영) | https://claude.ai/code/artifact/847c3297-c562-4ba3-8821-669dfcf53b28 | `logs/gs-vlnpe/04_render_obs_isaac/17DRP5sb8fy/report.html` |
| 17DRP5sb8fy | reproduce 2-B | https://claude.ai/code/artifact/f490e7f6-e65b-4fdd-8052-1a407e2b3e50 | `logs/gs-vlnpe/04_render_obs_isaac/17DRP5sb8fy_reproduce/report.html` |
| s8pcmisQ38h | gt_replay 2-A (SSIM 반영) | https://claude.ai/code/artifact/a0eeab7c-b5ba-45fa-819a-3191ec5b0149 | `logs/gs-vlnpe/04_render_obs_isaac/s8pcmisQ38h/report.html` |
| s8pcmisQ38h | reproduce 2-B | https://claude.ai/code/artifact/8ec9d496-befd-4325-a548-a57d52c7f034 | `logs/gs-vlnpe/04_render_obs_isaac/s8pcmisQ38h_reproduce/report.html` |

## 04c_light_explorer.py (조명 옵션 비교 도구)

| 씬 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| 17DRP5sb8fy (9개 조명 옵션) | https://claude.ai/code/artifact/b73b8323-304f-4564-b9b1-cbde33a25b9a | `logs/gs-vlnpe/04c_light_explorer/17DRP5sb8fy/report.html` |

## 그 외 임시 비교 리포트 (특정 스크립트 산출물 아님, 조사 중 만든 것)

| 내용 | Artifact 링크 | 로컬 경로 |
|---|---|---|
| pose 규약 변경 전/후 비교 | https://claude.ai/code/artifact/6446f9e3-a51d-479b-a80e-bc67abffb08b | `logs/gs-vlnpe/_convention_compare/17DRP5sb8fy/report.html` |
| h_nav 0.15→0.10 경로 비교 | https://claude.ai/code/artifact/f7541d43-2020-4362-a50b-77313573485f | `logs/gs-vlnpe/_hnav_compare/17DRP5sb8fy/report.html` |

---
상세 경위·수치는 각 단계별 `.claude/memory/*_result.md` 참고(예:
`260805_gs_vlnpe_04_render_obs_isaac_result.md`).
