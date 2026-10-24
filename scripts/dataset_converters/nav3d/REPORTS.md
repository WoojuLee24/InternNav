# nav3d 리포트 모음

리포트가 있는 곳을 한 곳에 모은다. 실행 명령은 [COMMANDS.md](COMMANDS.md), 구조는 [README.md](README.md).
2026-10-24 이전에 만든 리포트는 **옛 스크립트 이름**(예: 04d, 06k)을 쓴다 — 새 이름은 README 의 '옛 이름 → 새 이름' 표.

## 1. 게시된 리포트 (claude.ai, 비공개 — 다른 사람은 페이지의 Share 메뉴로 공유해야 열린다)
| 리포트 | 날짜 | 내용 | 링크 |
|---|---|---|---|
| nav3d 정리 검증 | 2026-10-24 | 폴더·이름 정리 뒤 vlnce · vlnpe · noeun 을 처음부터 일부 에피소드로 돌려 정리 전 코드와 비교 | https://claude.ai/artifact/KXZKxP9R5qPZ5Hvh3Fe6zn |
| VLN 파이프라인 ① 환경과 경로 데이터셋 | 2026-10-21 | 씬 확보 · 환경 · 경로 생성 · 렌더 | https://claude.ai/artifact/F8picTt7BhSRXLNUUXAWZ2 |
| VLN 파이프라인 ② labeling 학습 평가 | 2026-10-21 | instruction labeling · 데이터셋 · 학습 · 평가 | https://claude.ai/artifact/HgBoKBGedShctw89YWnbSn |
| VLN 파이프라인 ③ 재현과 relabel | 2026-10-21 | full 재현 · relabel 2x2 | https://claude.ai/artifact/S51KpgTbDm4swoTb6eSiSN |
| 경로 생성 알고리즘 상세 | 2026-10-21 | 3D navmesh (find_path · 탄성 밴드 · Chaikin) vs 2D (ESDF · A*) | https://claude.ai/artifact/RoCgoUtSYudmDwUAXvEEVw |
| 노은역 경로 선택 (옛 04 = `noeun/22_select_paths`) | 2026-10-16 | 층 이동 100 + 평지 100 · U1~U5 | https://claude.ai/artifact/EJvqTfBEoGTj9dGq7RfHrb |
| 노은역 카메라 렌더 검사 (옛 06b = `checks/noeun_camera_render_check`) | 2026-10-16 | R1~R6 | https://claude.ai/artifact/MbHmZctMqYj1Qs1dqrpYYc |
| vln_pe GT 로 렌더 방식 검증 (옛 06v = `vlnpe/52_camera_values`) | 2026-10-16 | 카메라 높이 · 자세 | https://claude.ai/artifact/9sEPsQfeSSCfLTwhZya2me |
| H1 궤적 vs 경로 (옛 06h = `vlnpe/51_h1_pairs`) | 2026-10-17 | 209 ep · 60 씬 | https://claude.ai/artifact/Kz65AVVAespSgFahj5fz9L |
| vln_pe 카메라 방식 MP3D (옛 06q = `vlnpe/53_camera`) | 2026-10-18 | 4 씬 28 ep | https://claude.ai/artifact/LY5LSiHYe5AtyfCEjGqoq9 |
| H1 물리 걷기 (옛 06r = `vlnpe/55_walk_h1`) | 2026-10-19 | MP3D 4 씬 28 ep · GT vs sg | https://claude.ai/artifact/Ajhy9xAPGyad8Mc13adRnM |

## 2. 단계별 report.html (실행하면 그 단계가 다시 만든다)
경로는 `scripts/dataset_converters/nav3d/logs/` 아래. full 로 돌리면 `logs/full/` 아래 같은 위치.

| 데이터셋 | 단계 | report.html |
|---|---|---|
| 공용 | 11 GT 비교 | `common/compare_vlnce_gt/report.html` |
| 공용 | 21 경로 다듬기 | `common/refine_paths/report.html` |
| 공용 | 61 라벨 | `common/label_paths/report.html` |
| vlnce | 51 렌더 | `vlnce/51_render/report.html` |
| vlnpe | 51 · 52 · 53 · 55 | `vlnpe/51_h1_pairs/` · `vlnpe/52_camera_values/` · `vlnpe/53_camera/` · `vlnpe/55_walk_h1/` 의 `report.html` |
| noeun | 11 자산 검사 · 12 navmesh | `noeun/11_check_assets/noeun_station/report.html` · `noeun/12_build_navmesh/noeun_station/report.html` |
| noeun | 22 경로 선택 | `noeun/paths/report.html` |
| 검사 | 계단 라벨 · 계단 방식 비교 · 2D 비교 | `checks/refine_stairs_label_check/` · `checks/refine_stairs_mode_compare/{split,none}/` · `checks/refine_vs_2d_compare/` |
| 검사 | vln_ce GT 재렌더 (habitat · Isaac) | `checks/vlnce_gt_rerender_check/check/` · `checks/vlnce_isaac_rerender_check/check/` |

검사 스크립트 중 json 만 쓰는 것(노은역 렌더 검사 · 로더 검사 등)은 `checks/<이름>*/summary.json`.

## 3. 결과 노트 (저장소 `.claude/memory/`)
| 파일 | 내용 |
|---|---|
| `261024_nav3d_vlnce_full_mini_result.md` | vlnce 기본 = 학습 데이터와 같게 · mini / full · 검증 (이 문서와 같은 날) |
| `261024_nav3d_restructure_verify_result.md` | 폴더·이름 정리 + 처음부터 검증 |
| `261023_nav3d_pipeline_refactor_result.md` | 실행기 · checks/ · 공용 모듈 (정리 1 차) |
| `261021_pipeline_report_result.md` · `261021_reproduce_relabel_report_result.md` | 파이프라인 리포트 ① ② ③ 요약 |
| `261016_nav3d_06_noeun_render_result.md` | 노은역 렌더 · RBQ 걷기 v3 ~ v6 · vln_pe H1 진행 기록 |
| `261016_nav3d_07_labeling_smoke_result.md` | 라벨 연결 smoke |
| `261015_nav3d_05_vlnce_render_result.md` | vln_ce 렌더 (옛 05) |
| `261014_nav3d_03_refine_result.md` · `261014_nav3d_00_mesh_gs_check_result.md` | 경로 다듬기 · 자산 검사 |
| `261013_nav3d_02_vlnce_gt_result.md` · `261013_nav3d_01_noeun_navmesh_result.md` | GT 비교 · 노은역 navmesh |

nav3d 폴더의 [HISTORY.md](HISTORY.md) 는 2026-10-24 이전 README (단계별 실험 기록과 수치, 옛 이름).
