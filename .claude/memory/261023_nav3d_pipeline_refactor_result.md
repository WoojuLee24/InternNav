# nav3d 파이프라인 정리 — 실행기 · checks/ · 공용 모듈 (2026-10-23)

목적: nav3d 스크립트 56 개의 실행 순서가 코드 안(인자 기본값)에 숨어 있던 것을 드러내고, **동작은 그대로** 두기.

## 용어
| 용어 | 뜻 |
|---|---|
| 단계(step) | `run_pipeline.py` 가 부르는 이름 하나 = 기존 번호 스크립트 한 번 호출 (입력·출력 경로를 다 적어서) |
| 대상(target) | `vlnce` (MP3D → vln_ce) · `vlnpe` (MP3D → vln_pe, 카메라 배치 / H1 걷기) · `noeun` (노은역 RBQ v6) |
| 다시 노출 | 함수를 공용 모듈로 옮긴 뒤 원래 파일이 같은 이름을 import 해 둔 것 — 원래 파일에서 꺼내 쓰던 코드도 그대로 돈다 |

## 이전과 달라진 것
| | 전 | 후 |
|---|---|---|
| 실행 순서 | README 표(10-16 에서 멈춤) + 인자 기본값 사슬 (04d → 04f → 04g 가 이전 버전 파일을 기본값으로) | `run_pipeline.py <대상> --list` · 단계별 입력·출력 명시 |
| 검증 스크립트 | 생산 스크립트와 같은 폴더 | `nav3d/checks/` 21 개 (출력을 다른 단계가 읽는 00 · 02 · 06h · 06v 는 최상위에 둠) |
| 번호 파일끼리 함수 공유 | importlib 로 `03_refine_paths.py` 등을 실행해서 꺼냄 | 공용 모듈 `nav3d_scenes` · `path_quality` · `mesh_probe` · `walk_inputs` · `render_poses` (코드 그대로 옮김) |
| 첫 걷기 입력 `eps_noeun_all_v2.json` | 만든 스크립트 없음 | `06k_make_walk_inputs.py` (바이트 단위 재현) |
| 파일 이름 | — | 그대로 (사용자 결정) |

gs_vlnpe 번호 파일(`05_export_vlnce.py` · `04_render_obs_isaac.py` 등)을 importlib 로 부르는 것은 그대로 (다른 세션 소유).

## 명령
```
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --list
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --dry_run
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --from floor_to_mesh --to straighten_stairs --overwrite
```
| 인자 | 뜻 | 왜 |
|---|---|---|
| `--only` · `--from` · `--to` | 돌릴 단계 | 한 단계 / 구간 |
| `--overwrite` | 출력이 있어도 돌림 | 기본은 멈춤 (지금 결과 보호) |
| `--dry_run` | 명령만 출력 | 개별 실행 명령 확인 |
| `--scenes` · `--jobs` | 02 · 03 씬 · 동시 수 | 기본 61 씬 · 4 |
| `--label_method` · `--label_model` | 07 | 기본 b1 · qwen3vl8b-local |

컨테이너가 다른 단계에 닿으면 멈추고 `--from <단계>` 이어 돌리기 명령을 출력한다.
habitat = `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` (torch2.9.1 이미지에는 habitat_sim 이 없다).

## 검증 (이전 = HEAD 4faf919 스냅샷, 같은 입력·같은 출력 경로)
| 검사 | 결과 |
|---|---|
| 옮긴 함수 13 개를 실제 데이터로 이전 ↔ 새 호출 (06 자세 400 · 06m/06p 200 walk · 04c/04d 60 ep) | 전부 비트 단위 같음 |
| CLI 04c · 04d gen/eps/merge · 04f · 04g 이전 ↔ 새 | 출력 json 전부 같음 |
| CLI 03 (노은역, 865 s) · 04 (1,400 s) 이전 ↔ 새 | 03 같음 · 04 에피소드 200/200 같음 (다른 칸은 스크립트 위치 절대경로 `navmesh` 하나 — 스냅샷 위치 차이). 새 04 출력은 기존 logs 파일과 전부 같음 |
| `run_pipeline.py` 로 habitat 단계 9 개 (04b ~ 04g, 걷기 결과는 기존 것) → 기존 logs 의 13 파일 | 메타데이터까지 전부 같음 (5 분) |
| `06k_make_walk_inputs.py` → `eps_noeun_all_v2.json` | 바이트 같음 |
| 모든 스크립트 `--help` (habitat · isaac 컨테이너) | 같음 (Isaac 없는 쪽은 같은 자리에서 같은 오류) |
| 06m 1 ep 렌더 (Isaac) vlnce | parquet · json · 이미지 전부 같음 |
| 06m 1 ep 렌더 vlnpe | parquet · depth · meta 같음. rgb 는 새 코드 첫 실행만 평균 0.25/255 다름 — **새 코드 두 번째 실행은 이전 코드와 rgb 까지 같음** → 렌더러 세션 차이 |
| pyflakes 이전 ↔ 새 | 새 미정의 이름 0 (남은 경고 = 다시 노출 import) |

검증 컨테이너에는 Isaac 이미지에 `scikit-image` 를 따로 설치해야 06m 이 돌았다 (실제 쓰던 `internnav-isaac` 컨테이너엔 설치돼 있던 것으로 보임).

## 아직 안 되는 것 / 주의
- **노은역 RBQ v6 데이터에 b1 라벨을 붙일 수 없다**: `07 --sources noeun` 은 06 의 `frames_meta.json`(카메라 자세·몸 높이·yaw)을 읽는데 06m 은 `render_meta.json` 에 그 값을 남기지 않는다.
- `logs/05_vlnce_gt_render/episodes.json` (05b · 06q · 06r · 07 입력) 을 만든 스크립트가 없다.
- 걷기(06k · 06r)는 물리라 다시 돌리면 결과가 조금 다를 수 있다 → 04d 가 바꿀 ep 가 달라질 수 있음. `--episode_ids 86` 은 사람이 고른 값.
- 지금 v6 데이터는 v5 걷기 + 23 ep 다시 걷기를 합치고 06m2 로 23 ep 만 다시 렌더한 것. `run_pipeline.py` 는 200 ep 를 처음부터 한다.
- `logs/_rw_sg.py` · `_rw_diag.py` (추적 안 되는 임시 스크립트) 는 `nav3d/07r_route_words_check.py` 경로를 쓰는데 이제 `checks/` 아래다.
