# nav3d vlnce: 기본 = 학습 데이터와 같게 · mini / full (2026-10-24)

명령: `scripts/dataset_converters/nav3d/COMMANDS.md` · 리포트 위치: `scripts/dataset_converters/nav3d/REPORTS.md`.

## 이전과 달라진 것 (vlnce 만, vlnpe 는 나중)
| | 전 | 후 |
|---|---|---|
| 에피소드 고르기 | `logs/common/episodes.json` (4 씬 29 ep 시험 표본, 만든 스크립트 없음) 이 필수 | 목록 없이 릴리스 학습 데이터의 모든 경로. 목록은 `--episodes_json` 로 일부만 고를 때 |
| 21 평지 경로 | 씬마다 10 개 (`--max_flat 10`) | 실행기가 `--max_flat 0` (전부). 테스트는 `--max_episodes N` |
| 에피소드 단위 | 경로마다 하나 (첫 지시문) | 릴리스처럼 지시문마다 하나 (`51 --instructions all`, 경로는 한 번 렌더). 이전 = `--instructions first` |
| 데이터셋 | mini 고정 | 실행기 `--data mini|full` (기본 mini). full = `data/InternData-N1`, logs 는 `logs/full/` |
| 라벨 대상 | 목록의 ep, sibling 0 | 51 이 내보낸 에피소드 전부 (`_index`), sib_idx = 같은 경로 안 지시문 순서 0 · 1 · 2 |

새 인자: 21 `--raw_root` · 51 `--instructions` `--data_root` `--scenes` (`--episodes_json` 기본 빈값) · 61 `--data_root` (`--episodes_json` 기본 빈값).
새 파일 `utils/data_roots.py`. 공용 모듈 폴더는 `lib/` 였다가 `utils/` 로 바꿨다 (.gitignore 의 `lib/` 규칙이 새 파일을 무시해서, 2026-10-24).

## 검증 (mini, 빈 logs 에서, 검증 사본)
| | 확인 | 결과 |
|---|---|---|
| T1 이전 동작 | 목록 + `--instructions first` + `--max_flat 10` vs 정리 전 코드(4faf919) 출력 | 렌더 데이터 60 파일 · 이미지 3,970 · 11 · 21 결과 전부 같음 |
| T2 새 기본 | 실행기 `vlnce --stage 1,2,5,7 --scenes 17DRP5sb8fy,1LXtFkjw3qL --max_episodes 3` (167 s) | wp · sg 각 57 ep = 그 경로들의 릴리스 ep 목록과 정확히 같음 · 지시문 = 릴리스 · 같은 경로 3 ep 는 프레임·action·자세 같음 · P1~P5 PASS · 로더 PASS |
| T3 라벨 | `--only label --max_episodes 6` | 생성됨. 같은 경로 3 지시문이 거의 같은 문장 (b1 `style.sibling: none`) — 사용자 결정: 그대로 (labeling 은 만드는 중) |

## 라벨 단계 VLM 서버 자동 (`run_pipeline --vlm_server auto`, 기본)
- 안 떠 있으면 실행기가 `common/vlm_server.py` 를 띄우고 응답을 기다린 뒤 라벨링, 끝나면(실패해도) 내린다. 이미 떠 있던 서버는 쓰고 안 내린다.
- 확인: (A) 서버 없음 → 띄움 · 새 라벨 6 개 생성(오류 0) · 내림, (B) 미리 띄운 서버 → 그대로 두고 사용, (C) 라벨 단계 실패 → 그래도 내림.
  끝난 뒤 프로세스 0 · 포트 8200 응답 없음 · GPU 사용 없음.

## H200 pod 에서 막힘 → 수정
- 막힌 점 (pod 보고): 21 이 `gs_vlnpe/logs/scene_meta/<씬>.json` 의 mesh_path 만 쓰는데 그 파일을 만드는 gs_vlnpe 01 은 pxr 필요 · 11 · 21 의 `--scene_root` 기본값 `data/scene_data/...` 는 이 PC 에만 있는 링크.
- 수정: 21 `--mesh_root` (주면 `geometry_utils.find_scene_mesh(<루트>, 씬)`, 안 주면 이전처럼 scene_meta). 실행기는 mini · full 모두 데이터 루트 기준 경로를 넘긴다.
- 확인: 이 PC 61 씬에서 find_scene_mesh(mini/scene_data/mp3d_n1) = scene_meta mesh_path (같은 파일 61/61).
  `data/scene_data` 링크와 gs_vlnpe logs 를 둘 다 뺀 사본에서 vlnce 1·2·5·7 단계 (2 씬) → 렌더 데이터 248 · 이미지 35,784 · 11 · 21 결과가 이전과 전부 같음.

## 남은 것
- 렌더된 데이터의 지시문은 릴리스 사람 지시문(GT 경로 설명) — 라벨을 데이터에 넣는 단계는 아직 없다 (labeling 진행 중).
- full 은 이 PC 에 없어 `--dry_run` 경로까지만 확인. full 씬 구성 미확인 (21 은 gs_vlnpe scene_meta 를 쓴다).
- vlnpe: 학습 데이터와 같게 · mini/full · wp 추가는 나중.
