# nav3d 폴더·이름 정리 + 처음부터 일부 에피소드 검증 (2026-10-24)

리포트: https://claude.ai/artifact/KXZKxP9R5qPZ5Hvh3Fe6zn (비공개, Share 메뉴로 공유). 구조·사용법은 `scripts/dataset_converters/nav3d/README.md`,
정리 전 기록은 같은 폴더 `HISTORY.md` (옛 이름 그대로).

## 이전과 달라진 것
- 폴더: `vlnce/ vlnpe/ noeun/` (단계) · `common/` (공용 단계 코드) · `utils/` (모듈) · `checks/` (검사) · `*/tools/` (파이프라인 밖) · `*_worker.py` (보조).
- 데이터셋 폴더 파일 = `<단계><순서>_<하는 일>.py`. 번호 앞자리: 1 준비 · 2 경로 · 3 경로 보정 · 4 걷기 · 5 렌더 · 6 라벨 · 7 검사.
- 연결 파일 (vlnce 11·21·61·71, vlnpe 11·21·71, noeun 21·71·72) = common/checks 스크립트 + 고정 인자 (`TARGET` · `ARGS`).
- `run_pipeline.py`: `--stage` · `--max_episodes` · `--episodes_json` · `--stairs_episode_ids` · `--walk_workers` 추가.
- 실제 `logs/` 도 같은 구조로 이동 (59 항목, 예 `06k_rbq_walk` → `noeun/walk`, `04_export_paths/noeun_station` → `noeun/paths`,
  `05_vlnce_gt_render/episodes.json` → `common/episodes.json`, 임시 스크립트 → `_scratch/`).

## 검증 (4faf919 vs 지금, 빈 logs 에서 처음부터)
- 범위: vlnce·vlnpe 2 씬 5 ep (라벨 2, H1 2 ep), noeun 경로 20 · RBQ 20 ep · 렌더 3 ep. 지금 코드는 run_pipeline 으로, 정리 전은 같은 명령을 옛 이름으로.
- vlnce: 11·21·51 (이미지 3,970)·61 (라벨 문장)·71 전부 같음.
- vlnpe: 51~53·55 준비·71·53 판정 같음. 54 rgb 평균 0.25/255 (같은 코드 두 번 0.275) · 55 H1 NE 흔들림 (성공/실패는 3 회 같음) = 실행마다.
- noeun: 11~45 · RBQ 걷기 프레임 · 51 vln_ce 이미지 1,072 · 71 · 72 같음. 51 vln_pe rgb 0.29/255 = 실행마다. RBQ 걷기는 실행마다 같다.
- 43 은 모두 성공이라 양쪽에 같은 2 ep 를 실패로 표시하고 검증.
- H1 sg ep85 NE 정리 전 5.41 vs 지금 8.34/8.15 가 눈에 띄어 번갈아 3 회씩 더: 정리 전 8.35·8.29·8.36, 지금 8.31·8.38·8.30 (모두 넘어짐,
  스텝 21~66). 처음 5.41 은 정리 전 코드 자신의 드문 값 → 코드 차이 아님.

## 주의 / 아직
- `--overwrite` 로는 렌더(vlnce 51 · vlnpe 54 · noeun 51) 재실행 안 됨 — 스크립트가 출력 폴더가 있으면 멈춘다.
- 노은역 usdz 기본 경로 `data/GS_USDZ/Subway/` 가 data 마운트에 없음 (검증은 `/ws/assets/GS_USDZ/` 링크). H1 걷기는 `data/sample_episodes/` 쓰기 필요.
- 깨끗한 isaac 이미지에는 scikit-image 를 설치해야 렌더가 돈다.
- 노은역 RBQ v6 라벨 불가 (라벨 단계가 읽는 자세 값을 51 렌더가 안 남김) · episodes.json 생성 스크립트 없음 — 그대로.
