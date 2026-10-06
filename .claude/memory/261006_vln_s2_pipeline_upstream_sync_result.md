# 261006 vln-s2-pipeline upstream 동기화 + 버전 현황 + label 품질 체크

## 1. 무엇이 달라졌나

**repo:** `/home/irteam/git/vln-s2-pipeline`
- `origin` = `WoojuLee24/vln-s2-pipeline` (fork)
- `upstream` = `KEMAL-MUDIE/vln-s2-pipeline` (원본, 이번에 추가)

**upstream 반영:** `9a8139f "061026"` (268 파일, 대부분 새 스크립트)를 fast-forward로 받았다.

**로컬 작업 커밋:** `2275855` (origin push 완료)
- `metadata_reproducer.py`: upstream head로 둔다 (v219 + `turn_threshold` 인자). 경로는 `local_paths`를 쓴다.
- `metadata_reproducer_v218.py`, `metadata_reproducer_v219.py`: 버전별로 고정해 둔 생성기
- `generate_v218_all_splits.py`: `VERSION=v218|v219`로 생성기를 고른다. LABEL 기본값은 `<VERSION>_reproduce`.

## 2. 명령어

모두 `cd /home/irteam/git/vln-s2-pipeline`에서 실행한다.

- upstream 업데이트:
  `git stash -u && git merge --ff-only upstream/master && git stash pop`
- v218 데이터 생성 (3 split):
  `python3 generate_v218_all_splits.py`
- v219 데이터 생성:
  `VERSION=v219 python3 generate_v218_all_splits.py`
  - `VERSION`: 생성기 선택 (기본 v218)
  - `--splits val_unseen`: 일부 split만 생성
  - `LABEL=<이름>`: 출력 파일 이름(`{split}_<LABEL>.json.gz`) 지정
- 결과 저장 위치: `/home/irteam/data-vol1/relabel_cache/datasets/`

**검증:** 기존 `*_v218_reproduce` / `*_v219_reproduce`와 비교했다.

| VERSION | val_unseen | val_seen |
|---|---|---|
| v218 | 0/1839 diff | 0/778 diff |
| v219 | 0/1839 diff | 0/778 diff |

v218과 v219끼리는 val_unseen 116개, val_seen 72개가 다르다. train은 비교하지 않았다.

## 3. 버전 현황

스크립트 기준 마지막 버전은 **v264**이고, 버전 스크립트는 총 483개다.

| 계열 | 범위 | 방식 |
|---|---|---|
| `run_gate3_gemma_v*` | v1~v100 (97) | Gemma가 문장 생성 |
| `run_gate4_visual_v*` | v2~v203 (198) | 모든 프레임을 Gemma에 보냄 |
| `generate_v*` | v24~v238 (69) | 규칙 기반 |
| `run_gate4_*` | v240~v251 | 그 밖의 Gemma 계열 |
| `run_gt_visanchor_v*` | v252~v264 (13) | GT 문장 일부 교체 |

- README의 v272(ChronoNav, 66.2%)는 스크립트가 없고, "GT + sibling subs"라서 GT 기반이다.

**버전별로 arg/config를 줘서 실행할 수 없다.** 파일마다 상수가 하드코딩돼 있다.
- `/mnt/nvme0` 경로: 1235곳
- 이전 버전 checkpoint에 의존: 373개 (`outputs/`는 gitignore돼 있어서 이 노드에 없음)
- temperature>0: 384개 (재실행해도 같은 문장이 안 나옴)
- 계열 간 번호 충돌: 92쌍
- GT 폴더에 직접 쓰는 `DEPLOY_PATH`가 있음

**재현 가능성**
- 규칙 기반 93개: 입력만 있으면 정확히 재현할 수 있다.
- Gemma 계열 390개: Kemal의 `outputs/` checkpoint를 받아야 재현할 수 있다. 2026-10-06 요청했다.

## 4. 기존 relabel label 품질 체크 (val_unseen 209개 전부, 44초)

명령어 (파일 1개):
`python3 /home/irteam/git/vln-annotator/tools/check_label.py <val_unseen_X.json.gz> --split val_unseen`
3 split label은 `scripts/dataset_converters/relabel_vlnce/check_label_quality.py --label all`로 한 번에 돌린다.

- 분류: vision_grounded 194, template 15
  - template: complete_metadata, gate3_gemma_v1~v6, generated_gemma, generated_gemma_visual(_v2), v4~v8
- 전부 PASS: 2개 (`gemma`, `patched`). `patched`는 GT 원본이다.
- 게이트별 FAIL 수:

| 게이트 | FAIL |
|---|---|
| G6_stop_grounded | 207 |
| G2_unique | 16 |
| G4_direction | 10 |
| G3_vocab | 1 |
| G5_distance | 1 |

- G6이 거의 전부에서 걸린다. label 문제인지 기준(25~60%)이 부적절한지 확인이 필요하다.

**검증 단계 제안 (빠른 것부터)**
1. 품질 체크: 전체 1분
2. 규칙 기반 재생성 diff: 버전당 몇 분
3. Gemma 계열 100 ep 샘플 비교
4. 공개 System2 zero-shot 평가: label당 약 1.7h (GPU). 1단계 상위 후보만 진행한다.
