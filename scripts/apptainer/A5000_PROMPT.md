노은역 랜드마크 `.house` 작업을 6000pro 서버에서 이어받는다. a5000에서 아래를 진행해라.

## 0. 제약
- **커밋·push·삭제는 하지 마라.** 막히면 멈추고 보고해라.
- **Gemma·Isaac·GroundingDINO를 다시 돌리지 마라.** 이미 끝난 단계다. 남은 일은 전부 CPU다.
- `scripts/apptainer/noeun_apptainer.sh`는 **6000pro 전용**이다. 경로가 박혀 있고 SIF가 sm120용이라 여기선 안 돈다. a5000은 기존 docker 실행 방식을 써라.

## 1. 코드 받기
```
git fetch origin && git checkout noeun-landmark-house
```
`origin`은 `https://github.com/GuideDog-ETRI/InternNav.git`. 커밋 `06298fb`.

## 2. 데이터 받기 (45 MB)
```
rclone copy gdrive:InternNav/data/noeun_landmark_handoff.tar.gz . --progress
rclone copy gdrive:InternNav/data/noeun_landmark_handoff.tar.gz.sha256 .
sha256sum -c noeun_landmark_handoff.tar.gz.sha256
tar xzf noeun_landmark_handoff.tar.gz
cat bundle/README.md
```
sha256 = `25fef25cbbad30b09cacba3db4ccc28e4f54c5eef117e2753c26e69ca26e31a4`

`<NAV>` = a5000의 InternNav 경로, `<DATA>` = 그 서버 data 루트:
```
cp -r bundle/logs    <NAV>/logs/
cp -r bundle/scripts <NAV>/scripts/
cp -r bundle/DATA/data <DATA>/
```
`73_goal_candidates.py`의 `--navmesh` 기본값이 `data/data/noeun_eval/...`이므로 `<NAV>/data`가 `<DATA>`를 가리켜야 한다. 아니면 `--navmesh`/`--mesh`/`--verify`를 직접 줘라.

## 3. 준비 확인
```
python3 scripts/dataset_converters/nav3d/noeun/70_make_house.py \
  --landmarks logs/scene_landmarks --out /tmp/t/noeun_station.house
```
**`[70] 검증: load_objects 가 N 개 읽음 (기대 N)`** 이 나오면 준비 끝. 안 나오면 멈추고 보고해라.
필요한 파이썬: `numpy`·`trimesh`·`habitat_sim`(73만). habitat 컨테이너면 충분하다.

## 4. 할 일

### (가) `.house` 최종본
```
python3 scripts/dataset_converters/nav3d/noeun/70_make_house.py \
  --landmarks logs/scene_landmarks \
  --out data/scene_data/noeun_station/noeun_station.house
```
사람이 리포트에서 고친 `corrections.json`을 받으면 **(가)와 (나) 둘 다**에 `--corrections corrections.json`을 붙여라. 한쪽만 주면 `.house`와 목적지가 어긋난다.

### (나) 목적지 후보
```
python3 scripts/dataset_converters/nav3d/noeun/73_goal_candidates.py \
  --landmarks logs/scene_landmarks --out logs/goal_candidates \
  --exclude_types pillar,canopy,arrow,sign,signage,car
```
기대값: 랜드마크 515 → 쓸 수 있는 것 471 → **목적지 1,412개**. 크게 다르면 멈추고 보고해라.

### (다) 경로 생성 — **74번을 써라. 22번이 아니다.**
```
python3 scripts/dataset_converters/nav3d/noeun/74_make_paths.py \
  --goals logs/goal_candidates/goals.json \
  --out_dir logs/noeun/paths_landmark
```
**22번을 쓰면 안 된다.** 22번은 출발점·도착점을 둘 다 무작위로 찍어서, (나)에서 고른 목적지 1,412곳을 쓰지 않는다. 그대로 돌리면 이번에 고치려던 문제("멈출 물건이 없는 목적지")가 그대로 남고, 10월에 만든 기존 경로 파일도 덮어쓴다.

74번은 **도착점이 이미 정해져 있고 출발점만 찾는다.** 판정은 22번과 같은 U1~U5 를 그대로 쓴다.
- `--hole_kinds mesh,gs` 가 **기본값**이다 (22번 기본값 `mesh` 와 다르다). `gs` 는 렌더에 바닥이 비치는 곳 139개다.
- eval/train 을 **랜드마크 단위**로 자른다. 목적지 단위로 자르면 같은 자판기가 양쪽에 들어가 평가가 샌다.
- 에피소드마다 `stop_cue`(랜드마크 이름·별명·공통 class·거리)와 `goal_yaw`가 박혀 나간다.

기대값: 평가 80 + 학습 1,200 · 목적지~멈출 대상 거리 중앙 1.3 m · **최대 2.0 m**.
작게 먼저 확인해라 — `--n_eval 6 --n_train 10 --out_dir /tmp/p74`.

### (라) 그 뒤
문장 생성(Gemma-4-31B, 이번엔 `stop_cue`가 채워진다) → 학습 400/1,200 두 셀 → 평가 80.
여기부터는 GPU가 필요하다. 진행 전에 보고해라.

## 5. 배경 — 왜 이렇게 하나
평가셋 40개 중 21개가 못 쓸 라벨이었다. 목적지를 아무 데나 찍고 "거기 멈출 만한 게 있나"를 나중에 찾았기 때문이다(대상이 16 m 밖인 경우도 있었다). 원인은 `stop_cue: null` — 노은역은 GS 스캔이라 `.house`가 없어서 VLM이 멈출 대상을 지어냈다.

**순서를 뒤집었다.** 역 안의 물건을 먼저 찾아 목록으로 만들고, 그 옆 0.6~2 m를 목적지로 삼는다. 로봇이 랜드마크 옆 약 1 m에 서므로 목적지에서 최대 3 m = SR@3m 충족. `.house`를 만들어 주면 기존 MP3D 파이프라인을 한 줄도 안 고친다.

## 6. 이미 정해진 것 (바꾸지 말 것)
- 바닥 구멍은 **CRITICAL 197곳만** 거른다. minor 1,993곳까지 빼면 역을 거의 못 쓴다.
- **기둥·표지판·자동차·차양·화살표 제외** (297개, 37 %). 물체를 못 찾아서가 아니라 **흰 기둥이 103개라 "흰 기둥 옆에서 멈춰라"가 어느 기둥인지 가리키지 못하기 때문**이다.
- 랜드마크당 목적지 **3자리**. 2자리면 942개라 필요한 1,280개(학습 1,200 + 평가 80)에 모자란다.

## 7. 참고
- `scripts/apptainer/REPRODUCE.md` — 0~6단계 전체 재현 명령
- `scripts/apptainer/checks/README.md` — 어떤 측정으로 무엇을 결정했는지
- `.claude/memory/261113_noeun_landmark_house_result.md` — 용어·결정·겪은 함정
- 리포트 https://claude.ai/code/artifact/d7945fda-99fa-4e5f-9f05-ab3a134f446c
