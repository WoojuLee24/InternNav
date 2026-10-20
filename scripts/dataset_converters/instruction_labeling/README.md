# instruction_labeling

목표: InternData-N1-v0.5-mini `vln_ce` 의 GT 지시문을, InternNav 입력과 같은 조건의 이미지 + GT waypoint 로
VLM 이 재현하게 한다. 계획: `.claude/tasks/instruction_labeling.md`. 교훈: `LESSONS.md`.

## P1 렌더 (`render/`)
habitat-sim 0.3.3 (`/usr/bin/python`) 으로 InternData 릴리스와 같은 카메라(640x480, hfov 79, 높이 1.25 m,
pitch 0 / 30) 를 재현한다. 설정은 `configs/render.yaml`.

| 파일 | 하는 일 |
|---|---|
| `habitat_rig_sim.py` | rig 센서를 단 habitat sim. `local_pose_cv` = 릴리스 `pose.<rig>` 규약 (시작 기준, x 전방 / y 좌 / z 위, OpenCV c2w) |
| `release_io.py` | 릴리스 parquet·이미지 읽기, lerobot episode ↔ raw R2R episode 매핑 (지시문 일치) |
| `p1r_replay_verify.py` | **렌더러 검증**: 릴리스 action 을 그대로 재생해 pose·RGB·depth 를 릴리스와 대조 |
| `action_gen.py` | reference_path → 이산 action 생성기 (val 용) |
| `p1a_action_gen.py` | **생성기 검증**: 후보 생성기 action 을 릴리스 train action 과 대조 |
| `render_split.py` | action 없는 split(val_unseen) 을 trajectory 단위로 렌더 → episode 단위 LeRobot 포맷으로 export |
| `verify_split.py` | export 결과 검사 (개수, 스키마, 이미지, 사본 동일성, 지시문, 05b V5/V6) |
| `viz_topdown.py` | navmesh 평면도 위 reference_path / 걸은 경로 / 카메라 방향 그림 |

명령 (한 줄):
```
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1r_replay_verify.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes 17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX --gpu 0 --out logs/instruction_labeling/p1r
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1a_action_gen.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes all --gpu 1 --out logs/instruction_labeling/p1a
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/render_split.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen --gpu 0
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/verify_split.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/viz_topdown.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen --n_per_scene 2 --out logs/instruction_labeling/topdown
```

| 인자 | 뜻 |
|---|---|
| `--config` | 렌더 설정 yaml (경로, 카메라, agent, jpeg, action 생성기) |
| `--scenes` | `all` 또는 쉼표 구분 scene 이름 |
| `--gpu` | habitat 렌더 GPU id (EGL) |
| `--split` | `val_unseen` 등 raw R2R split 이름. train 은 릴리스가 있으므로 render_split 에서 거부 |
| `--max_traj` / `--max_eps` | scene 당 상한 (디버그) |
| `--pixel_every` | p1r 에서 픽셀 비교 프레임 간격 (마지막 프레임은 항상) |

출력: `data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r/<split>/<scene>/{data,meta,videos}` (train 과 같은 구조),
`<split>/_index/<scene>.jsonl` (episode_index ↔ raw episode_id·trajectory_id), 렌더 캐시 `vln_ce_render/_traj_cache/`.
렌더는 trajectory 당 1회, episode(지시문) 마다 이미지 바이트 사본 (릴리스 train 과 같은 방식).

## P2 labeling (`label/`, `configs/`)

### 용어
| 용어 | 뜻 |
|---|---|
| label | 같은 R2R 경로(episode)에 자동 생성기로 새로 쓴 지시문. episode·경로·이미지는 GT 와 같고 **문장만** 다르다 |
| 방법(method) | `configs/methods/<method>.yaml` 의 **그룹별 인자 조합**. `b0`(기준) → `b1`(2026-10-20 지정, 아래). 더 좋다고 SR 로 확인된 조합만 새 번호로 추가한다 |
| 변형 이름 | b0 에서 바꾼 인자가 자동 suffix 로 붙는다: `b0_slight-omit_fewshot-3_rooms-house-v2` |
| `-v1`, `-v2` | 문구·규칙표 같은 버전 자원의 키. 정의를 고칠 때는 덮어쓰지 않고 새 키를 만든다 (gen_hash 가 바뀌어 결과가 섞이지 않는다) |
| SR | 성공률. 멈춘 위치가 목표 3 m 안. val_unseen 1,839 episode, mini s1 1 epoch, ld30 |
| OS | 경로 중 한 번이라도 목표 3 m 안에 들어간 비율 (지나쳐도 성공으로 침). OS − SR 이 크면 "도착했는데 안 멈춤" |
| 2x2 | {학습 문장 GT / label} × {평가 문장 GT / label}. **label→GT 가 핵심 지표** (label 로 학습해 사람 문장을 따르는가) |

### b1 (현재 기준, 2026-10-20)
`configs/methods/b1.yaml` = b0 + `route.slight=omit` + `style.fewshot=3` + `frames.keyframe_deg=0` + `route.rooms=house-v2`.
relabel_half s1 label→GT SR **37.4** (b0 32.1, GT 42.5). 이후 ablation 은 `--method b1 --set ...` (이름 `b1_<arg>-<value>`).

옛 이름과 생성 내용(gen_hash)이 같아 `label/rename.py` 로 복사했다 (다시 생성하지 않음). 이미 돌았거나 돌고 있는 학습·평가는 옛 이름으로 남아 있다.
| b1 이름 | 옛 이름 (결과 폴더) |
|---|---|
| `b1` | `b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2` |
| `b1_stop_cue-A-grounded-v1` | `b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2_stop_cue-A-grounded-v1` (node2 학습 중, 옛 이름) |
| `b1_sibling-profiles-v1_fewshot_match-gtstyle-v1` | `b0_..._rooms-house-v2_sibling-profiles-v1_fewshot_match-gtstyle-v1` (정보량 검사는 옛 이름) |
| `b1_select-keyframes-startsurround-v1` | `b0_..._rooms-house-v2_select-keyframes-startsurround-v1` |
| `b1_select-keyframes-startsurround-v1_sibling-profiles-v2_fewshot_match-gtstyle-v1` | `b0_..._rooms-house-v2_select-keyframes-startsurround-v1_sibling-profiles-v2_fewshot_match-gtstyle-v1` |
| `b1_fewshot-0` (복사 안 함) | `b0_slight-omit_keyframe_deg-0_rooms-house-v2` |

```
/usr/bin/python scripts/dataset_converters/instruction_labeling/label/rename.py --from b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2 --method b1
```

### b0 와 v278 / v278fix 의 차이
v278 / v278fix 는 `vln-s2-pipeline`(Kemal 생성기)으로 만든 이전 label, b0 는 이 폴더의 생성기다.

| 항목 | v278 | v278fix | **b0** (이 폴더) |
|---|---|---|---|
| 생성 방식 | 2단계: gate3 이 프레임마다 장면을 글로 요약 → gate4 가 **글만 보고** 문장 작성 | 같음 | **1단계(e2e)**: 이미지 여러 장 + 경로 단어를 VLM 한 번 호출에 바로 넣음 |
| VLM | gemma-4-31B-it **AWQ 4bit** | 같음 | gemma-4-31B-it **bf16** (`model=` 로 교체 가능) |
| 입력 이미지 | waypoint 에 카메라를 놓고 따로 렌더 (HFOV 90) | 같은 방식, HFOV 79 | **InternNav 학습·평가와 같은 이미지**: train = InternData 릴리스, val_unseen = P1 렌더 (실측 검증) |
| 프레임 선택 | waypoint 인덱스를 프레임 인덱스로 씀 (버그: 회전 프레임 대부분 빠짐) | waypoint 기준으로 수정 | **실제 걸은 궤적**에서 start / 회전을 마친 프레임 / goal |
| 카메라 방향 | 회전·goal 프레임이 **거울상 방향** (heading 부호 버그) | 같은 버그 | 올바른 방향 (top-down 그림으로 확인) |
| 검은 프레임 | 포함 | 제거 | 해당 없음 (실제 궤적 프레임) |
| seed | 고정 안 함 | `(seed, ep, stage, attempt)` 고정 | 고정 |
| 경로 설명 | gate2 경로 분석 요약 | 같음 | 숫자 없는 경로 단어 (vln-s2-pipeline e2e baseline 프롬프트) + 선택 옵션(방·정지 단서) |
| 설정 | `configs/labels/<label>.yaml` 상속 | 같음 | 그룹별 인자 + `--set` (아래) |

정리: v278 → v278fix 는 **프레임 버그 3개 수정**(waypoint 기준 key frame, HFOV 79, 검은 프레임 제거).
v278fix → b0 는 **생성 방식(e2e)·이미지(InternNav 조건)·방향 버그·양자화**가 함께 바뀌었다.
장면 전달 방식만 v278fix 식으로 되돌린 `b0_delivery-2stage-v1` 은 label→GT 33.8 로 b0(32.1)와 비슷하다
→ b0 의 이득은 2단계/1단계 차이보다 **입력 이미지·방향**에서 온 것으로 본다 (변수가 섞여 있어 완전 분리는 안 됨).

### 파일
| 파일 | 하는 일 |
|---|---|
| `configs/methods/b0.yaml` | b0 = 그룹별 인자 값 (모든 인자와 뜻이 주석으로 있다) |
| `configs/methods/b1.yaml` | b1 = 현재 기준 (b0 에서 4개 인자 변경) |
| `configs/route.yaml` | 경로 단어 문구, 시작 회전 확률, 방 이름(`rooms`), 정지 단서 후보(`stop_cue`), 정지 단서 고르기(`stop_select`) |
| `configs/style.yaml` | 프롬프트 template, 길이·정지 표현·few-shot 등 규칙 줄 |
| `configs/frames.yaml` | 프레임 선택 규칙(`select`), 도착 직전 프레임(`approach`) |
| `configs/delivery/<name>.yaml` | 장면 정보 전달: `image`(이미지 직접), `2stage-v1`(v278fix 식 장면 요약 글) |
| `configs/model/<name>.yaml` | vLLM 서버 주소·모델 이름, `managed:` 서버 인자 |
| `label/annotate.py` | **실행 진입점**. split 실행, 재개, 실패율 중단, json.gz 조립 (build_relabel_dataset 형식) |
| `label/config.py` | method + `--set` → 생성 cfg, 결과 이름, gen_hash (None 인 옵션은 hash 에서 빠짐 → 새 옵션은 기본 None) |
| `label/backend.py` | VLM 백엔드 `server` / `managed` |
| `label/e2e.py` / `label/two_stage.py` | delivery 별 생성기 |
| `label/frames.py` / `label/route_words.py` / `label/rooms.py` / `label/stop_cues.py` / `label/style.py` / `label/sources.py` | 프레임 선택 / 경로 단어 / MP3D 방 순서 / 정지 단서 후보 / 규칙 줄·few-shot / 입력 소스 (train 릴리스 ↔ raw 매핑) |
| `label/build_stop_cues.py` | 정지 단서 후보(A·P·R) 를 split 전체에 미리 계산해 캐시 (`stop_select` 가 읽음) |
| `label/rehash.py` | 설정 정리 후 기존 결과의 gen_hash 갱신 |
| `label/rename.py` | 생성 내용(gen_hash)이 같은 기존 label 을 새 방법 이름으로 복사 (예: SOTA → `b1`) |
| `eval/text_metrics.py` | 문장 지표 (BLEU/METEOR/ROUGE/CIDEr 등, navcomposer venv). SR 과 순위가 안 맞으니 1차 필터로만 |
| `eval/check_stop_cues.py` | 정지 단서 후보 검증 (후보 비율, GT 동의율, VLM 판정, 그림) |
| `eval/dev_subset.json` | val_unseen dev 집합 (101 trajectory / 303 episode) |

### 명령 (한 줄)
```
# 1) VLM 서버 (node1, GPU 8장, gemma DP8)
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/irteam/venvs/vllm_gemma/bin/vllm serve /home/irteam/data-vol2/checkpoints/VLM/gemma-4-31B-it --served-model-name gemma-4-31B-it --port 8100 --tensor-parallel-size 1 --data-parallel-size 8 --max-model-len 16384 --limit-mm-per-prompt '{"image": 16}' --gpu-memory-utilization 0.9

# 2) 라벨 생성 (val_unseen 1,839 ≈ 2.5 분, train 10,684 ≈ 12 분 @ gemma DP8)
/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --set route.slight=omit --set style.fewshot=3 --set frames.keyframe_deg=0 --set route.rooms=house-v2 --split val_unseen
/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --set route.slight=omit --set style.fewshot=3 --set frames.keyframe_deg=0 --set route.rooms=house-v2 --split train

# 3) dev 문장 지표 (선택)
PYTHONPATH= /home/irteam/venvs/navcomposer/bin/python scripts/dataset_converters/instruction_labeling/eval/text_metrics.py --split val_unseen --subset dev --label b0=data/InternData-N1-v0.5-mini/instruction_labels/b0/val_unseen_dev/val_unseen_b0.json.gz --out logs/instruction_labeling/p3/b0_dev
```

| `annotate.py` 인자 | 뜻 | 기본값 / 예 |
|---|---|---|
| `--method` | `configs/methods/<method>.yaml` | `b0` |
| `--set` | 인자 하나 바꾸기 `<group>.<arg>=<value>`, 그룹 값 전체는 `<group>=<value>`. 여러 번 쓸 수 있다 | `--set route.rooms=house-v2`, `--set model=qwen3vl32b-bf16` |
| `--split` | `train` 또는 `val_unseen` | |
| `--subset` | `''`(전체) 또는 `dev`(303 episode) | 빠른 비교는 `dev` |
| `--limit` | 앞에서 N episode 만 (스모크) | 0 = 전체 |
| `--backend` | `server`: 떠 있는 서버(model yaml 의 `base_url`)에 요청 / `managed`: job 이 서버를 띄우고 끝나면 내림 | `server` |
| `--gpus` | `managed` 일 때 GPU | model yaml |
| `--out_root` | 출력 루트 | `data/InternData-N1-v0.5-mini/instruction_labels` |

출력: `<out_root>/<이름>/<split>/<split>_<이름>.json.gz` (+ `meta/ep_*.json`: 입력 프레임·경로 단어·재시도 기록).
같은 설정(gen_hash)이면 이어서 생성하고, 설정이 다르면 기존 결과를 덮어쓰지 않는다.

### `--set` 으로 바꿀 수 있는 인자 (b0 기본값 → 선택지)
| 인자 | 뜻 | b0 기본 | 선택지 |
|---|---|---|---|
| `frames.select` | 프레임 고르는 규칙 | `keyframes-v1` (start + 회전점 + goal) | `keyframes-goalside-v1` (goal 좌우 추가), `keyframes-startsurround-v1` (출발 지점 좌·우·뒤 추가) |
| `frames.keyframe_deg` | 경로가 이 각도 이상 꺾이는 waypoint 만 key frame | 25 | **0 = 모든 waypoint** |
| `frames.extra_rigs` | 프레임마다 같이 넣을 rig | `[]` | `[125cm_30deg]` (30° 아래) |
| `frames.approach` | 도착 직전 프레임 | none | `v1` |
| `frames.max_images` | 프롬프트당 이미지 상한 | 16 | 서버 `--limit-mm-per-prompt` 와 맞춤 |
| `route.slight` | slight_deg 미만 회전을 말하는가 | say | **`omit`** |
| `route.start_turn` | 시작 회전을 말하는 규칙 | always | `gtrate-v1` (GT 비율로) |
| `route.rooms` | MP3D `.house` 방 순서를 경로 단어에 추가 | none | `house-v1` ("a -> b"), **`house-v2`** ("leave the a, enter the b") |
| `route.stop_cue` | 정지 지점 근처 물체/장소를 사실로 제공 | none | `A-grounded-v1/-v2` (보이는 물체), `P-v1/-v2` (계단·문간), `R-v1` (방) |
| `route.stop_select` | 후보 캐시(A·P·R)에서 정지 단서 고르기 | none | `rule-v1`, `vlm-v1`, `rotate-v1` |
| `route.turn_deg` / `slight_deg` / `around_deg` | 회전 / 살짝 / 뒤돌기 각도 기준 | 30 / 60 / 135 | |
| `delivery` | 장면 전달 방식 | image | `2stage-v1` |
| `style.length` | 길이 목표 | "15-40" | `free`, `gtdist` |
| `style.stop` | 끝맺음 표현 | must | `stop`, `wait`, `stand`, `none`, `optional`, `gtdist`, `prefer-stop-v1/-v2`, `mix-v1` |
| `style.fewshot` | 다른 경로의 GT train 문장 N 개를 문체 예시로 | 0 | **3** |
| `style.sibling` | 형제(같은 경로 3 episode)마다 다른 서술 방식 줄 | none | `profiles-v1` (step / landmark / brief), `profiles-v2` (+ landmark·brief 는 시작 회전 GT 비율·첫머리 규칙) |
| `style.fewshot_match` | fewshot 예시를 형제 서술 방식과 같은 GT 문체 묶음에서 | none | `gtstyle-v1` |
| `style.detail` / `style.structure` | 묘사 규칙 / 문장 구조 | specific-v1 / none | `plain-v1` / `room-v1` |
| `model` | VLM | gemma31b-bf16 | `qwen3vl32b-bf16`, `qwen3vl235b-a22b-fp8` |
| `gen.temperature` / `max_tokens` / `max_retries` | 샘플링 | 0.35 / 140 / 5 | 재시도마다 +0.1 |
| `gen.min_words` / `max_words` / `require_*` | 품질 검사 (실패하면 재시도, 마지막 시도는 경고로 남김) | 12 / 70 / true | |

### 학습·평가 (relabel_half 2x2)
```
# 4) 공유 디스크에 라벨 복사 (다른 노드가 가져감)
L=b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2; for s in train val_unseen; do cp -f data/InternData-N1-v0.5-mini/instruction_labels/$L/$s/${s}_$L.json.gz /home/irteam/data-vol2/relabel_half/labels/$s/; done
# 5) 평가 yaml + 셀 만들고 commit/push (node1)
python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels $L --emit data,yaml --splits train,val_unseen && python3 scripts/train_eval/qwenvl_train/relabel_half/make_cells.py $L
# 6) 학습 노드 queue 첫 줄: pull + 라벨 복사 + 학습 데이터 빌드 (노드마다 data 가 로컬이라 필요)
bash -c 'set -e; git remote update; git pull --ff-only origin feature/input_v0.1; for L in <L>; do for s in train val_unseen; do mkdir -p data/vln/mp3d/r2r/v1/$s; cp -f /home/irteam/data-vol2/relabel_half/labels/$s/${s}_$L.json.gz data/vln/mp3d/r2r/v1/$s/; done; python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels $L --emit data --splits train,val_unseen; done'
# 7) 셀 3개: 학습+GT평가 / 자기 label 평가 / GT 학습 모델의 이 label 평가
python3 scripts/train_eval/qwenvl_train/runner.py --machine h200 --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.<L>_eval.gt.py
python3 scripts/train_eval/qwenvl_train/runner.py --machine h200 --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.<L>_eval.<L>.py --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s1.train.<L>_eval.gt_internvla-n1-system2_*/ | head -1)
python3 scripts/train_eval/qwenvl_train/runner.py --machine h200 --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.gt_eval.<L>.py --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s1.train.gt_eval.gt_internvla-n1-system2_*/ | head -1)
```
- 라벨 이름에 `.` 은 쓸 수 없다 (셀 파일명이 `.` 으로 train/eval 을 나눔).
- 시간: 라벨 1개 ≈ 22.3 h (학습+GT평가 18.6 h, 자기 label 평가 2.0 h, GT 모델 평가 1.7 h, H200 8장).
- 결과: `/home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s1.train.<L>_eval.gt_*/logs/*/result_h200.json`.

## 성능 결과 (relabel_half s1, val_unseen 1,839, ld30, seed 1회)
값은 SR / OS (%). 1회 학습의 오차는 ±1~2%p 로 추정하므로 그보다 작은 차이로 결론 내리지 않는다. 2026-10-19 KST 기준.

| 학습 label | 바꾼 것 (b0 대비) | label→**GT** | label→label | GT→label |
|---|---|---|---|---|
| GT (상한) | 사람 문장 | **42.5** / 52.5 | — | — |
| **b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2** | C + 모든 waypoint + 방 이동(v2) | **37.4** / 50.1 | 평가 중 | 평가 중 |
| b0_slight-omit_fewshot-3_rooms-house-v2 | C + 방 이동(v2) | 35.9 / 48.9 | 51.9 / 64.1 | 41.4 / 53.6 |
| b0_slight-omit_fewshot-3_select-keyframes-startsurround-v1_rooms-house-v2 | C + 출발 주변 + 방(v2) | 35.6 / 48.0 | 51.2 / 64.1 | 42.2 / 54.9 |
| b0_stop_select-vlm-v1 | 정지 단서: VLM 이 후보에서 고름 | 35.0 / 47.7 | 51.7 / 66.2 | 40.3 / 52.6 |
| b0_stop_select-rotate-v1 | 정지 단서: 형제마다 다른 종류 | 34.7 / 45.9 | 51.5 / 63.1 | 40.1 / 52.2 |
| b0_stop_cue-A-grounded-v1 | 정지 단서: 보이는 물체 | 34.6 / 47.4 | 51.9 / 64.6 | 41.4 / 52.7 |
| b0_delivery-2stage-v1 | 장면 전달만 v278fix 식 | 33.8 / 52.5 | 54.7 / 68.0 | 38.1 / 52.9 |
| b0_slight-omit | 살짝 회전 생략 | 33.8 / 54.2 | 50.5 / 65.7 | 39.4 / 50.2 |
| b0_slight-omit_fewshot-3_keyframe_deg-0 | C + 모든 waypoint | 33.7 / 52.6 | 52.5 / 66.7 | 43.6 / 54.9 |
| b0_slight-omit_fewshot-3_select-keyframes-startsurround-v1 | C + 출발 주변 | 33.7 / 50.1 | 50.8 / 64.2 | 38.9 / 52.3 |
| b0_stop_select-rule-v1 | 정지 단서: 규칙(계단>문간>물체>방) | 33.0 / 45.0 | 48.6 / 63.2 | 39.5 / 52.2 |
| b0_slight-omit_fewshot-3 (= C) | 살짝 회전 생략 + GT 문체 예시 3개 | 32.7 / 50.6 | 49.4 / 63.1 | 39.5 / 50.6 |
| **b0** | — | 32.1 / 49.4 | 56.3 / 67.9 | 40.5 / 52.2 |
| v278fix_fewshot-3 | (이전 최고) | 33.6 / 43.3 | 32.7 / 43.0 | 29.3 / 43.4 |
| **v278fix** | (이전 기준선) | 31.9 / 40.0 | 31.9 / 41.9 | 30.0 / 43.4 |
| v218fix | 템플릿 (LLM 없음) | 27.3 / 37.0 | 40.5 / 57.6 | 30.9 / 47.6 |

읽는 법과 결론:
- **최고는 C+WP+방v2 37.4** (b0 +5.3, v278fix +5.5, GT 와의 차이 5.1%p). 방 이동(house-v2)을 넣은 3개가 모두 상위다.
- **GT→label (label 문장의 정보량)**: b0 계열 38~44 vs v278fix 계열 28~30. b0 문장은 GT 모델이 따라가도 GT 수준(42.5)에 가깝다 → 경로 정보는 충분하다.
- **label→label 이 높고 label→GT 가 낮다** (b0 56.3 vs 32.1): 자기 문체에는 잘 따르지만 사람 문장으로 일반화가 덜 된다. 실패 유형은 목표를 지나침 (OS=1 & SR=0: b0 17 % vs v278fix 8 %).
- v278fix 위 ablation(문체·길이·temperature)은 모두 ±2%p 안. b0 위에서는 경로 정보(방·waypoint)가 효과가 컸다.
- 대기: slight-omit+WP+방v2 (fewshot 없음, node2), Qwen3-VL-32B / 235B 로 만든 C+방v2 (node3 / node4).
- 자세한 분석: `.claude/memory/261015_instruction_labeling_sr_first_result.md`, 이전 label: `relabel_vlnce/report_relabel_2x2_20261013.md`.
