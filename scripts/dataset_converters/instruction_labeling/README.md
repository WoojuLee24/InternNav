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
방법 = `configs/methods/<method>.yaml` 의 **그룹별 인자 조합** (frames / route / delivery / style / model / gen).
ablation 은 인자 하나를 바꾼다: `--set <group>.<arg>=<value>` (그룹 전체 값이면 `--set delivery=2stage-v1`).
결과 이름은 자동 suffix: `b0_slight-omit_fewshot-3`. 새 기준(b1 …)은 SR 로 검증된 뒤에만 methods 에 추가한다.

| 파일 | 하는 일 |
|---|---|
| `configs/methods/b0.yaml` | b0 = 그룹별 인자 값 (모든 인자 목록과 뜻이 주석으로 있다) |
| `configs/route.yaml` / `configs/style.yaml` | 경로 단어 문구·확률표 / 프롬프트 template·규칙 줄·정지 표현 (버전 키) |
| `configs/delivery/<name>.yaml` | 장면 정보 전달: `image`(이미지 직접), `2stage-v1`(v278fix 식 장면 요약 글) |
| `configs/model/<name>.yaml` | vLLM 서버 주소·모델 이름 |
| `label/config.py` | method + `--set` → 생성 코드용 cfg, 결과 이름, gen_hash |
| `label/backend.py` | VLM 백엔드: `--backend server`(떠 있는 서버) / `managed`(job 이 서버를 띄우고 끝나면 내려 GPU 반납). 결과·이름에 영향 없음 |
| `label/annotate.py` | split 실행, 재개, 실패율 중단, json.gz 조립 (build_relabel_dataset 형식) |
| `label/e2e.py` / `label/two_stage.py` | delivery type 별 생성기 |
| `label/style.py` / `label/route_words.py` / `label/frames.py` / `label/sources.py` | 규칙 줄·few-shot / 경로 단어 / 프레임 선택 / 입력 소스 |
| `eval/text_metrics.py` | 문장 지표 (navcomposer venv) |

```
/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --split val_unseen --subset dev
/usr/bin/python scripts/dataset_converters/instruction_labeling/label/annotate.py --method b0 --set route.slight=omit --set style.fewshot=3 --split val_unseen --subset dev
PYTHONPATH= /home/irteam/venvs/navcomposer/bin/python scripts/dataset_converters/instruction_labeling/eval/text_metrics.py --split val_unseen --subset dev --label b0=data/InternData-N1-v0.5-mini/instruction_labels/b0/val_unseen_dev/val_unseen_b0.json.gz --out logs/instruction_labeling/p3/b0_dev
```

### VLM 백엔드 (`--backend`)
| 값 | 동작 | 언제 |
|---|---|---|
| `server` (기본) | `configs/model/<name>.yaml` 의 `base_url` 서버에 요청 | 서버를 여러 실험이 같이 쓸 때 |
| `managed` | job 시작 시 `managed:` 블록 설정으로 vLLM 서버를 자식 프로세스로 띄우고, 끝나면(예외 포함) 내린다. data parallel = GPU 수 / TP | queue 에서 학습 job 과 번갈아 돌릴 때 (GPU 반납) |

`--gpus 0,1,2,3` 로 managed 서버의 GPU 를 고른다. 서버 로그는 `<출력 폴더>/vllm_server.log`.
