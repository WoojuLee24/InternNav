# nav3d 실행 명령

모든 명령은 **저장소 루트**(`InternNav/`)에서, 명령은 한 줄로 쓴다. 구조와 단계 설명은 [README.md](README.md), 리포트 위치는 [REPORTS.md](REPORTS.md).

## 0. 용어
| 용어 | 뜻 |
|---|---|
| 데이터셋 | `vlnce` (MP3D → vln_ce) · `vlnpe` (MP3D → vln_pe) · `noeun` (노은역) |
| mini · full | 공개 학습 데이터 두 벌. mini = `data/InternData-N1-v0.5-mini` (이 PC), full = `data/InternData-N1` (학습 서버). 구조는 같고 루트만 다르다 |
| 단계 번호 | 앞자리 = 1 준비 · 2 경로 만들기 · 3 경로 보정 · 4 걷기 · 5 렌더 · 6 라벨 · 7 검사, 뒷자리 = 그 안 순서 |
| wp · sg | wp = R2R 경유점을 따라 계획한 경로, sg = 시작·끝만 주고 계획한 경로. 따로 만들고 따로 저장한다 (`nav3d_wp` · `nav3d_sg`) |

## 1. 컨테이너
| 이름 | 이미지 | python | 쓰는 단계 |
|---|---|---|---|
| habitat | `dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-5090` | `/usr/bin/python` | Isaac 이 아닌 단계 전부 (vlnce 51 렌더 포함) |
| isaac | `dnwn24/internnav:torch2.9.0-cuda13.0-issacsim5.1.0-ros2-jazzy-5090` | `/workspace/isaaclab/_isaac_sim/python.sh` | noeun 걷기 · 렌더, vlnpe 렌더 · H1 걷기 |

- habitat_sim 은 torch2.9.**0** habitat 이미지에만 있다 (torch2.9.1 · gdsim 이미지에는 없다).
- isaac 이미지를 새로 띄우면 `scikit-image` 를 먼저 설치한다: `/workspace/isaaclab/_isaac_sim/python.sh -m pip install scikit-image`
- 실행기는 어느 컨테이너에서나 `python3` 로 돈다. 다른 컨테이너 단계에 닿으면 멈추고 이어 돌릴 `--from <단계>` 명령을 출력한다.

## 2. 실행기 `run_pipeline.py`
```
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --list
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --dry_run
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce
```

| 인자 | 뜻 | 기본 · 왜 |
|---|---|---|
| `vlnce` · `vlnpe` · `noeun` | 데이터셋 | — |
| `--data mini \| full` | 공개 학습 데이터 종류 | mini. 지금은 vlnce 만 (vlnpe 는 나중에, noeun 은 무관) |
| `--list` | 단계 목록 (번호 · 컨테이너 · 스크립트) | — |
| `--dry_run` | 실행할 명령만 출력 | 각 단계를 스크립트로 따로 돌릴 명령이 그대로 나온다 |
| `--stage 1,2` | 단계 묶음만 (번호 앞자리) | — |
| `--only a,b` · `--from a` · `--to b` | 이름으로 고르기 · 구간 | 이름은 `--list` 에 나온다 |
| `--overwrite` | 출력이 이미 있어도 돌림 | 기본은 멈춤 — 지금 결과를 실수로 덮지 않게 |
| `--scenes a,b` | 씬 제한 (테스트) | 빈값 = 전부 |
| `--max_episodes N` | 개수 제한 (테스트) | 0 = 전부. vlnce: 씬마다 평지 경로 N 개(층 이동은 전부) · 라벨 N 개, noeun: 경로 · 렌더 ep, vlnpe: H1 걷기 ep |
| `--jobs N` | 씬마다 도는 단계(11 · 21)의 동시 프로세스 수 | 4 |
| `--episodes_json f` | `{씬: [릴리스 episode_index]}` 로 일부만 | vlnce: 빈값 = 전부. vlnpe: 아직 필요 (기본 `logs/common/episodes.json`) |
| `--label_method` · `--label_model` | 라벨 방법 · VLM | b1 · qwen3vl8b-local |
| `--vlm_server auto \| external` | 라벨 단계 VLM 서버를 실행기가 띄우고 내릴지 | auto (안 떠 있으면 띄우고 끝나면 내림, 이미 떠 있던 서버는 그대로) |
| `--stairs_episode_ids` · `--walk_workers` | noeun 계단 직선화 ep · RBQ 걷기 동시 수 | 86 · 5 |

## 3. vlnce (MP3D → vln_ce)
기본은 **학습 데이터와 같게**: 릴리스 학습 데이터의 모든 경로(trajectory) · 지시문마다 에피소드 하나 (R2R 은 경로 하나에 지시문 보통 3 개, 프레임은 같다).

```
# 전체 (mini, habitat 컨테이너)
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --stage 1,2,5,7
# 테스트: 2 씬 · 씬마다 평지 경로 3 개 (+ 층 이동 전부)
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --stage 1,2,5,7 --scenes 17DRP5sb8fy,1LXtFkjw3qL --max_episodes 3
# full (학습 서버, data/InternData-N1 이 있을 때). logs 는 logs/full/ 로 따로 쌓인다
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --data full --stage 1,2,5,7
# 처음부터 라벨까지 한 번에 (VLM 서버는 실행기가 띄우고 끝나면 내린다)
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce
# 라벨만
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnce --only label
```

라벨(61) 단계의 VLM 서버 (`--vlm_server auto`, 기본):
- 서버가 안 떠 있으면 실행기가 `common/vlm_server.py` 를 띄우고, 응답할 때까지 기다린 뒤(모델 로드 ≈ 4 분, 최대 `--vlm_wait_s` 900 초) 라벨링하고,
  끝나면 — 실패해도 — 내린다. 로그: `logs/run_pipeline/vlnce/vlm_server.log`.
- 이미 떠 있으면 그 서버를 쓰고 **내리지 않는다** (다른 곳에서 쓰는 서버일 수 있다).
- 이 PC 로컬 서버 모델(`qwen3vl8b-local`)만 자동으로 띄운다. 다른 `--label_model` 이나 `--vlm_server external` 이면 떠 있는지만 확인한다.
- 서버를 직접 띄우려면: `PYTHONPATH=/ws/src/InternNav/.venv_label/site /usr/bin/python scripts/dataset_converters/nav3d/common/vlm_server.py`

| 출력 | mini | full |
|---|---|---|
| 데이터셋 | `data/InternData-N1-v0.5-mini/vln_ce_nav3d/traj_data/r2r/nav3d_{wp,sg}/<씬>/` | `data/InternData-N1/vln_ce_nav3d/...` |
| 에피소드 색인 (릴리스 ep ↔ 내보낸 ep · trajectory) | `.../vln_ce_nav3d/traj_data/_index/nav3d_{wp,sg}/<씬>.jsonl` | 같은 위치 (full 루트) |
| 라벨 | `data/InternData-N1-v0.5-mini/instruction_labels_nav3d/<방법>/nav3d_{wp,sg}/` | `data/InternData-N1/instruction_labels_nav3d/...` |
| logs | `scripts/dataset_converters/nav3d/logs/{common,vlnce,checks}/` | `scripts/dataset_converters/nav3d/logs/full/...` |

- 렌더된 데이터의 지시문은 아직 **릴리스 사람 지시문**(GT 경로를 설명한 것)이다. 생성 경로에 맞는 지시문은 instruction labeling(6 단계)으로 만들고 있다.
- 예전 동작(목록에 있는 ep 만 · 경로마다 에피소드 하나): `51_render.py --episodes_json <목록> --instructions first`.

### val_unseen (평가 split — 2x2 라벨 실험용, 실행기 밖에서 스크립트로)
릴리스 렌더가 없어 원본 json(`raw_data/r2r/val_unseen`)만 쓴다: GT = reference_path 경유점을 find_path 로 이은 경로, 21 · 51 의 ep 키 = `trajectory_id`,
내보낸 에피소드의 `episode_id` = 원본 episode_id (색인 `_index` 에 있다 — 라벨을 평가 에피소드에 붙일 때 쓴다). 11 단계는 없다.
```
D=data/InternData-N1-v0.5-mini; R=scripts/dataset_converters/nav3d; L=$R/logs/common/refine_paths_val_unseen
A="--split val_unseen --raw_root $D/vln_ce/raw_data/r2r --scene_root $D/scene_data/mp3d_ce/mp3d --mesh_root $D/scene_data/mp3d_n1 --out_dir $L"
python3 -c "import gzip,json;print('\n'.join(sorted({e['scene_id'].split('/')[-2] for e in json.load(gzip.open('$D/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz'))['episodes']})))" \
  | xargs -P 11 -I{} /usr/bin/python $R/vlnce/21_refine_paths.py --scene {} --max_flat 0 $A
/usr/bin/python $R/vlnce/21_refine_paths.py --aggregate $A
/usr/bin/python $R/vlnce/51_render.py --split val_unseen --nav03_dir $L/scenes --out_dir $R/logs/vlnce/51_render_val_unseen
```
출력: `data/InternData-N1-v0.5-mini/vln_ce_nav3d_val_unseen/traj_data/{r2r/nav3d_{wp,sg},_index,_traj_cache}` (train 과 폴더를 나눈다).

## 4. noeun (노은역 → vln_ce · vln_pe v6)
```
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --list
# habitat 컨테이너: 1~3 단계 + 걷기 입력
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --to make_walk_inputs
# isaac 컨테이너: 첫 걷기부터 (habitat 단계에 닿으면 멈추고 이어 돌릴 명령을 알려 준다)
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --from walk_first
# 테스트: 경로 20 (층 이동 10 + 평지 10) · 렌더 3 ep
python3 scripts/dataset_converters/nav3d/run_pipeline.py noeun --max_episodes 10
```
출력: `logs/noeun/paths/` (경로 json v4 ~ v6) · `logs/noeun/walk/` (걷기 입력 · 결과) · `data/.../vln_{ce,pe}_noeun_nav3d_rbq_v6`.
일부 ep 만 다시 렌더: `noeun/tools/rerender_rbq_subset.py --help`.

## 5. vlnpe (MP3D → vln_pe) — 나중에 vlnce 처럼 바꿀 예정
```
python3 scripts/dataset_converters/nav3d/run_pipeline.py vlnpe --list
```
지금은 `logs/common/episodes.json` 의 4 씬 29 ep 만 처리하고, sg 만 만든다 (wp 는 나중에 추가). 방식 A(경로 위 카메라) 51~54, 방식 B(H1 물리 걷기) 55.

## 6. 스크립트를 따로 돌리기
실행기 없이 스크립트를 직접 돌려도 같다. 명령은 `--dry_run` 에서 복사한다. 예:
```
/usr/bin/python scripts/dataset_converters/nav3d/vlnce/21_refine_paths.py --scene 17DRP5sb8fy --max_flat 0
/usr/bin/python scripts/dataset_converters/nav3d/vlnce/51_render.py --instructions all --scenes 17DRP5sb8fy
/usr/bin/python scripts/dataset_converters/nav3d/noeun/45_straighten_stairs.py --help
```
- 데이터셋 폴더의 11 · 21 · 61 · 71 등은 **연결 파일**이다 (`common/` · `checks/` 스크립트 + 고정 인자 `ARGS`).
- 모든 스크립트는 `--help` 에 설명과 사용 예가 있다.

## 7. 주의
- `--overwrite` 는 실행기의 "출력 있음" 멈춤만 건너뛴다. 렌더 스크립트(vlnce 51 · vlnpe 54 · noeun 51)는 출력 폴더가 있으면 스스로 멈추므로, 다시 렌더하려면 그 폴더를 먼저 옮기거나 지운다.
- 노은역 usdz 기본 경로 `data/GS_USDZ/Subway/noeun_station_collision.usdz` 가 이 PC 의 data 마운트에 없다. 같은 파일이 `/ws/assets/GS_USDZ/` 에 있다.
- vlnpe H1 걷기는 `data/sample_episodes/` 에 lmdb 캐시를 쓴다 (쓰기 가능해야 한다).
- full 은 이 PC 에 없어 `--dry_run` 경로까지만 확인했다. full 의 씬 구성(mini 와 같은 61 씬인지)은 아직 모른다.
- 실행기는 mini · full 모두 데이터 루트 기준 경로를 넘긴다 (`<루트>/scene_data/mp3d_ce/mp3d` · 21 의 mesh `<루트>/scene_data/mp3d_n1`).
  그래서 `data/scene_data` 링크와 gs_vlnpe `scene_meta`(pxr 필요) 가 없는 서버(H200 pod)에서도 돈다 — 이 PC 에서 둘 다 빼고 확인, 결과 같음.
  스크립트를 따로 돌릴 때는 `--scene_root` · 21 `--mesh_root` 를 직접 준다 (기본값 `data/scene_data/...` 는 이 PC 에만 있다).
