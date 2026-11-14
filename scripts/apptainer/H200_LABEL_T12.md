# H200 노은역 랜드마크 학습 데이터 instruction 라벨링 (2026-11-15)

6000pro 에서 노은역 랜드마크 경로 1,200 개의 학습용 vln_ce 사진을 찍어 Gdrive 에 올렸다. **instruction 칸만 비어 있다**.
H200 에서 `62b_label_landmark.py` 로 b1 · 후(post) 두 라벨을 만든다. 6000pro 에서 11 경로 x 3 문장으로 시험해 오류 0, 한 번에 통과 33/33 확인 (리포트 14 단계).

## 0. 코드
`git fetch old && git checkout noeun-landmark-house && git pull` — 필요한 것: `nav3d/noeun/62b_label_landmark.py` · `61b_extend_zones.py` · `zones/noeun_zones_v2.json` · `landmark_nouns/` · `instruction_labeling/configs/route.yaml` (rooms.noeun-v2 · stop_cue.noeun-paths-v1 추가).

## 1. 데이터 받기 · 확인 · 풀기 (한 줄)
`rclone copy gdrive:InternNav/data/vln_ce_noeun_landmark_t12_train.tar . && rclone copy gdrive:InternNav/data/vln_ce_noeun_landmark_t12_train.tar.sha256 . && sha256sum -c vln_ce_noeun_landmark_t12_train.tar.sha256 && tar xf vln_ce_noeun_landmark_t12_train.tar`
- 크기 29,944,238,080 bytes, sha256 `bcc8002d05b7be370eaa7c1e5a7b1d64634cdf3165296ab6aa2efb467786b224`.
- 풀면 `vlnce_train_t12/traj_data/r2r/noeunstation/` (parquet · meta · videos 의 프레임별 jpg/png) + `vlnce_train_t12/noeun_meta/` (render_meta · sel_train · paths json — 62b 가 읽는다).

## 2. 설정 (사용자 결정 2026-11-15)
| 라벨 | 62b 인자 | 이름 · gen_hash |
|---|---|---|
| b1 | `--methods b1 --landmark_nouns noeun-v1 --rooms noeun-v2 --stop_cue noeun-paths-v1` | `b1_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v2_stop_cue-noeun-paths-v1` · `8ee0be42a6e4` |
| 후 | `--methods post --landmark_nouns noeun-v1 --rooms noeun-v2` | `b1_facts-memo-v1_knobs-v2_delivery-joint3-v1_writer-gepa-v3-joint-gemma_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v2` · `b7b15a339bc3` |

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--landmark_nouns noeun-v1` | 품질검사가 "물건을 말했다" 고 인정하는 단어 (역 단어 148 개) | 기본(mp3d)은 집 물건뿐이라 역 문장이 거짓 경고로 5 번까지 다시 만들어졌다 |
| `--rooms noeun-v2` | 구역 이름 (platform · concourse · street …) | v1 은 바깥 출구 107 경로에 이름이 없었다. v2 = G-D street 추가 |
| `--stop_cue noeun-paths-v1` | 경로를 만들 때 고른 랜드마크를 한 줄로 (몇 번째 · 옆 물체 · 계단 위 포함) | b1 만 (후 는 받는 자리가 없다) |

## 3. 순서
1. 서버 없이 입력 확인: `/usr/bin/python scripts/dataset_converters/nav3d/noeun/62b_label_landmark.py --data_root <DIR>/vlnce_train_t12 --dry_run --limit 20 --rooms noeun-v2` — 구역 coverage 최소가 0 이 아니어야 한다.
2. 표본: 위 표 인자에 `--managed --ep_list 0,1,4,6,10,15,148,371,22,27,16` → 6000pro 결과와 gen_hash 가 같은지, 오류 0 인지. **표본을 사용자에게 보여 주고 "진행" 을 받는다.**
3. 전체: `--managed` 로 b1 · 후 각각 (1,200 경로 x 3 문장). `--managed` 는 `configs/model/gemma31b-bf16.yaml` 의 vLLM (GPU 0-7, DP8) 을 띄우고 끝나면 내린다. 이미 meta 가 있는 ep 는 건너뛴다 (끊겨도 같은 명령).
4. 점검: 오류 0 · 3,600 문장 x 2 · `quality_issues` 비율 · 무작위 10 경로 사진과 문장 대조.

출력: `<DIR>/instruction_labels/nouns-noeun-v1/<이름>/landmark_t12_train/` (meta/ep_*.json + landmark_t12_train.json.gz + 위 폴더의 landmark_nouns.json).
episode_id = 경로번호 x 10 + 형제(0..2), trajectory_id = 경로번호 = vlnce_train_t12 의 episode_index.

## 4. 돌려줄 것
`instruction_labels/nouns-noeun-v1/` 를 tar 로 묶어 `gdrive:InternNav/data/noeun_eval/landmark_labels_t12/` 에 올리고 크기 · sha256 · 문장 수를 6000pro 세션에 알린다.
(라벨을 데이터셋 tasks 에 붙이는 단계(63 · 64 방식)는 6000pro 에서 따로 한다.)

## 지킬 것
- 원본 데이터는 고치지 않는다. 데이터 삭제 금지. 커밋 · push 는 사용자 확인 후. 비밀번호를 파일 · 로그에 남기지 않는다. 명령은 한 줄로.
