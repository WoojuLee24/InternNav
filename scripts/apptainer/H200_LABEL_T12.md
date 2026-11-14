# H200 노은역 랜드마크 — instruction 라벨링 → 학습 데이터 → S2 학습 (2026-11-15)

6000pro 에서 랜드마크 경로 1,200 개의 학습용 vln_ce 사진을 찍어 Gdrive 에 올렸다 (instruction 칸만 비어 있음).
H200 에서 **라벨 4 종 → 학습 데이터 8 개 → 학습 셀 8 개** 를 돌린다. 6000pro 에서 11 경로로 모든 단계를 시험했다 (라벨 오류 0 · 한 번에 통과 33/33, 학습 로더 검사 PASS).

## 용어
- **b1** = 사진을 보고 한 번에 쓰는 기본 라벨. **후 (post)** = b1 + 사실 메모 · 말투 손잡이 · 형제 3 문장을 한 번에(joint3) · 다듬은 작성기.
- **lm** = 멈출 곳 랜드마크 줄 포함 ("You stop next to the second poster, on your right"). 경로를 만들 때 고른 랜드마크.
- **형제** = 같은 경로를 다르게 말한 문장 3 개. 학습에는 경로 하나가 형제 3 문장으로 3 번 들어간다 (scene landmark_s0..s2).
- **gen_hash** = 라벨 설정 지문 (같으면 설정이 같다).

## 0. 코드
`git fetch old && git checkout noeun-landmark-house && git pull`

## 1. 데이터 받기 · 확인 · 풀기 (한 줄)
`rclone copy gdrive:InternNav/data/vln_ce_noeun_landmark_t12_train.tar . && rclone copy gdrive:InternNav/data/vln_ce_noeun_landmark_t12_train.tar.sha256 . && sha256sum -c vln_ce_noeun_landmark_t12_train.tar.sha256 && tar xf vln_ce_noeun_landmark_t12_train.tar`
- 크기 29,944,238,080 bytes, sha256 `bcc8002d05b7be370eaa7c1e5a7b1d64634cdf3165296ab6aa2efb467786b224`. 아래 `<DIR>` = 푼 곳.

## 2. 라벨 4 종 (62b)
공통 인자: `--data_root <DIR>/vlnce_train_t12 --managed --landmark_nouns noeun-v1 --rooms noeun-v2`

| 라벨 | 추가 인자 | gen_hash |
|---|---|---|
| b1 | `--methods b1` | `1dd95593a6ff` |
| b1lm | `--methods b1 --stop_cue noeun-paths-v1` | `8ee0be42a6e4` |
| post | `--methods post` | `b7b15a339bc3` |
| postlm | `--methods post --stop_cue noeun-paths-v1` | `7743e94b44d6` |

| 인자 | 뜻 | 왜 이 값인가 |
|---|---|---|
| `--landmark_nouns noeun-v1` | 품질검사가 "물건을 말했다" 고 인정하는 단어 (역 단어 148 개) | 기본(mp3d)은 집 물건뿐 → 역 문장이 거짓 경고로 5 번까지 다시 만들어졌다 |
| `--rooms noeun-v2` | 구역 이름 (platform · concourse · street …) | v1 은 바깥 출구 107 경로에 이름이 없었다 (v2 = G-D street) |
| `--stop_cue noeun-paths-v1` | 멈출 곳 랜드마크 줄 (몇 번째 · 옆 물체 · 계단 위 포함) | 경로의 랜드마크를 문장에 넣기 |
| `--managed` | `configs/model/gemma31b-bf16.yaml` 의 vLLM (GPU 0-7, DP8) 을 띄우고 끝나면 내림 | |

순서:
1. 입력 확인 (서버 없이): `/usr/bin/python scripts/dataset_converters/nav3d/noeun/62b_label_landmark.py --data_root <DIR>/vlnce_train_t12 --dry_run --limit 20 --rooms noeun-v2` — 구역 coverage 최소가 0 이 아니어야 한다.
2. 표본: 4 종 각각 `--ep_list 0,1,4,6,10,15,148,371,22,27,16` → gen_hash 가 표와 같은지 · 오류 0 인지. **표본 문장을 사진과 함께 사용자에게 보여 주고 "진행" 을 받는다.**
3. 전체 4 종 (1,200 경로 x 3 문장 = 3,600 문장씩). 이미 meta 가 있는 ep 는 건너뛴다 (끊겨도 같은 명령).
4. 점검: 4 종 모두 오류 0 · 3,600 문장 · `quality_issues` 비율 · 무작위 10 경로 사진과 문장 대조.

출력: `<DIR>/instruction_labels/nouns-noeun-v1/<이름>/landmark_t12_train/`.
→ 이 폴더를 tar 로 묶어 `gdrive:InternNav/data/noeun_eval/landmark_labels_t12/` 에 올린다 (6000pro 기록용).

## 3. 학습 데이터 8 개 (64b)
`/usr/bin/python scripts/dataset_converters/nav3d/noeun/64b_build_train_landmark.py --src <DIR>/vlnce_train_t12 --labels_root <DIR>/instruction_labels/nouns-noeun-v1`
- 원본을 `<mini>/vln_ce_noeun_src/landmark_t12/` 로 rsync (28 GB) 하고, `<mini>/vln_ce_noeun-lm{400,1200}-{b1,b1lm,post,postlm}/` 8 개를 만든다 (data · videos 는 링크). `--mini` 기본 `/home/irteam/data-vol1/InternData-N1-v0.5-mini`.
- 400 = 앞에서 경로 400 개 (x 3 형제 = 1,200 ep), 1200 = 전부 (3,600 ep).
- 검사: `/usr/bin/python scripts/dataset_converters/nav3d/checks/loader_check.py --vlnce "lm1200-b1lm=<mini>/vln_ce_noeun-lm1200-b1lm/traj_data/r2r=60,15,15"` → PASS.

## 4. 학습 셀 8 개
셀: `scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-lm{400,1200}-{b1,b1lm,post,postlm}.py` (make_cells_landmark.py 로 생성).
학습 설정 = 예전 노은역 셀과 같음: 공개 InternVLA-N1-System2 출발 · r2r_60cm_15_15 · 1 epoch · val 없음 · 평가 없음. 결과 폴더 `relabel_half_noeun_lm-v01/`.
- 한 셀: `python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-lm400-b1.py --machine h200 --no-eval`
- 먼저 400 셀 하나를 돌려 시작 · 손실 값이 정상인지 보고 사용자에게 알린 뒤 나머지 7 개.
- 끝나면 셀마다 ckpt 경로 · 학습 시간 · 마지막 loss 를 6000pro 세션에 알린다.

## 지킬 것
- 원본 데이터는 고치지 않는다. 데이터 삭제 금지. 커밋 · push 는 사용자 확인 후. 비밀번호를 파일 · 로그에 남기지 않는다. 명령은 한 줄로. 긴 작업(전체 라벨 · 학습)은 표본을 보여 주고 사용자가 "진행" 하면.
