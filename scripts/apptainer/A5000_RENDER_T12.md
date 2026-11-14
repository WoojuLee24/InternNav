# A5000 학습용 vln_ce 사진 분담 (회전 기준 12° 걷기 결과) — a6000 data2 세션용 (2026-11-14 21:15)

> **2026-11-14 21:35 취소** — A5000 사진이 6000pro 와 RGB 평균 1.9/255 이상 달라(합격 ≤ 0.5) 쓰지 않는다. 학습 사진은 6000pro GPU 0·1 이 모두 찍는다. A5000 은 찍지 않아도 된다.

뒤쪽 350 ep 걷기 고맙다 (318/350 · 왔다갔다 12.3% — 6000pro 앞쪽 12.6% 와 같은 수준, 설정 확인됨).
**먼저 하나만 알려 달라**: `A5000_WALK_T12.md` 1 단계(48 ep 같은지 시험)를 했다면 `cmp_walk.py` 마지막 두 줄과 걸린 시간.

## 받을 것 (Gdrive `gdrive:InternNav/data/noeun_eval/landmark_walk_t12_20261114/render/`)
| 파일 | 내용 | 둘 곳 |
|---|---|---|
| `sel_train.json` (md5 cecb20b781389aaf1d09a0946f0bb9f9, 30 MB) | 최종 학습 1,200 ep (앞 1,402 + A5000 뒤 350 합쳐 고름) | `logs/noeun/paths_landmark_ext/walk_t12/` |
| `ref_ep0_2_6000pro.tar` (86 MB) | 6000pro 가 찍은 ep 0·1·2 사진 910 장 (비교용) | `data/shard_ref/` 에 풀기 |
| `imgdiff.py` | 두 out_root 의 사진 비교 | `logs/env_cmp/tools/` |

코드: `git pull` (브랜치 noeun-landmark-house) — `51b_render_rbq_shard.py` · `noeun_render_train_t12.sh` (6000pro 용, 참고).

## 1. 사진 같은지 시험 (GPU 1 장, 약 5 분)
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk_t12/sel_train.json --out_root data/shard_test/A5 --log_dir logs/shard_test/A5 --ep_start 0 --ep_end 3
python3 logs/env_cmp/tools/imgdiff.py data/shard_ref data/shard_test/A5
```
- 합격: **depth 455 장 모두 identical**, RGB meanabs mean ≤ 0.5 (6000pro 끼리 0.27~0.43). 떨어지면 멈추고 숫자 보고.
- 속도(장/초)도 적는다 (6000pro 4 장/초).

## 2. 본 렌더 (합격 후) — GPU 마다 하나 (`CUDA_VISIBLE_DEVICES=<g>`), 모두 같은 out_root · log_dir
| A5000 GPU | `--ep_start` | `--ep_end` | 장 수 |
|---|---|---|---|
| 0 | 757 | 861 | 14,846 |
| 1 | 861 | 963 | 14,909 |
| 2 | 963 | 1081 | 14,869 |
| 3 | 1081 | 1200 | 14,886 |
(6000pro: GPU0 0-398 · GPU1 398-757, 각 ≈ 49,800 장 · 21:10 시작 · 약 3.5 h)
```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk_t12/sel_train.json --out_root data/noeun_landmark_v1/vlnce_train_t12 --log_dir logs/noeun/paths_landmark_ext/render_log/vlnce_train_t12 --ep_start 757 --ep_end 861 --skip_existing
```
- `--skip_existing` = 그 ep 의 마지막 사진이 있으면 건너뜀 → 끊기면 같은 명령으로 이어 찍는다.
- 끝줄 `[51b] 렌더 104 ep (새로 104) · 14846 프레임` 처럼 장 수가 표와 같은지 확인.
- CPU 48 코어 — `uptime` load 가 48 을 넘으면 알려 달라.

## 3. 돌려줄 것 (Gdrive `.../landmark_walk_t12_20261114/a5000_render/`)
- 사진: `data/noeun_landmark_v1/vlnce_train_t12/traj_data/r2r/noeunstation/videos/chunk-000/observation.images.{rgb,depth}.60cm_15deg/episode_000{757..1199}_*` 를 tar 로 (약 59,500 장 × 2)
- 자세: `logs/noeun/paths_landmark_ext/render_log/vlnce_train_t12/vlnce/sentinels/pose__60cm_15deg.part_*.npz` 4 개
- 6000pro 에서 같은 위치에 풀고 `--assemble` 로 1,200 ep 를 한 번에 조립한다.

## 지킬 것
- 커밋 · push 는 사용자 확인 후. 데이터 삭제 금지. 비밀번호를 파일 · 로그에 남기지 않는다.
