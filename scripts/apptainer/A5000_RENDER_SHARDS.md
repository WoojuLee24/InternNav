# A5000 학습용 vln_ce 사진 분담 — a6000 data2 세션용 (2026-11-14 17:30)

> **2026-11-14 18:50 취소** — 회전 기준 12° 로 다시 걷기 때문에 sel_train 이 바뀐다. `A5000_WALK_T12.md` 를 따른다. 렌더 분담은 새 걷기 뒤 다시 정한다.

앞 문서 `A5000_RENDER_PROMPT.md` 의 준비(코드 pull · rbq10 assets · usdz · /pylib skimage · 시험 렌더)를 이어서 한다.
바뀐 점: **학습용은 vln_ce 만** 찍는다 (vln_pe 학습용은 안 찍음). 평가용은 학습용이 끝난 뒤 따로 정한다.

## 나눔 (sel_train 성공 ep 1200 · 183,612 장, 프레임 수로 나눔)
A5000 은 6000pro 의 0.3 배 속도(사용자 실측) → GPU 하나당 6000pro GPU 의 0.3 몫. 모든 GPU 가 약 4 시간에 끝나게 잡았다.

| 서버 · GPU | ep 범위 (`--ep_start`..`--ep_end`, 끝 제외) | 장 수 |
|---|---|---|
| 6000pro GPU0 | 0 .. 395 | 57,408 (17:24 시작) |
| 6000pro GPU1 | 395 .. 753 | 57,614 (17:24 시작) |
| **A5000 GPU0** | **753 .. 856** | 17,178 |
| **A5000 GPU1** | **856 .. 960** | 17,099 |
| **A5000 GPU2** | **960 .. 1086** | 17,113 |
| **A5000 GPU3** | **1086 .. 1200** | 17,200 |

## 할 일
1. **코드** (6000pro 에서 커밋 → push → A5000 pull, 사용자 확인 후). 앞 문서 목록에 더할 새 파일:
   - `scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py` (조각 렌더 · 조립. 51 은 안 고침)
   - `scripts/apptainer/noeun_render_train_shards.sh` · `scripts/apptainer/A5000_RENDER_SHARDS.md`
   - `scripts/dataset_converters/nav3d/noeun/46_mark_narrow.py` · `walk_rbq_worker.py` 수정(`--narrow_only` 등, 기본 꺼짐)
   - `logs/env_cmp/tools/imgdiff.py` · `cmp_assembled.py` (비교 도구 — logs 가 git 에 안 들어가면 Gdrive)
2. **Gdrive** 로 A5000 에 줄 것: `logs/noeun/paths_landmark_ext/walk/sel_train.json` (34 MB, 걸은 자세가 다 들어 있다 — 렌더는 이것만 있으면 된다).
3. **A5000 시험** (GPU 1장): 6000pro 와 같은 ep 0..3 을 찍어 비교한다.
   `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json <sel_train.json> --out_root data/shard_test/A5 --log_dir logs/shard_test/A5 --ep_start 0 --ep_end 3`
   - 6000pro 의 같은 3 ep (Gdrive 로 받기: `data/noeun_landmark_v1/vlnce_train/.../episode_00000{0,1,2}_*`) 와 `python3 imgdiff.py <6000pro 폴더> <A5000 폴더>`.
   - 합격 기준 (6000pro 끼리 두 번 찍은 값): **depth 245/245 완전히 같음**, RGB 평균 차이 ≤ 0.5/255 (6000pro 끼리 0.27~0.43).
4. **본 렌더** — GPU 마다 1 개 (`CUDA_VISIBLE_DEVICES=<g>`), 위 표 범위. 모두 같은 `--out_root` · `--log_dir`:
   `... 51b_render_rbq_shard.py --walk_json <sel_train.json> --out_root data/noeun_landmark_v1/vlnce_train --log_dir logs/noeun/render_log/vlnce_train_shard --ep_start 753 --ep_end 856 --skip_existing`
   - `--skip_existing` = 그 ep 의 마지막 jpg·png 가 있으면 렌더 건너뜀 → 끊겨도 같은 명령으로 이어 찍는다.
   - 끝나면 로그 끝줄 `[51b] 렌더 103 ep (새로 103) · 17178 프레임` 처럼 장 수가 표와 같은지 확인.
5. **돌려줄 것** (Gdrive `gdrive:InternNav/data/noeun_landmark_v1/a5000_shards/`):
   - 사진: `<out_root>/traj_data/r2r/noeunstation/videos/chunk-000/observation.images.{rgb,depth}.60cm_15deg/episode_000{753..1199}_*` (tar 로 묶기, 약 68,590 장 × 2)
   - 자세: `<log_dir>/vlnce/sentinels/pose__60cm_15deg.part_*.npz` 4 개
   - 6000pro 에서 같은 위치에 풀고 `--assemble` 로 1200 ep 를 한 번에 조립한다 (모든 ep 의 pose 가 있어야 조립됨 — 빠지면 멈춘다).

## 지킬 것
- 6000pro 의 실행 중인 렌더(GPU 0·1)는 건드리지 않는다. 커밋·push 는 사용자 확인 후. 데이터 삭제 금지. 비밀번호를 파일·로그에 남기지 않는다.
