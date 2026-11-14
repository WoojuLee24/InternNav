# 노은역 랜드마크 경로 · 걷기 · 사진 — 2026-11-14, A6000 (6000pro)
리포트: https://claude.ai/artifact/2Vee7vw9GEHarBnbG8Q4EK

- 설정: 73 goals = logs/goal_candidates_spots6_buf50 (랜드마크 122 · 목적지 695, max_spots 6, 계단 0.5 m), 74 eval_over 1.5, paths_per_goal 3.
- 74b_make_paths_parallel.py (새): 목적지별 rng [seed,74,gi] → workers 수와 무관. 샘플 28개 w1 vs w8 세 파일 바이트 동일 (84 분 vs 15.5 분).
  다 찬 split 목적지는 풀지 않게 고침 (헛계산 제거, 결과 동일).
- 전체: logs/noeun/paths_landmark (평가 80 / 22 랜드마크 · 학습 1200 / 98) · 24 workers 99.6 분 · GOAL_ON_STAIRS 0.
  검증 logs/env_cmp/tools/verify_paths.py → verify/. 문제: 73 같은 랜드마크 안 spot 중복 29 (원본 384 에도 5), 평가 3 개가 다른 랜드마크 학습 목적지와 같은 자리 (누수), 계단 0.5 m 경계 3.
- 걷기·사진: rbq10 assets = gdrive .../landmark_house_20261113/rbq10_assets.tar → /data/users/dnwn24/assets (/ws/assets 로 bind),
  GS_USDZ = /data/users/dnwn24/data/GS_USDZ. skimage 는 apptainer_cache/noeun/pylib_isaac (+PYTHONPATH /pylib). 명령 앞부분 logs/env_cmp/isaac_walk_cmd.txt.
  샘플 42: 22/28 (넘어짐 6). 44+45 후 20/28 — 45 가 계단 경로 2 개를 넘어뜨림. 51 vlnpe 3 ep 323 장 165 s.
- 남은 결정: (A) 중복·누수 고친 뒤 경로 재생성, (B) 넘어짐 보충용 여유 경로, (C) 45 빼고 44 만.

## 추가 (2026-11-14 오후) — 74c · 개찰구 B · 전체 걷기 · 고르기 · 나눠 찍기
- 74c: 누수 12 평가A→평가B, 새로 학습 +350 · 평가A +35 · 평가B +88 → 학습 1,549 · 평가A 103 · 평가B 100 (`logs/noeun/paths_landmark_ext/`).
- 개찰구: 46_mark_narrow (폭<1.0 m ±1.5 m, 360/1752 ep) + walk `--narrow_only --steer_gain 1.5` (B). 좁은 ep 기본 197/360 → B 290/360. 기본 성공→B 실패 29.
- 전체 RBQ 걷기 성공 1,550/1,752 (88.5%) — 실제 물리 걷기 기준. 계단(층 이동) 399/455 가 가장 약함. "경로 그림 검사 99%"(03)는 다른 자(통과 여부 검사)라 비교하지 말 것.
- 고르기: 학습 1200 · 평가A 80 · 평가B 80 (`walk/sel_*.json`).
- 학습용은 vln_ce 만 (64 가 vln_ce 만 읽음). 51b_render_rbq_shard.py (새 파일): ep 범위 렌더 + part npz → `--assemble`. 51 과 비교: depth 동일, RGB 0.43/255 (51 끼리 0.27, 렌더 잡음), parquet·meta 동일.
  나눔: 6000pro GPU0 0-395 · GPU1 395-753 · A5000(0.3배 속도) 753-856/856-960/960-1086/1086-1200. 지시문 `scripts/apptainer/A5000_RENDER_SHARDS.md`.
- 라벨: ↑ 중 yaw>7.5° 165/117k (0.14%). **좌우 왔다갔다 43.5% of turns vs R2R 6.8%** (zigzag.py / zigzag_ref.py) — 결정 대기.
- 함정: walk frames[i].action = frame i-1→i 로 온 동작 (frames[0].action=-1). 반대로 맞추면 yaw 통계가 46% 처럼 엉터리가 된다.

## 18:50 왔다갔다(zigzag) 원인·수정 시험
- 원인: 회전 기준 7.5° = 회전 한 번(15°)의 절반 → 돌고 나면 반대쪽 경계에 걸려 되돌아 돎. 회전의 43.5% 가 왔다갔다 (R2R 6.8%). 옛 이상 이산화도 52% → 예전 데이터도 같은 문제였을 가능성 (측정 안 했었음).
- 렌더 중단 (vlnce_train 일부 사진 남음, 삭제 안 함). lookahead 는 효과 없어 코드 되돌림.
- 실제 걷기 48 ep (좁은16·계단16·평지16, --narrow_only --steer_gain 1.5):
  T7.5 성공 42/48 왔다갔다 41.6% 옆오차 p95 0.095 | T12 46/48 · 13.0% · 0.104 | T14 44/48 · 10.4% · 0.113
- 선택: 12° (성공 최고, 왔다갔다 1/3). 다음: 사용자 확인 후 1752 ep 한 번에 다시 걷기 (--turn_threshold_deg 12), A5000 은 동일성 시험 통과 후 뒤쪽부터.

## 21:10 12° 다시 걷기 끝 · 고르기 · 사진 시작
- 6000pro 앞 1,402 (worker 4+4, 18:40→21:04, load ≈ 110~120 · idle 40%): 성공 1,269. A5000 뒤 350 (worker 3, 91 분): 318. 합계 **1,587/1,752 (90.6%)** (예전 7.5° 1,550).
  종류별: 좁은 통로 300/360 · 계단 265/284 · 평지 1,022/1,108.
- 47_merge_select.py (새 파일) → walk_t12/{walk_all,sel_*,stats}.json. 학습 1,200 · 평가 A 80 · 평가 B 80 모두 채움.
- 라벨: 왔다갔다 학습 12.6% · 평가 A 11.8% · 평가 B 12.1% (예전 44%). 회전 비율 34%. ↑ 중 방향 > 7.5° 394/103,971 (0.38%, 대부분 좁은 통로).
- A5000 뒤쪽 결과 설정 확인 (args 동일), 왔다갔다 12.3%. 48 ep 같은지 시험 결과는 아직 못 받음.
- 사진 vln_ce 학습: data/noeun_landmark_v1/vlnce_train_t12 · 6000pro GPU0 0-398 · GPU1 398-757 (21:08 시작), A5000 757-861-963-1081-1200 (A5000_RENDER_T12.md).

## 개찰구 넘어짐 정리 (리포트 12 단계)
- 원인: 좁은 통로 안 제자리 회전 → 몸이 개찰구에 걸림 (대합실 x 14.5~17.7, y ≈ −2).
- 경로 쪽: 32 곧게 펴기 앞뒤 1.5 m (씀) · 44 출발점 옮기기 (씀, 개찰구엔 효과 없음) · 45 계단 펴기 (안 씀) · 개찰구 경로 제외 (안 씀) · 넉넉히 만들기 (씀).
- 걷기 쪽 (개찰구 실패 8 + 대조 8): 보정 어디서나 (라벨 어긋남 커서 안 씀) · **B 좁은 구간만 걸으며 보정 1.5 배 8/8 (씀)** · C 좁은 구간만 회전 기준 20° 5/8 (안 씀).
- 좁은 통로 360: 기본 197 → B 290 → B+12° 300. 좁은 구간 넘어짐 140 → 38 → 30. 남은 30 은 사진으로 원인 확인 필요. 평가 실행기에 보정 없을 수 있음 (미확인).
- 21:35 A5000 렌더 같은지 시험 탈락 (RGB 평균 차이 ≥ 1.9/255, 합격 ≤ 0.5 · 6000pro 끼리 0.27~0.43) → A5000 렌더 안 씀. 6000pro GPU0 0-398→757-963 · GPU1 398-757→963-1200 (이어서 자동). 실측 0.22~0.26 s/장 → 02:15~02:50 끝 예상.

## 2026-11-15 03:00 학습용 vln_ce 사진 끝 · Gdrive 업로드
- 6000pro GPU0 (0–397 → 757–962) · GPU1 (398–756 → 963–1199), 21:08 시작 → 02:38 끝, 오류 0.
- 조립 1200 ep · 159,022 프레임 (goal 보이는 비율 중앙값 0.865). rgb = depth = 159,022, parquet 1200 — 기대와 같음.
- 출력 `data/noeun_landmark_v1/vlnce_train_t12/` (+ `noeun_meta/`: sel_train · stats · paths · render_meta).
- Gdrive `gdrive:InternNav/data/vln_ce_noeun_landmark_t12_train.tar` 29,944,238,080 bytes (크기 같음), sha256 bcc8002d05b7be370eaa7c1e5a7b1d64634cdf3165296ab6aa2efb467786b224 (`.sha256` 같이 올림).
- 표본 눈 확인 (ep 5·500·850·1150 처음·중간·끝): 정상. ep 5 끝 프레임은 벽 바로 앞이라 흐림.
- 다음: H200 instruction 라벨링 (사용자). 평가 A·B 사진 형식 미정.

## 2026-11-15 06:20 instruction 라벨 시험 (b1 · post, 6000pro GPU0)
- 새 파일 `nav3d/noeun/62b_label_landmark.py`: 62 를 importlib 로 불러 `items()` 만 바꿔 끼움 (62 수정 없음). gen_hash 예전과 같음 (b1 0d611d54de08 · post 5bc6460a380b).
  - reference_path = 걸은 자세(시작 + action 1 프레임만, trunk z − 0.55) → ours_to_hab → RDP 0.3. start_rotation = 첫 프레임 몸 yaw (quat_from_nav3d_yaw). frame k 의 action = k 로 오게 한 동작 (0 번 -1).
  - 처음엔 모든 프레임을 써서 제자리 회전 흔들림(≤0.4 m)이 첫 구간이 됨 → 고침. 처음 회전 단어 일치 1109/1200.
- vLLM: 이 서버엔 없어서 `sif/vllm-openai-v0.24.0.sif` pull, `scripts/apptainer/noeun_vllm_gemma.sh <gpu> <port>` (bf16, GPU 1 장 64 GB).
- 시험 8 ep (0,1,4,6,10,15,148,371) × 3 × 2 = 48 문장 오류 0, 32·33 s. 출력 `data/noeun_landmark_v1/instruction_labels_test/`.
- 문제: 구역 없는 경로 107 (다른 출구 지상 바깥, y≈6.7, noeun_zones_v1 밖) · b1 거짓 no_landmark 로 5 회 재시도 7/24 · b1/post 는 stop_cue(우리 랜드마크) 를 안 받음.
- 결정 대기: 그대로 H200 vs 고쳐서 (no_landmark · 구역 v2 · stop_cue 옵션).

## 2026-11-15 07:00 구역 v2 · 멈출 곳 줄 · 역 단어 (사용자 결정: 바깥 = street, stop_cue 는 b1 만)
- `61b_extend_zones.py` -> `zones/noeun_zones_v2.json` (v1 점 그대로 + G-D street 1,402 · B1 49 · B2 45 점, walk_all 기준). route.yaml `rooms.noeun-v2` 추가. 구역 없는 경로 107 -> 0.
- route.yaml `stop_cue.noeun-paths-v1` (kind paths) 추가 — 62b 가 e2e.stop_cue_line 을 바꿔 끼워 paths json stop_cue 로 한 줄 (그냥 662 · 옆 물체 236 · 부분 162 · 몇 번째 140). 옆 물체 이름 = `landmark_nouns/landmark_names_v1.json` (goals.json 에서 뽑음). 방향 = 마지막 자세 기준, 뒤(>135°) 면 생략 193.
- `--landmark_nouns noeun-v1` (`landmark_nouns/noeun-v1.json` 148 단어) — 출력은 `nouns-noeun-v1/` 아래. route.yaml 추가 후에도 기존 gen_hash 그대로 (b1 0d611d54de08 · post 5bc6460a380b).
- 새 b1 (rooms-noeun-v2 · stop_cue-noeun-paths-v1, gen_hash 8ee0be42a6e4) 11 ep x 3: 한 번에 통과 33/33, 모든 문장이 랜드마크를 말함. 지어낸 말 조금 (ep 371 "on the floor", ep 16 "second set of stairs").
- 결정 대기: 후 를 구역 v2 · 역 단어로 바꿀지. 커밋 안 함. Gemma 서버 GPU0 켜져 있음.
- 07:15 후 도 구역 v2 · 역 단어 (사용자 결정): gen_hash b7b15a339bc3, 11 ep x 3 한 번에 통과 33/33, ep 6 "street". 후 는 멈출 곳 줄 없음 (ep 27 정답 second poster -> "glass railing").
- 정정: ep 16 "second set of stairs" 는 맞음 (계단 두 번, 층계참에서 오른쪽).
- H200 설정: b1 = `--landmark_nouns noeun-v1 --rooms noeun-v2 --stop_cue noeun-paths-v1` / 후 = `--landmark_nouns noeun-v1 --rooms noeun-v2`.

## 2026-11-15 08:00 H200 에서 라벨 → 학습 (사용자 결정: b1 · post 각각 랜드마크 줄 있음/없음, 400 · 1200, 공개 S2 출발 1 epoch)
- 62b: 후(post) 에도 `--stop_cue noeun-paths-v1` — joint.facts_memo · LP.joint_results · config.load 를 62b 안에서 바꿔 끼움 (facts+stop_cue 는 config 가 막던 조합). post+lm gen_hash 7743e94b44d6, 11 ep 한 번에 33/33.
- 라벨 4 종: b1 1dd95593a6ff · b1lm 8ee0be42a6e4 · post b7b15a339bc3 · postlm 7743e94b44d6 (모두 rooms noeun-v2 · nouns noeun-v1).
- 64b_build_train_landmark.py: `<mini>/vln_ce_noeun-lm{400,1200}-{label}/traj_data/r2r/landmark_s{0,1,2}` (형제별 scene). 시험(11 ep, --test_eps) loader_check PASS x4.
- 셀 8 개 `relabel_half_noeun/s1.noeun-lm{400,1200}-{b1,b1lm,post,postlm}.py` (make_cells_landmark.py), EXP 폴더 relabel_half_noeun_lm-v01. noeun_base.stage1 그대로 — val_ratio 0, 평가 없음.
- H200 안내: scripts/apptainer/H200_LABEL_T12.md.
