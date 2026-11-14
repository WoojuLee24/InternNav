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
