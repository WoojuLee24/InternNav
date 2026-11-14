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
