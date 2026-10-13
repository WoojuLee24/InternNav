# relabel 2x2 — 2026-10-13 부터 stage1 만 (stage2 생략)

사용자 결정 (2026-10-13): 새 label 의 relabel 2x2 실험은 **stage1 셀만** 학습/평가한다.
- 큐에 넣는 셀: `s1.train.<L>_eval.gt` (학습 + GT 평가), `s1.train.<L>_eval.<L>`, `s1.train.gt_eval.<L>`
- `s2.*` 셀(stage2 학습, stage2 평가)은 넣지 않는다.
- 적용: node3(v278fix_len-free), node4(v278fix_g4temp-100) 의 남은 s2 줄 3개씩을 주석 처리
  (`# [10-13 stage2 생략: ...]`, 백업 `queues/node{3,4}.txt.bak_1013_nos2`). node2 는 다른 세션이 이미 제거.

배경: relabel_half 결과에서 stage2 는 모든 셀에서 stage1 과 같은 순위 경향만 보이고 (예: GT 학습 s1 42.5 / s2 33.7,
v278fix 계열 s1 ~30-33 / s2 ~23-26), stage2 1 셀에 학습+평가 ~10h 가 든다.
