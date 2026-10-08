# 지도 3종 비교 종합 리포트 — navmesh vs 밴드 vs recast v2

2026-08-21 · **[Artifact 링크](https://claude.ai/code/artifact/aea56605-e2ef-4868-af5b-69f4939a988e)**

이 세션(#03·#04·gt_wall_clearance·벤치)의 비교 표를 한 페이지로 종합한 리포트.

## 담긴 내용
1. 세 지도의 질문 구조 비교 (밴드 = "장애물 없나" / recast = "걸을 바닥 있나")
2. 정확도 표 (#04): 밴드 95.56/75.83 → recast v2 96.23/**92.30**% · navmesh = 100% 상한 ·
   F/R/S 분해 · ablation 3단 · 게이트 G1~G3 (s8 G1 실패 → default 유지)
3. GT·follower 생존 표 (세 지도 모두 이탈 ≤1~2칸 — 원 증상 미재현)
4. 속도 벤치: 밴드 1.7~4.2 ms · navmesh 웜 37~73 ms · recast v2 17~41 ms(csum 캐시 후) —
   전부 렌더(~349 ms) 대비 소액 → **선택 기준은 속도가 아니라 정확도·제약**
5. navmesh r_b 실험 = 가능(이산 캐시, 0.01 격자화 우회) · 불가능한 것은 샘플별 연속 r_b뿐
6. 시나리오 추천표 (habitat+이산 → navmesh / Isaac·연속 r_b·다축 embodiment → recast v2)
7. 한계 (씬 2채 · 상한이 habitat · s8 잔여 F 745 · map_source='recast' 미연결)

수치 출처: [[260820_04_rewrite_result]] · [[260820_gt_wall_clearance_result]] ·
[[260821_map_bench_result]] · 원 리포트 링크는 Artifact 8절.
