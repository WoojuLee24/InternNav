# Memory Index

- [VLN-PE GT Follower Agent Plan](project_gt_follower_agent.md) — GT oracle agent 구현 계획 (미완료), 플랜 파일: `/root/.claude/plans/vln-pe-gt-episode-misty-moon.md`
- [BEV Injection Architecture Plan](project_bev_injection.md) — Phase 1 구현 완료(feature/bev_v0.1, merge 대기); S1/S2 모드(fpv/bev/fpv_bev), 지원범위/커맨드/함정/Phase 2 과제
- [BEV Injection 관련 핵심 파일](reference_bev_key_files.md) — 구현 시 참조 파일 목록, Habitat LOOKDOWN 흐름, 체크포인트별 depth 사용 여부
- [기존 파일 수정 금지 — 신규 파일만 추가](feedback_new_files_only.md) — 신규 feature는 반드시 신규 파일 + config arg 선택 구조로 제안
- [train_eval/qwenvl_train Python 구조](project_train_eval_qwenvl_python.md) — 셸 대신 Python으로 train+eval 한 프로세스 구동, Params 단일 소스로 train==eval 보장(batch_size 완료); 다음 단계는 config 통일
