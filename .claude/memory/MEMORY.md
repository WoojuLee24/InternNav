# Memory Index

- [VLN-PE GT Follower Agent Plan](project_gt_follower_agent.md) — GT oracle agent 구현 계획 (미완료), 플랜 파일: `/root/.claude/plans/vln-pe-gt-episode-misty-moon.md`
- [BEV Injection Architecture Plan](project_bev_injection.md) — Phase 1 구현 완료(feature/bev_v0.1, merge 대기); S1/S2 모드(fpv/bev/fpv_bev), 지원범위/커맨드/함정/Phase 2 과제
- [BEV Injection 관련 핵심 파일](reference_bev_key_files.md) — 구현 시 참조 파일 목록, Habitat LOOKDOWN 흐름, 체크포인트별 depth 사용 여부
- [기존 파일 수정 금지 — 신규 파일만 추가](feedback_new_files_only.md) — 신규 feature는 반드시 신규 파일 + config arg 선택 구조로 제안
- [교정에서 일반원리 추출](feedback_extract_principle_from_corrections.md) — `lesson` 스킬 사용, 즉흥적으로 하지 않고 정해진 절차로 원인파악+일반원리 추출
- [train_eval/qwenvl_train Python 구조](project_train_eval_qwenvl_python.md) — 셸 대신 Python으로 train+eval 한 프로세스 구동, Params 단일 소스로 train==eval 보장(batch_size 완료); 다음 단계는 config 통일
- [Quantization Kit (S2 벤치마크)](project_quantization_kit.md) — two-step look-down iteration, VLN-PE/VLN-CE 분리 데이터셋, 커맨드, 주요 파라미터

### Understanding (구 `.claude/understanding/`, Claude 작성 코드 이해 문서)
- [BEV Input Parameter](understanding_bev_input.md) — InternVLA-N1-w-NavDP vs DualVLN, H1/Habitat/dataset 파라미터 비교
- [runner.py 커맨드 구조](understanding_command.md) — config 하나로 train(torchrun)+eval(eval.py) 구동
- [S2 작동 흐름 (Dual System)](understanding_dual_system_flow.md) — 관련 파일/핵심 함수
- [Habitat Dual System 작동 흐름](understanding_habitat_dual_system_flow.md) — HabitatVLNEvaluator 관련 파일/핵심 함수
- [S2 입력 파이프라인](understanding_input.md) — internvla_n1_policy.py s2_step 5단계 변환 흐름

### Task Results (구 `.claude/tasks/*_result.md`, 완료된 작업 결과 보고)
- [S1/S2 통합 Image Provider 구현](260707_s1s2_bev_depth_panorama_result.md) — view x type x mode x combine
- [UnifiedImageProvider 전체 조합 스모크 테스트](260707_unified_provider_combos_result.md)
- [Habitat eval(rollout) 10개 조합 검증](260708_habitat_eval_unified_provider_result.md)
- [bev_gt 컬러/방향 불일치 조사 (1차)](260709_bev_gt_color_orientation_bug_result.md) — s1.bev.rgb.concat_s2.fpv (Isaac/H1)
- [image_base BEV occupancy 4개 config 검증](260709_image_base_occ_result.md)
- [image_base s1.fpv 스모크 테스트](260709_image_base_s1fpv_smoke_result.md)
- [S1 concat 모드 구현](260709_s1_concat_bev_result.md) — BEV를 3번째 토큰 슬롯으로 추가
- [bev_gt 컬러/방향 버그 수정 (2차)](260710_bev_gt_color_orientation_bug_result.md) — world-align 회전 버그 수정
