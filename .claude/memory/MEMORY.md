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
- [gs_vlnpe FPV/BEV 정합 검증 기하](understanding_gs_vlnpe_fpv_bev_geometry.md) — `action[t] @ diag(1,-1,-1,1)`이 진짜 c2w; FPV(상대)는 축 오류를 상쇄해 못 잡고 BEV/world(절대)만 잡음; mesh 앵커가 최종 판별 근거

- 리포트 링크 전체 목록: repo 파일 `scripts/dataset_converters/gs_vlnpe/pipeline_reports.md` (단일 소스 — 48개 링크 + 로컬 경로 + 권장 커맨드)

### Task Results (구 `.claude/tasks/*_result.md`, 완료된 작업 결과 보고)
- [S1/S2 통합 Image Provider 구현](260707_s1s2_bev_depth_panorama_result.md) — view x type x mode x combine
- [UnifiedImageProvider 전체 조합 스모크 테스트](260707_unified_provider_combos_result.md)
- [Habitat eval(rollout) 10개 조합 검증](260708_habitat_eval_unified_provider_result.md)
- [bev_gt 컬러/방향 불일치 조사 (1차)](260709_bev_gt_color_orientation_bug_result.md) — s1.bev.rgb.concat_s2.fpv (Isaac/H1)
- [image_base BEV occupancy 4개 config 검증](260709_image_base_occ_result.md)
- [image_base s1.fpv 스모크 테스트](260709_image_base_s1fpv_smoke_result.md)
- [S1 concat 모드 구현](260709_s1_concat_bev_result.md) — BEV를 3번째 토큰 슬롯으로 추가
- [bev_gt 컬러/방향 버그 수정 (2차)](260710_bev_gt_color_orientation_bug_result.md) — world-align 회전 버그 수정
- [gs_vlnpe 03 GT path 재현](260803_gs_vlnpe_03_reproduce_result.md) — M1.3 C1 완료; A* 20/20, **같은 루트 40/40**(격자 원점을 월드 정렬로 바꿔 마지막 1건 해결), 하드 충돌 0. **지표는 기준선 ⓒ와 함께 봐야 한다** — "A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값"(chamfer 0.072/0.100, 헤딩 6°). 실측은 그 1.45×/1.27×. **GT는 cubic spline이다**(knot 복원 40/40, 간격 0.8 m) — "컨트롤러 궤적" 추정 정정.  `refine_radius` 0.15→0.10(GT 역추정 R*=0.05 + r_b 하한). **기준선을 낮추는 것 ≠ GT에 가까워지는 것** — refine을 min_move로 바꾸면 기준선은 0.080→0.058이지만 실제 경로는 0.107→0.142로 악화(argmax가 루트를 복도 중심선으로 정규화하기 때문). 기본값 argmax 유지. **원칙: 논문이 숫자로 준 값(voxel 0.05, A* 0.2)은 고정, 안 준 값(원점 등)만 조정** — A* 0.1이 더 좋지만 채택 안 함. **bezier 스무딩**은 오버슈트 없어 clearance·헤딩이 낫지만 GT보다 짧고 덜 꺾임(①0.67) → cubic 유지, 지표① 게이트를 0.5~2.0 양쪽으로. **ESDF 맵 5건(floor_z 씬단일·평면가정·표면샘플구멍·r_b·다운샘플) 전부 고칠 것 없음** — 씬1/씬2가 레버마다 반대로 움직여 전역 파라미터로는 한계. 다음은 씬 수 확대
- [gs_vlnpe VLN-PE 데이터셋 지원 + 측정오류 정정](260807_gs_vlnpe_vlnpe_dataset_result.md) — `--dataset {vln_n1,vln_pe}` 추가(신규 `dataset_utils.py` registry, 로더 4중복 통합). vln_pe 포맷 실측 확정(wxyz 쿼터니언→OpenCV `R[:,[1,2,0]]*[-1,-1,1]` mesh-anchor 검증, intrinsic fx=128 USD 유도, depth=raw×10). **⚠️ 초기 진단 3건이 오류였고 문서에 정정 표 있음** — "가구 관통/문턱 밟기"는 cam_xyz(몸통보다 0.2m 앞)로 잰 측정 오류(body_xyz로는 0건), "씬2 배회가 원인"도 오진(진짜는 03이 씬 대표 floor_z 하나만 써서 다락을 1층 맵으로 계획한 다층 버그). **진짜 원인**: r_b=0.25 과대(H1 통과 clearance 0.141~0.158)→0.20으로 씬1 20/20; 다층 floor_z 자동감지→씬2 같은루트 1/11→9/14; 천장 어두움은 `/rtx/sceneDb/ambientLightIntensity`=0(raw SimulationApp이 IsaacLab kit 우회)이고 **씬 로드 후** 적용해야 먹힘. **rgb/depth 비정렬은 실재**(대조군: vln_n1은 (0,0) peak 1.00x, vln_pe는 매 프레임 ±4px 상이 + VLN-PE 코드 자체 `delay by get_rgb` 주석, 사후보정 불가). three_light는 down disk가 바닥을 과하게 밝혀 상하명암이 GT와 반대(−15.7 vs GT +51.1)→`--light ambient_only --rtx_ambient 10` 권장. vln_pe는 `--refine_radius 0.30`이 chamfer·clearance 동시 개선(vln_n1은 반대라 0.10 유지)
- [gs_vlnpe 04 Isaac Sim 렌더러 — M1.4 완료(2-A/2-B PASS)](260805_gs_vlnpe_04_render_obs_isaac_result.md) — raw `isaacsim.SimulationApp`(AppLauncher 우회), depth는 두 씬 모두 PASS(mm 단위). 조명은 DomeLight→3-light 카메라추적→**신규 `04c_light_explorer.py`(GT/렌더/차이 HTML 비교, 9개 옵션)로 육안 비교 후 DomeLight(2M) 최종 확정**(camera_light가 수치는 더 좋았으나 육안상 "너무 강함"). **RGB 판정을 edge_corr(엄격, FAIL 0.19~0.22)에서 SSIM(관대, PASS 0.65~0.71)로 교체** — 사용자 요청으로 재계산해보니 edge_corr가 과도하게 엄격한 지표였음이 확인됨. **2-B(mesh-anchor 생성 경로)도 Isaac 렌더러로 처음 실행해 두 씬 다 PASS**(mesh-anchor median 0.0009~0.0016m). RTX 셰이더 최초컴파일·`simulation_app.close()` 지연·**print 버퍼링 손실(SIGKILL 시 로그 진행상황 안 보임)** 셋 다 "가짜 실패"(버그 아님, report.html 파일로 직접 확인해야). RTX `/rtx/useViewLightingMode`(camera light)는 GUI 프리뷰 아니라 실제 렌더 설정. USD xform op 순서(rotate/orient가 translate보다 먼저) 반대로 하면 위치 틀어지는 버그 실측 발견
- [gs_vlnpe 04 렌더링(2-A GT검증+2-B 생성, Open3D)](260804_gs_vlnpe_04_render_obs_result.md) — M1.4 완료. Open3D OffscreenRenderer(멀티머티리얼은 read_triangle_model 필수, look_at은 world벡터라 축변환 불필요). 2-A/2-B 전부 PASS. **버그 수정 3건**: ① depth 1px 오프셋(Open3D set_projection이 픽셀 모서리 컨벤션 — cx/cy +0.5px로 MAE 0.0074→0.00005) ② 밝기(indirect_light_intensity 재보정, 최종 310000) ③ **rgb 명암대비 과다**(vln_n1 GT rgb 자체가 실사진 아닌 합성데이터임을 재확인 → ColorGrading REINHARD 톤매핑으로 std비율 1.3~1.4x→0.977). `synthesize_action_poses`(위치+tangent yaw+pitch)는 analytic 유도를 4개 에피소드 역산으로 검증(회전오차 3e-8)
- [gs_vlnpe 03 C2(random) 구현](260804_gs_vlnpe_03_c2_random_result.md) — 버그 수정: navigable이 "장애물없음"과 "mesh데이터 없는 건물밖 허공"(씬2의 ~40%)을 구분못해 랜덤 start/goal이 밖으로 샘 → compute_scan_coverage_mask 추가로 해결
- [정합 문제는 눈으로 판단 않고 실측](feedback_verify_alignment_quantitatively.md) — matchTemplate/MAE 스윕으로 픽셀오프셋·밝기 확정, 매끈한 신호는 변별력 낮음 주의 — M1.3 마지막 조각. 무작위 h_b/pitch + navigable 최대 연결성분에서 start/goal(≥2m), reproduce와 계획 코드경로 동일(guard clause만). GT 없어 분포로만 검증(하드게이트는 r_b/충돌 유지). **새 발견**: 무작위 start/goal이 reproduce보다 훨씬 자주 r_b 위반 병목을 지나감(씬1 20개 중 7개) → 재추출(reject-and-resample)로 해결. 두 씬 20/20 PASS
- [gs_vlnpe 03d 8축 그리드서치](260804_gs_vlnpe_03d_grid_search_result.md) — GT/기준선ⓒ/생성경로 3자 거리를 astar_cell_m/refine_radius/connectivity/clearance_weight/spacing/smooth_step/h_nav/r_b/downsample_mode 8축에서 독립 스윕(51 조합, 9중 회귀 통과). **connectivity=4가 실제 하드충돌 1건**(새 발견), `d(ⓒ,path)`로 "refine=정규화" 메커니즘 수치 확인, clearance tie-break bias는 개선 없음. 기본값 불변 — 씬1/씬2가 계속 반대로 움직여 전역 파라미터 한계 재확인
- [gs_vlnpe 재현 가능성의 한계](260804_gs_vlnpe_reproducibility_limits.md) — "논문과 완전히 동일해도 GT와 bit-for-bit 같을 수 없는 이유" 정리. 원점·tie-break·refine반경·솎기 규칙이 논문에 미기재(측정으로 확인) + refine 목적함수가 "GT재현"이 아니라 "장애물에서 멀어지기"라 충실한 구현일수록 GT에서 멀어지는 역설. "GT=컨트롤러 실행궤적이라 못맞춘다"는 이전 가설은 반증됨(GT는 cubic spline으로 잘 재현됨)
- [gs_vlnpe 02 ESDF 맵](260803_gs_vlnpe_02_esdf_result.md) — M1.2 완료; occupancy 저장 + GT 궤적 clearance 게이트, h_nav/h_obs 밴드, 입력은 obj
- [gs_vlnpe 01_prepare_scene 씬 정합 게이트](260802_gs_vlnpe_01_prepare_scene_result.md) — M1.1 완료; USD/정규화 불필요로 게이트만 남김, mesh 표면거리 <1mm, negative test 포함. **camera_extrinsic = 에피소드별 로봇 키 h_b + 카메라 pitch**(2026-08-03 정정)
- [h1 eval hang 원인 조사](260724_h1_eval_hang_investigation_result.md) — env.step() 내부 Isaac stall, system memory 초과 아님(배제), watchdog 대응
