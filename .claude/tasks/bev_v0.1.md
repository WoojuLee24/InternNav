# Task: bev_v0.1 — BEV projection 모듈화 (Habitat / Isaac Sim / 학습 환경, S1+S2)

## 추가 요청 #4 (2026-06-05): 학습도 추론코드처럼 BEV
- 목표: 학습 경로가 추론과 동일한 모듈(`visual_input_provider.BEVProcessor`/`BEVImageProvider`)과
  동일한 모드 키(`bev_s1_mode`: fpv|bev|fpv_bev)를 사용하도록 신규 코드 작성. 기존 파일 수정 금지.
- 신규 파일:
  1. `internnav/model/basemodel/internvla_n1/internvla_n1_bev_provider.py` —
     `InternVLAN1BEVProviderForCausalLM(InternVLAN1ForCausalLM)`: forward에서 provider로 traj_images 치환
     + `apply_bev_to_traj()` (테스트 가능 함수) + `parse_bev_cli_args()` (CLI 사전 파싱)
  2. `internnav/trainer/internvla_n1_bev_provider_trainer.py` — thin launcher
     (기존 `internvla_n1_bev_trainer.py`와 동일한 monkey-patch 방식)
  3. `scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh` — 1GPU debug 실행 스크립트
  4. `tests/unit_test/test_bev_provider_training.py` — 학습↔추론 BEV 일치(parity) 검증 포함
- 설계 판단 (자율 결정, 근거):
  - 기존 `internvla_n1_argument.py`(ModelArguments)를 수정할 수 없으므로 신규 trainer가
    `parse_known_args`로 `--bev_*` 인자를 사전 추출 후 나머지를 base parser에 전달.
    추출된 설정은 모델 클래스의 pending-settings로 stash → `__init__`에서 `self.config`에 기록
    (checkpoint config.json에 저장되어 resume 시에도 유지).
  - 모드 매핑: bev_s1_mode bev=rgb_gt, fpv_bev=fpv_concat_gt 등가.
    `bev_image_type`(rgb|occ), `bev_depth_source`(gt|depthanythingv2)로 기존 5개 bev_mode 전부 커버.
  - dav2 depth 추정은 untracked 기존 파일(`internvla_n1_bev.py`)의 헬퍼를 lazy import —
    해당 파일은 커밋하지 않음(사용자의 기존 untracked 작업물).
  - S2 학습 주입은 데이터/프롬프트 파이프라인 변경이 필요해 이번 범위 제외(기록).
- 완료 조건: pytest 통과(기존+신규), import smoke, parity 테스트(학습 BEV == 추론 provider BEV)

### 검증 결과 #4 (2026-06-08, python3)
- 신규 파일 4개 (기존 파일 수정 0건):
  - `internnav/model/basemodel/internvla_n1/internvla_n1_bev_provider.py`
  - `internnav/trainer/internvla_n1_bev_provider_trainer.py`
  - `scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh`
  - `tests/unit_test/test_bev_provider_training.py`
- ✅ BEVProcessor ↔ 기존 학습 헬퍼(`_depth_rgb_to_bev_batch`/`_depth_to_bev_occ_batch`) parity = 0.0 (exact)
- ✅ `pytest tests/unit_test/test_bev_provider_training.py` — 13 passed
  (parity 3종: 학습 rgb/occ == legacy helper, 학습 processor == 추론 processor)
- ✅ `pytest tests/unit_test/` 전체 — 41 passed (회귀 없음)
- ✅ launcher 스크립트 실행: `--bev_*` 인자 추출/strip 후 base parser로 위임 정상
  (qwenvl_base import은 스크립트 실행 시 sys.path[0]로 해결 — legacy trainer와 동일 동작)
- 모드 매핑 (eval bev_s1_mode → legacy bev_mode): fpv→none, bev+rgb→rgb_gt,
  bev+occ→occ_gt, fpv_bev→fpv_concat_gt, +depthanythingv2 2종 = 기존 5개 전부 커버
- ⚠️ 실제 학습 e2e는 checkpoint(InternVLA-N1-System2)/데이터셋/GPU 멀티 필요로 미수행
- 참고: S2 학습 주입은 데이터·프롬프트 파이프라인 수정 필요 → 이번 범위 제외(추론 S2는 이미 지원)

## 작업 목표
memory `project_bev_injection.md` 및 계획 파일 `/root/.claude/plans/feature-bev-immutable-ritchie.md` 기반.
BEV projection을 다음 모든 경로에서 사용 가능하도록 모듈화:
- **Habitat** eval (`habitat_vln_bev` evaluator)
- **Isaac Sim / H1** eval (`internvla_n1_bev` agent + BEV policy)
- **학습 환경** (기존 `internvla_n1_bev_trainer.py` / `internvla_n1_bev.py`와 동일한 BEV 연산을 공용 모듈에서 가져다 쓸 수 있도록)
- S1(NavDP)·S2(LLM) 양쪽 주입 지원, config로 on/off

## 신규 파일 (기존 파일 수정 절대 금지)
1. `internnav/model/utils/visual_input_provider.py` — S1VisualInput, VisualInputProvider ABC, BEVProcessor, FPVProvider, BEVImageProvider, BEVFeatureProvider(stub), create_visual_provider()
2. `internnav/model/basemodel/internvla_n1/internvla_n1_policy_bev.py` — InternVLAN1NetBEV(InternVLAN1Net)
3. `internnav/agent/internvla_n1_agent_bev.py` — InternVLAN1AgentBEV, `@Agent.register('internvla_n1_bev')`
4. `internnav/habitat_extensions/vln/habitat_vln_evaluator_bev.py` — HabitatVLNEvaluatorBEV, `@Evaluator.register('habitat_vln_bev')`
5. (검증) `tests/unit_test/test_visual_input_provider.py` 또는 standalone 검증 스크립트

## 수정 가능 범위
- 위 신규 파일만. 기존 파일은 일절 수정 금지 (memory feedback_new_files_only.md).
- 신규 example config 파일 추가는 허용 (`scripts/eval/configs/*_bev_*.py`).

## 금지 사항
- `data/`, `checkpoints/`, `*.ply`, `*.parquet` 수정 금지
- dev/main/master push 금지
- `git add -A` / `git add .` 금지

## 완료 조건 / 검증
1. import smoke test: `/workspace/isaaclab/_isaac_sim/python.sh -c "from internnav.model.utils.visual_input_provider import create_visual_provider, BEVProcessor, FPVProvider, BEVImageProvider"`
2. BEVProcessor 수치 검증: 합성 depth/rgb로 BEV 출력 shape/값 확인 (depth_rgb_to_bev와 일치)
3. FPVProvider no-op 검증: get_s1_input → images=None (기존 경로 보존)
4. registry 검증: Agent registry에 'internvla_n1_bev', Evaluator registry에 'habitat_vln_bev' 등록 확인 (heavy deps import 실패 시 graceful 처리 기록)
5. `pytest tests/unit_test/` 기존 테스트 회귀 없음

## 진행 기록
- 2026-06-05: 브랜치 생성, 계획 수립
- 2026-06-05: 구현 완료 — 신규 파일 7개 (기존 파일 수정 0건)
  - `internnav/model/utils/visual_input_provider.py` (핵심 모듈, eval+학습 공용)
  - `internnav/model/basemodel/internvla_n1/internvla_n1_policy_bev.py` (H1 policy)
  - `internnav/agent/internvla_n1_agent_bev.py` (`internvla_n1_bev` 등록)
  - `internnav/habitat_extensions/vln/habitat_vln_evaluator_bev.py` (`habitat_vln_bev` 등록)
  - `scripts/eval/configs/habitat_dual_system_mini_5090_bev_cfg.py`
  - `scripts/eval/configs/h1_internvla_n1_async_bev_cfg.py`
  - `tests/unit_test/test_visual_input_provider.py` (20개 테스트)

## 구현 노트 (판단 근거)
- `Evaluator.register`의 decorator가 클래스를 반환하지 않아(`base.py` 버그성 동작)
  `@` 데코레이터 사용 시 클래스명이 None이 됨 → registry에서 부모 클래스를 조회하고
  서브클래스는 `Evaluator.register(...)(cls)` 형태로 사후 등록.
- H1 agent는 부모 `__init__`이 `get_policy()`로 policy를 생성하며 모델을 로드하므로,
  이중 로드를 피하기 위해 `internvla_n1_bev_trainer.py`와 같은 monkey-patch 방식으로
  `get_policy`를 일시 교체 (try/finally로 복원).
- S2 BEV 주입 시 `<image>` placeholder 수와 image 리스트 길이가 일치해야 하므로
  look-down turn에 BEV 이미지당 토큰 1개를 추가하는 방식으로 구현 (단순 insert 불가).
- Habitat S1/S2가 소비하는 프레임은 look-down 프레임(2×LOOKDOWN=60°) →
  pitch 기본값 `bev_cam_pitch_deg + 60`, `bev_s1_pitch_deg`/`bev_s2_pitch_deg`로 override 가능.
- 학습 연동: `BEVProcessor.bev_chw`/`occupancy`가 `internvla_n1_bev.py`의
  `rgb_gt`/`occ_gt` 모드와 동일 파라미터(z_min=-0.2, z_max=2.5) 배치 API 제공.
- depths_dp는 FPV 유지 (학습 rgb_gt 모드가 traj_depths를 바꾸지 않는 것과 동일 규약).
- 이 컨테이너에는 `/workspace/isaaclab/_isaac_sim/python.sh`가 없음 → 시스템 `python3`
  (torch 2.9.0+cu130, CUDA OK)로 검증 수행.

## 검증 결과 (2026-06-05, python3)
1. ✅ import smoke test 통과
2. ✅ `pytest tests/unit_test/test_visual_input_provider.py` — 20 passed
   (수치검증: depth_rgb_to_bev 일치, 단일점 기하 검증, depth_scale 등가성, intrinsics 리스케일)
3. ✅ `pytest tests/unit_test/` 전체 — 25 passed (기존 테스트 회귀 없음)
4. ✅ registry: Agent에 'internvla_n1_bev', Evaluator에 'habitat_vln_bev' 등록 확인
5. ✅ 두 신규 config 모두 eval.py 방식(importlib)으로 로드 성공
6. ✅ GPU smoke: cuda:0 + bfloat16 S1 stack [1,2,224,224,3] 정상
- ⚠️ 시뮬레이터 실 구동(e2e)은 checkpoint/scene 데이터 및 Isaac/Habitat 런타임 필요로 미수행

## 추가 요청 #3 (2026-06-05): S1/S2별 fpv / bev / fpv_bev 모드
- `bev_s1_mode`, `bev_s2_mode` config 키 추가 ('fpv' | 'bev' | 'fpv_bev')
  - S1 'fpv_bev': BEV를 T축 concat → [B, 2T, H, W, 3] (학습 fpv_concat_gt 규약, depth 복제)
  - S2 'bev': look-down FPV 프레임을 BEV로 대체 (prompt 토큰 정합 유지)
  - S2 'fpv_bev': FPV 뒤에 BEV 추가 (기존 동작)
  - legacy bool `bev_s1`/`bev_s2`는 하위호환 매핑 (False→'fpv')
- 검증: pytest 28 passed (모드별 테스트 6개 추가), 9개 모드 조합 생성 OK, config 로드 OK
