---
name: BEV Injection Architecture Plan
description: InternVLA-N1 S1/S2 BEV 주입 — Phase 1 구현 완료 (feature/bev_v0.1, 2026-06-05), 모드/지원범위/커맨드/미구현 항목
type: project
---
# BEV Visual Input 주입 — Phase 1 구현 완료

## 상태 (2026-06-05)
브랜치 `feature/bev_v0.1`에 구현 완료, merge 대기 중 (사용자 승인 필요).
기존 파일 수정 0건, 신규 파일 7개. 검증: pytest 28 passed (기존 회귀 없음).
태스크 기록: `/ws/src/InternNav/.claude/tasks/bev_v0.1.md`

## 신규 파일 (구현됨)
| 파일 | 내용 |
|---|---|
| `internnav/model/utils/visual_input_provider.py` | 핵심: BEVProcessor, VisualInputProvider ABC, FPVProvider, BEVImageProvider, BEVFeatureProvider(stub), create_visual_provider() |
| `internnav/model/basemodel/internvla_n1/internvla_n1_policy_bev.py` | InternVLAN1NetBEV — s2_step/s1_step_latent 오버라이드 |
| `internnav/agent/internvla_n1_agent_bev.py` | `@Agent.register('internvla_n1_bev')` (H1/Isaac) |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator_bev.py` | `Evaluator.register('habitat_vln_bev')` (Habitat) |
| `scripts/eval/configs/habitat_dual_system_mini_5090_bev_cfg.py` / `h1_internvla_n1_async_bev_cfg.py` | 예시 config (registry 등록 import 포함 — 제거 금지) |
| `tests/unit_test/test_visual_input_provider.py` | 28개 중 신규 테스트 (수치/기하/모드 검증) |

## 모드 (S1/S2 독립, 9조합)
config 키: `bev_s1_mode`, `bev_s2_mode` ∈ 'fpv' | 'bev' | 'fpv_bev'
- S1 'bev': [goal_bev, cur_bev] 대체 (depth는 FPV 유지 = 학습 rgb_gt 규약)
- S1 'fpv_bev': T축 concat [B,2T,H,W,3] (= 학습 fpv_concat_gt 규약, depth 복제) — **fpv_concat 학습 ckpt 필요**
- S2 'bev': look-down FPV를 BEV로 대체 / 'fpv_bev': FPV 뒤 BEV 추가 (<image> 토큰 정합 유지)
- eval↔학습 bev_mode 대응: fpv↔none, bev↔rgb_gt, fpv_bev↔fpv_concat_gt

### 축 정리 (혼동 주의)
직교 축 4개: ① RGB 입력 구성 `bev_s1_mode/bev_s2_mode`(fpv|bev|fpv_bev) ② BEV 렌더 `bev_image_type`(rgb=컬러|occ=흑백 점유맵) ③ depth 출처 `bev_depth_source`(gt|depthanythingv2, **BEV 만드는 재료**용) ④ 대상 시스템(S1/S2 독립). "9조합"은 ①을 S1×S2로 곱한 3×3. legacy `bev_mode` 5개 = ①×②×③ 매핑(_MODE_EQUIV).
- **depth를 'fpv/bev'로 고르는 축은 없음.** `apply_bev_to_traj`는 어떤 모드든 `traj_depths`를 그대로 반환 → BEV로 바뀌는 건 RGB 스트림(`traj_images`)뿐. depth는 두 역할만: (a) BEV 만드는 재료, (b) 모델 depth 입력=항상 FPV 원본.

### "depth는 FPV 유지 = rgb_gt 규약"의 정확한 의미
`rgb_gt` 모드는 모델의 두 입력 스트림 중 RGB(`traj_images`)만 BEV로 교체, depth(`traj_depths`)는 FPV 원본 유지. 학습이 이 형식만 봤으므로 eval도 같게 줘야 함(규약=train/eval 입력형식 일치). depth까지 BEV로 바꾸면 미학습 분포 → 성능 깨짐.

### 시점 혼합 어색함 + ckpt별 depth 사용 (사용자 의문 → 검증 결과)
- "BEV-RGB + FPV-depth"는 **좌표계가 섞임**(위에서 본 RGB vs 1인칭 depth) → 직관적으로 어색한 게 맞음 = 디자인 스멜.
- 단 NavDP backbone(`navdp_backbone.py:151-193` DAT_RGBD_Patch_Backbone)은 **픽셀 단위 RGBD 융합이 아님**: `rgb_model`/`depth_model` 별도 ViT → 각 256 토큰 → `torch.cat(dim=1)` 토큰 concat. "RGB픽셀=depth픽셀" 강제 안 함 → 시점 불일치가 치명적이진 않음. 학습으로 각 스트림 용도(BEV-RGB=레이아웃, FPV-depth=즉각 장애물) 분담 학습됨.
- 어색함 피하려면 `fpv_bev`(=fpv_concat_gt)가 가장 일관: 원본 FPV RGBD 쌍 유지 + BEV를 추가 프레임으로만.
- **ckpt별 depth 입력 여부 (BEV 제약이 ckpt 따라 다름)**:
  - `InternVLA-N1-w-NavDP`(`navdp_async`): FPV depth를 모델 입력으로 **사용**(`internvla_n1.py:290-298` depths_dp) → "depth FPV 유지" 규약이 진짜 중요.
  - `InternVLA-N1-DualVLN`(`nextdit_async`): `generate_traj`(L349 `depths_dp=None`)가 RGB만 씀, **외부 FPV depth 입력 안 받음**(필요시 내부 DepthAnythingV2 추정) → "depth FPV 유지"는 해당 없음. 상세표는 [[reference_bev_key_files]].

## 지원 범위 (정직한 현황)
- Habitat eval: ✅ S1+S2 / Isaac Sim H1 eval: ✅ S1+S2
- 학습 S1: ✅ **신규 provider 경로** `internvla_n1_bev_provider_trainer.py` + `internvla_n1_bev_provider.py`
  — 추론과 동일한 `BEVProcessor`·동일 모드 키(`bev_s1_mode`) 사용, parity=0.0 테스트 보장.
  (legacy `internvla_n1_bev_trainer.py` + `--bev_mode`도 그대로 유지)
- ❌ S2 학습 주입 미구현 (데이터/프롬프트 파이프라인 필요) → S2 BEV eval은 zero-shot
- ❌ feature-level (BEVFeatureProvider = NotImplementedError stub, Phase 2)

## 학습 신규 파일 (요청 #4, 2026-06-08)
| 파일 | 내용 |
|---|---|
| `internnav/model/basemodel/internvla_n1/internvla_n1_bev_provider.py` | `InternVLAN1BEVProviderForCausalLM` — forward에서 BEVProcessor로 traj_images 치환 + 순수함수 `apply_bev_to_traj()` + `parse_bev_cli_args()` |
| `internnav/trainer/internvla_n1_bev_provider_trainer.py` | launcher — `--bev_*` 사전 파싱·strip 후 base train() 위임 (monkey-patch) |
| `scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh` | 1GPU debug 실행 |
| `tests/unit_test/test_bev_provider_training.py` | 13 tests (parity 포함) |
- CLI 인자: `internvla_n1_argument.py` 수정 불가 → launcher가 `parse_known_args`로 `--bev_*`만 추출,
  모델 `config`에 stash (checkpoint config.json 저장 → resume 유지)

## 커맨드 (모두 한 줄, copy-paste 안전)
### Habitat 추론
`python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_5090_bev_cfg.py`

### Isaac Sim (H1) 추론
`/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_bev_cfg.py`

### 학습 (추론과 동일 모듈 BEVProcessor 사용, parity=0.0)
신규 provider 방식 (권장 — eval과 BEV 일치):
`bash scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh data/InternData-N1-v0.5-mini/vln_ce bev`
또는 직접 실행 (스크립트로 실행해야 qwenvl_base import 해결):
`python internnav/trainer/internvla_n1_bev_provider_trainer.py --bev_s1_mode bev --bev_image_type rgb --bev_depth_source gt --deepspeed scripts/train/qwenvl_train/zero2.json --model_name_or_path checkpoints/InternVLA-N1-System2 --vln_dataset_use r2r_125cm_0_30 --data_root data/InternData-N1-v0.5-mini/vln_ce --seed 42 --num_history 4 --system1 nextdit_async --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --output_dir checkpoints/debug/bev_s1_provider --num_train_epochs 1.0 --eval_strategy no --save_strategy steps --save_steps 500 --learning_rate 1e-4 --bf16 --dataloader_num_workers 0 --report_to none --logging_steps 1`
- `--bev_s1_mode`: fpv(기준선,기존동작) | bev | fpv_bev / `--bev_image_type`: rgb | occ / `--bev_depth_source`: gt | depthanythingv2
- (legacy 방식도 유지) `internvla_n1_bev_trainer.py` + `--bev_mode rgb_gt` 명시 — train_bev_debug.sh는 미지정 시 none

### 공통
- 기준선(fpv, 기존 동작) 비교: eval config에서 `visual_provider='fpv'` 또는 `bev_s1_mode/bev_s2_mode='fpv'`; 학습은 `--bev_s1_mode fpv`
- 테스트: `python3 -m pytest tests/unit_test/test_visual_input_provider.py tests/unit_test/test_bev_provider_training.py` (총 41 passed)

## S2 prompt/image dump 디버그 모드 (eval 전용, 2026-06-08 구현)
S2 LLM이 매 스텝 실제로 본 prompt·sources·입력이미지를 저장. `debug_modes`에 `prompt` 토큰 추가로 ON (기존 `bev`와 병행 가능: `"bev,prompt"`). habitat+isaac 둘 다 지원, 신규 파일만 추가(기존 무수정).
- 저장 레이아웃: `<debug_dir>/s2_step_XXXXXX/` → `prompt.txt`(apply_chat_template 결과), `sources.txt`(마지막 user turn 텍스트=sources[0]['value'] 상당), `output.txt`(llm_output), `images/NN.jpg`(모델 입력 이미지 순서대로), `meta.json`(episode_idx/look_down/이미지수/`<image>`토큰수/output_pixel·action)
- 신규 파일: `internnav/model/utils/prompt_dump.py`(PromptDumper) / `internvla_n1_policy_debug.py`(InternVLAN1NetDebug, isaac) / `internnav/agent/internvla_n1_agent_debug.py`(`@Agent.register('internvla_n1_bev_debug')`) / `habitat_vln_evaluator_debug.py`(`Evaluator.register('habitat_vln_debug')`) / `tests/unit_test/test_prompt_dump.py`(4 passed) / 예시 config 2개
- 커맨드 (Habitat): `python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_5090_debug_cfg.py`
- 커맨드 (Isaac/H1): `/workspace/isaaclab/_isaac_sim/python.sh scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_debug_cfg.py`
- 캡처 seam (함정): isaac는 policy `s2_step` 오버라이드 후 explicit `dump()`; habitat은 evaluator가 S2를 인라인 → `tokenizer.decode` autodump 트리거. **`processor.__call__`은 dunder라 인스턴스 오버라이드 불가** → 클래스 레벨 패치(+`_active_dumper` 스코프 가드)로 이미지 캡처. `apply_chat_template`·`decode`는 일반 메서드라 인스턴스 오버라이드 OK.

### 학습 경로 prompt-dump (2026-06-09 구현)
학습 샘플의 대화(prompt/sources)와 traj_images를 dataset 레벨에서 저장. eval과 동일하게 `debug_modes`에 `prompt` 토큰으로 ON.
- 저장: `<debug_dir>/sample_XXXXXX/` → `prompt.txt`(대화 렌더링), `sources.txt`(첫 human turn=sources[0]['value'] 상당), `images/NN.jpg`(traj_images 프레임), `meta.json`. **인덱스 키 폴더 → DataLoader worker-safe·결정적**. `max_samples`(기본 20)까지만.
- 신규 파일: `prompt_dump.py`의 `TrainingPromptDumper` / `internnav/dataset/prompt_dump_dataset.py`(`NavPixelGoalDatasetPromptDump`+`set_training_dumper`) / `internnav/trainer/internvla_n1_prompt_dump_trainer.py`(launcher) / `scripts/train/qwenvl_train/bev/train_prompt_dump_debug.sh`. 테스트는 `test_prompt_dump.py`에 추가(총 6 passed).
- 커맨드: `bash scripts/train/qwenvl_train/bev/train_prompt_dump_debug.sh [data_root] [bev_s1_mode]` (또는 launcher 직접: `--debug_modes prompt --debug_dir ... --prompt_dump_max_samples 20` + 기존 `--bev_*`)
- launcher는 BEV provider 셋업(`internvla_n1_bev_provider_trainer`와 동일) + dataset 클래스 monkey-patch 합성 → BEV 학습 + prompt dump 동시 가능. `--debug_modes/--debug_dir`은 실제 ModelArgument라 peek만(strip 안 함), base parser도 받음.
- **clean sources seam**: collator는 input_ids만(decode시 image pad 폭증) → clean `sources[0]['value']`는 `preprocess_qwen_2_visual` 안에서만 존재. 그래서 그 함수를 wrap해 `_last_sources` stash + dataset `__getitem__` 직후 dump. traj_images는 **pre-BEV FPV**(forward에서 BEV 치환) → 실제 BEV 이미지는 `debug_modes='bev'` forward dump로(병행 `"bev,prompt"`).
- 범위: VLN(`NavPixelGoalDataset`)만. iign(`VLLNDataset`) 미적용.

## S1 BEV ablation 학습+habitat eval 스크립트 (2026-06-09)
b4_eff128_base.sh(H200 8GPU, nextdit_async=DualVLN) 기반. 학습 가능 조합 = S1 fpv/bev (S2 BEV 학습 미구현 → 제외). 신규 파일만, 코드 변경 0.
- 학습: `scripts/train/qwenvl_train/bev/train_{fpv,bev}_b4_eff128.sh` — launcher만 `internvla_n1_bev_provider_trainer.py`로 교체 + `--bev_s1_mode {fpv|bev} --bev_image_type rgb --bev_depth_source gt`. 두 스크립트는 `--bev_s1_mode`만 차이(ablation). 학습 끝나면 best ckpt로 habitat eval 자동 호출.
- eval: `scripts/train/qwenvl_train/bev/eval_habitat_{fpv,bev}.sh` — `eval.py --config <bev cfg> --bev_s1_mode .. --bev_s2_mode .. "$@"`. bev.sh=bev/fpv(BEV_S1/BEV_S2 env로 조절 가능), fpv.sh=fpv/fpv(provider no-op).
- **eval 모드 선택 방식 = eval.py CLI arg override** (사용자 최종 결정): `--bev_s1_mode`/`--bev_s2_mode` 추가(eval.py:48-60 argparse, :150-153 apply — `--model_path` 패턴). **config 1개 고정**(`scripts/eval/configs/habitat/dualvln_system_mini_5090_bev_cfg.py`, visual_provider/bev_*+DualVLN model_path 보유)로 모든 조합 커버, ckpt는 `--model_path` override.
- 시도했다 폐기: 모드별 config 파일 / env-driven config (둘 다 거절). eval.py 무수정 원하면 config의 bev_s1/s2_mode 직접 수정도 가능하나, **arg 방식으로 확정**.
- 검증: config compile + 4 shell `bash -n` OK.

## 구현 시 발견한 함정 (재작업 시 필수 참고)
- `Evaluator.register` decorator가 클래스를 반환하지 않음 → `@` 사용 시 클래스명이 None. 부모는 `Evaluator.evaluators['habitat_vln']`로 조회, 등록은 `Evaluator.register('habitat_vln_bev')(cls)` 사후 호출
- H1 agent: 부모 `__init__`이 get_policy()로 모델 로드 → 이중 로드 방지 위해 `get_policy` monkey-patch (try/finally 복원, internvla_n1_bev_trainer.py 선례)
- S2 주입은 `<image>` placeholder 수 = 이미지 리스트 길이 정합 필수 (단순 insert 불가, 토큰 추가 방식)
- Habitat S1/S2 프레임은 look-down 프레임(2×LOOKDOWN=60°) → pitch 기본값 `bev_cam_pitch_deg + 60` 자동 설정 (`bev_s1_pitch_deg`/`bev_s2_pitch_deg` override). 이전 계획의 "habitat pitch 0°"는 실제 소비 프레임 기준으로 보정됨
- depth 단위 규약: get_s1_input=metric(m), get_s2_extra=raw×depth_scale (H1: 10.0, Habitat evaluator는 m 전달 + s2_depth_in_meters 강제)

## Phase 2 (미구현)
`internvla_n1_model_bev.py` — `InternVLAN1ForCausalLMBEV.generate_traj(precomputed_features=...)` 오버라이드로 DINOv2 bypass + BEVFeatureProvider 구현. S2 학습 주입도 후속 과제.

관련: [[reference_bev_key_files]], [[feedback_new_files_only]]
완전 계획 파일: `/root/.claude/plans/feature-bev-immutable-ritchie.md`
