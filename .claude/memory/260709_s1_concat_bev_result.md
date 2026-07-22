# S1 concat 모드 (BEV를 3번째 토큰 슬롯으로 추가) — 구현 결과

계획: `/root/.claude/plans/s1-purrfect-torvalds.md` (siamese only, separate encoder는 범위 제외)

## 이전과 달라진 점

- `s1_image_view='bev'` + `s1_combine_mode='concat'`이 더 이상 `NotImplementedError`를 던지지 않고, BEV 이미지를 `[goal, cur]`에 3번째 슬롯으로 추가해서(`[goal, cur, bev]`) S1(`nextdit_async`)에 같이 먹입니다. 기존 `s1_combine_mode='replace'`(BEV가 FPV를 대체)는 완전히 그대로 동작합니다.
- `MemoryEncoder.memory_pos`(위치 임베딩)가 concat일 때만 `[512,384]→[768,384]`로 자동으로 넓어집니다 (`s1_image_view=='bev' and s1_combine_mode=='concat'`일 때만, 그 외엔 기존과 동일하게 512).
- 학습/eval 양쪽 다 지원 — `internvla_n1.py`의 `forward()`(학습)와 `generate_traj()`(eval) 둘 다 새 `_encode_s1_memory_tokens` 공유 헬퍼를 통해 3슬롯 stacking을 함.

## 바뀐 파일 (모두 최소 diff, guard clause 기반 — flag off일 때 기존 경로 100% 동일)

| 파일 | 변경 |
|---|---|
| `internnav/model/utils/visual_input_provider.py` | `S1VisualInput`에 `extra_images` 필드 추가 |
| `internnav/model/utils/unified_image_provider.py` | `concat`+`bev` 분기가 `_get_s1_bev_concat`(기존 `_get_s1_bev_replace` 재사용) 호출하도록 |
| `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` | `_s1_bev_concat()` 헬퍼 + `MemoryEncoder(max_len=768 if ... else 512)` (2곳) |
| `internnav/model/basemodel/internvla_n1/internvla_n1.py` | `_encode_s1_memory_tokens` 공유 헬퍼 추출, `forward()`/`generate_traj()`에 `traj_bev_images`/`bev_images` 파라미터 추가 |
| `internnav/model/basemodel/internvla_n1/internvla_n1_bev_provider.py` | `traj_bev_images` 순수 패스스루 (MRO 중간 클래스라 필수) |
| `internnav/model/basemodel/internvla_n1/internvla_n1_unified_provider.py` | `out.extra_images → traj_bev_images` 연결 |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator_unified.py` | `_apply_s1_provider` 3-tuple 반환, `generate_traj(..., bev_images=...)` |
| `scripts/train_eval/qwenvl_train/image_base/s1.bev.rgb.concat_s2.fpv.py` | docstring 갱신 (KNOWN LIMITATION 해제) |
| `scripts/debugging/unit_test/test_unified_image_provider.py` | 신규 테스트 2개 |

## 검증 커맨드 및 argument 의미

```bash
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/s1.bev.rgb.concat_s2.fpv.py --machine 5090 --no-eval --max-steps 3 --debug-dir <dir>
```
- `--config`: `s1_image_view='bev', s1_combine_mode='concat'`인 실험 config
- `--machine 5090`: 단일 GPU 인프라 프리셋
- `--no-eval`: 학습만 (평가 스킵)
- `--max-steps 3`: 스모크 테스트용 (3 step만 학습 후 종료)
- `--debug-dir`: BEV/fpv/depth 디버그 이미지 저장 경로

## 결과

1. **유닛 테스트**: `pytest scripts/debugging/unit_test/` 전체 51개 통과 (신규 2개 포함: `test_s1_bev_concat_returns_extra_images_not_replace`, `test_memory_encoder_widened_max_len_768`).
2. **concat 학습 스모크 (3 step)**: 정상 완료. `model.memory_encoder.memory_pos: [768, 384]` 확인. loss 0.94→0.97→0.81 (에러 없음).
3. **회귀 확인**:
   - `replace` config: `memory_pos: [512, 384]`(그대로), loss 1.05→1.03→1.09, 정상.
   - `none` config: `memory_pos: [512, 384]`(그대로), loss 1.00→1.00→1.60, 정상.
4. **디버그 이미지**: `step_XXXXXX_b00_3_bev_gt.jpg`(3번째 슬롯으로 들어가는 실제 BEV 이미지) 정상 생성 확인 — 상단 첨부 이미지처럼 유효한 top-down 투영.

## 추가 수정 (2026-07-09, eval 시 크래시 발견 후)

`python scripts/train_eval/qwenvl_train/runner.py --config .../s1.bev.rgb.concat_s2.fpv.py --machine 5090 --no-train --debugpy eval ...`로 실제 eval을 돌리면 `MemoryEncoder.forward`에서 `RuntimeError: size 768 vs 512` 크래시가 났음.

**원인**: `--no-train`만 주고 `--model-path`를 안 주면 학습된 체크포인트가 없어서 `default_model_path`(`InternVLA-N1-DualVLN`, concat으로 학습된 적 없는 기존 체크포인트)로 폴백함. 이 체크포인트의 `config.json`엔 `s1_combine_mode='concat'`이 없어서 `memory_encoder`가 512(2슬롯)로 재구성되는데, eval config는 concat(3슬롯=768)을 요구해서 충돌. **버그가 아니라 체크포인트/설정 불일치** — concat은 `replace`/`none`과 달리 아키텍처(`memory_pos` 크기) 자체가 바뀌므로, concat으로 학습 안 된 체크포인트로는 평가가 원천적으로 불가능함.

**수정**: `internvla_n1.py`의 `_encode_s1_memory_tokens`에 명시적 체크 추가 — 토큰 수(`slot*256`)와 `memory_encoder.memory_pos`의 크기가 안 맞으면, `MemoryEncoder.forward` 내부의 알아보기 힘든 broadcast 에러 대신 원인과 해결법을 설명하는 `RuntimeError`를 바로 던짐.

**실제로 concat eval을 돌리려면**: concat config로 실제 학습(스모크 테스트의 `--max-steps 3`는 체크포인트를 저장하지 않음 — `max_steps>0`이면 저장 스킵)해서 체크포인트를 만든 뒤, 그 체크포인트를 `--model-path`로 지정해서 eval하거나, `--no-train` 없이 train+eval을 한 번에 돌려야 함.

## End-to-end 검증 (2026-07-09, 1-step 학습 → 체크포인트 저장 → eval)

`runner.py`의 `--max-steps` 스모크 경로는 저장을 안 하므로, `Params.train_argv()`를 직접 불러 `--save_strategy steps --save_steps 1 --max_steps 1 --eval_strategy no --load_best_model_at_end False`로 override해서 1-step 학습 후 실제로 `checkpoint-1/`을 저장하고, 그 체크포인트로 eval까지 돌려서 확인함.

1. `checkpoint-1/config.json` 확인 → `s1_image_view=bev`, `s1_combine_mode=concat` 정상 저장됨 (pending-settings → config → checkpoint 라운드트립이 실제로 동작함을 증명).
2. `--no-train --model-path checkpoint-1`로 eval 1회 시도 → `KeyError: 'internvla_n1'` / `AutoProcessor`가 모델 타입을 인식 못 함. **원인은 concat 기능과 무관** — 직접 만든 학습 스크립트가 `runner.py`의 `_prep_ckpt_aux_files`(학습 후 `preprocessor_config.json`/`chat_template.json`을 체크포인트에 복사하는 단계)를 안 거쳐서 생긴 문제. `system2_ckpt`에서 두 파일을 수동으로 복사해 넣고 재시도.
3. 재시도 → **정상 완료**. `NE=7.11, SPL=0.000, SR=0.000` 출력, 에러 없음. 디버그 이미지(`step_XXXXXX_b00_3_bev_gt_cur.jpg`, S1에 실제로 들어가는 3번째 슬롯 BEV)도 정상 생성 확인.

이걸로 학습→저장→eval 전체 경로가 실제로 동작함을 확인했습니다 (기존 스모크 테스트는 학습만, 별도 eval 스모크는 저장 없이 default 체크포인트로만 확인했었음 — 이번에 이 gap을 메움).

## 범위 제외 (다음 작업 후보)
- separate encoder(`s1_bev_encoder`, 전용 `bev_model`) — siamese만 구현.
- `navdp_async` 동일 패턴 — 지금 미사용이라 범위 밖.
- eval 스모크 테스트(`--no-train --eval-max-episodes 1`)는 concat으로 실제 학습된 체크포인트가 있어야 가능 — 이번엔 3-step 스모크라 저장된 체크포인트 없음(원래 스크립트도 `max_steps>0`일 때 저장 안 함). 실 학습 후 별도 확인 필요.
