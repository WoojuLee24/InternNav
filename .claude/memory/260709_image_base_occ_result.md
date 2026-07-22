# 결과 보고: image_base BEV occupancy(occ/occ.binary) × concat/replace 4개 config 신규 생성 + train/eval 디버그 검증

관련 이전 작업: [260707_unified_provider_combos_result.md](260707_unified_provider_combos_result.md) (구 `image_old/` 디렉토리에서 occ 계열 검증), [260709_s1_concat_bev_result.md](260709_s1_concat_bev_result.md) (bev+concat 아키텍처 구현/검증), [260709_image_base_s1fpv_smoke_result.md](260709_image_base_s1fpv_smoke_result.md) (image_base 리네이밍 이후 최초 스모크)

## 이전과 달라진 점

`image_base/`(리네이밍 이후, 점(.) 구분 네이밍)에는 지금까지 `s1.bev.rgb.*`(컬러 BEV)만 있었고 occupancy(`s1_image_type='depth'`) 조합이 하나도 없었음. 이번에 4개 신규 config 추가:

| 파일 | s1_image_view | s1_image_type | s1_image_mode | s1_combine_mode |
|---|---|---|---|---|
| `s1.bev.occ.replace_s2.fpv.py` | bev | depth | raw | replace |
| `s1.bev.occ.concat_s2.fpv.py` | bev | depth | raw | concat |
| `s1.bev.occ.binary.replace_s2.fpv.py` | bev | depth | binary | replace |
| `s1.bev.occ.binary.concat_s2.fpv.py` | bev | depth | binary | concat |

"occ"는 별도 `type` 값이 아니라 `s1_image_view='bev' + s1_image_type='depth'`의 의미(코드상 `_OCC_MODE_MAP`)이며, "occ" vs "occ.binary"는 `s1_image_mode`(`raw` vs `binary`)로 구분됨. 모두 `bev_depth_source='gt'` 포함, S2는 기존 `fpv` no-op 그대로.

## 실행 커맨드 및 argument 의미

```
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/<config>.py --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/<name>
```
- `--config`: 검증할 config 파일
- `--machine 5090`: 단일 GPU 인프라 프리셋 (train/eval 배치 크기 등 결정)
- `--no-eval` / `--no-train`: 각각 학습만 / 평가만 실행 (평가만 실행 시 `--debug-dir`에 자동으로 `/eval` 서브디렉토리 추가됨)
- `--max-steps 3`: 스모크 테스트 — 학습 3-step으로 제한(체크포인트 저장 안 함), 평가도 3 episode로 제한
- `--debug-dir`: BEV/fpv/depth 디버그 이미지 저장 경로
- `--model-path`: (concat eval만) 사용할 체크포인트 명시 지정

**Train 검증** (4개 모두): `--no-eval --max-steps 3`
**Eval 검증 — replace 2개**: `--no-train --max-steps 3` (기본 체크포인트 `InternVLA-N1-DualVLN` 사용 — 아키텍처 영향 없음)
**Eval 검증 — concat 2개**: `--no-train --max-steps 3 --model-path /ws/src/InternNav/checkpoints/image_base/s1.bev.rgb.concat_s2.fpv/checkpoint-1` (사용자 확인 후 결정: 재학습 대신 기존 rgb-concat 체크포인트 재사용 — 아래 참고)

## concat eval에 기존 rgb-concat 체크포인트를 재사용한 이유

`s1_combine_mode='concat'`은 `MemoryEncoder.memory_pos`를 512→768로 넓히는 **아키텍처 변경**이라(`s1_image_view=='bev' and s1_combine_mode=='concat'`일 때만, `image_type`/`mode`와 무관), concat으로 실제 학습된 체크포인트가 있어야 eval이 가능함(안 그러면 `RuntimeError`로 명확히 실패 — [260709_s1_concat_bev_result.md](260709_s1_concat_bev_result.md) 참고). `--max-steps 3` 스모크는 체크포인트를 저장하지 않으므로, 매번 새로 concat 학습을 돌리는 대신 이전 세션에서 이미 저장된 `s1.bev.rgb.concat_s2.fpv/checkpoint-1`(1-step 학습, `config.json`에 `s1_combine_mode='concat', s1_image_view='bev'` 확인됨)을 재사용. 이 체크포인트는 RGB 데이터로 학습됐지만 `memory_pos` 크기는 `image_type`과 무관하므로 occupancy 입력에도 아키텍처적으로 호환됨. 단, vision encoder 가중치는 RGB 분포로 학습돼 있어 occupancy 입력에 대해서는 out-of-distribution — **loss/SPL/SR 수치는 무의미**하며, 목적은 오직 "코드 경로가 크래시 없이 돌고 디버그 이미지가 올바르게 생성되는지" 확인.

## 결과 요약 (8/8 PASS)

### Train (4개, `--no-eval`)

| config | 결과 | memory_pos | 시각화 |
|---|---|---|---|
| `s1.bev.occ.replace_s2.fpv` | PASS, loss 1.11→1.46 | [512, 384] | 108장 (`step_*_{1_fpv,2_gt_depth,3_bev_gt,5_gt_topdown,w3_bev_world_gt,w5_gt_topdown_world}.jpg`) |
| `s1.bev.occ.concat_s2.fpv` | PASS, loss 1.34→1.47 | **[768, 384]** (자동 확장 확인) | 108장, 동일 패턴 |
| `s1.bev.occ.binary.replace_s2.fpv` | PASS, loss 1.26→1.16 | [512, 384] | 108장 |
| `s1.bev.occ.binary.concat_s2.fpv` | PASS, loss 1.12→1.49 | **[768, 384]** | 108장 |

### Eval (4개)

| config | 체크포인트 | 결과 |
|---|---|---|
| `s1.bev.occ.replace_s2.fpv` | 기본(`InternVLA-N1-DualVLN`) | PASS, NE=1.81, SPL=0.570, SR=0.667 (3 episode 평균 진행) |
| `s1.bev.occ.binary.replace_s2.fpv` | 기본 | PASS, NE=2.98, SPL=0.577, SR=0.667 |
| `s1.bev.occ.concat_s2.fpv` | 재사용 concat ckpt | PASS(크래시 없음), NE=7.11, SPL=0.000, SR=0.000 (수치 무의미 — 위 설명 참고) |
| `s1.bev.occ.binary.concat_s2.fpv` | 재사용 concat ckpt | PASS(크래시 없음), NE=7.11, SPL=0.000, SR=0.000 |

## 시각화 확인

- `3_bev_gt.jpg`(train, raw occ): 회색=free, 흰색 해칭=occupied(카메라 FOV 콘 형태), 검정=unknown — 기하학적으로 정상.
- `3_bev_gt.jpg`(train, binary occ): 3채널을 [free,occ,unk]→RGB로 매핑해 녹색=free, 빨강=occupied, 파랑=unknown으로 표시 — raw와 동일한 콘 형상이 색상만 다르게 나타남, 정상.
- `3_bev_gt_cur.jpg`(eval): train과 동일한 콘 형상 확인(binary=녹색 free 콘, raw=흰색/회색 free 콘) — replace와 concat 양쪽 다 occupancy 내용 자체는 동일하게 생성됨(3번째 슬롯으로 들어가느냐 대체되느냐만 다름).
- concat 2개는 `memory_pos`가 학습 로그에서 `[768, 384]`로 정상 확장됨을 직접 확인(replace/원본 결과 512와 대조).

## 시각화 결과 경로

```
logs/input_v0.1/image_base/
  s1.bev.occ.replace_s2.fpv/{train,eval}/
  s1.bev.occ.concat_s2.fpv/{train,eval}/
  s1.bev.occ.binary.replace_s2.fpv/{train,eval}/
  s1.bev.occ.binary.concat_s2.fpv/{train,eval}/
```

## 변경/신규 파일

- 신규: `scripts/train_eval/qwenvl_train/image_base/s1.bev.occ.replace_s2.fpv.py`
- 신규: `scripts/train_eval/qwenvl_train/image_base/s1.bev.occ.concat_s2.fpv.py`
- 신규: `scripts/train_eval/qwenvl_train/image_base/s1.bev.occ.binary.replace_s2.fpv.py`
- 신규: `scripts/train_eval/qwenvl_train/image_base/s1.bev.occ.binary.concat_s2.fpv.py`
- 코드 변경 없음 (기존 occupancy 파이프라인이 이미 `image_base`의 `s1_image_view/type/mode/combine` 축을 통해 완전히 지원하고 있었음 — config 4개 추가만으로 충분).
