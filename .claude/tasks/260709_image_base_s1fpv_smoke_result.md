# image_base/ 의 s1.fpv 조합 3-step 스모크 테스트 + 시각화 — 결과

계획: `/root/.claude/plans/python-scripts-train-eval-qwenvl-train-r-majestic-sunbeam.md`

## 요약

`image_base/`의 s1.fpv 5개 config를 3-step 스모크 테스트로 돌리던 중, `s2_image_view='bev'`와 `'bev_ld'`가 학습 시 **픽셀 단위로 동일한 이미지**를 만드는 버그를 발견 → 원인 확인 → 수정 → 재검증까지 완료.

## 발견한 버그

**Eval**(`habitat_vln_evaluator_unified.py:423-427`)은 `s2_image_view`에 따라 소스를 이미 올바르게 구분하고 있었음:
```python
if self.s2_source_view == 'bev_ld':
    s2_src_rgb, s2_src_depth = look_down_rgb_np, look_down_depth_m
else:
    s2_src_rgb, s2_src_depth = fpv_rgb_np, fpv_depth_m   # 'bev'도 'fpv'도 여기로
```
**Training**(`internvla_n1_lerobot_dataset.py`)은 `s2_image_view` 값을 전혀 보지 않고 항상 `lookdown_image`(pitch_2)만 `get_s2_extra()`에 넘기고 있었음 → `s1.fpv_s2.bev.rgb.concat`(`s2_image_view='bev'`), `s1.fpv_s2.depth.colormap.concat`, `s1.fpv_s2.depth.normalized.concat`(둘 다 `s2_image_view='fpv'`) 3개 config가 **학습 때는 lookdown, eval 때는 FPV**를 쓰는 학습/평가 입력 불일치 상태였음 (`CLAUDE.md` 가이드라인 5번 위반). `bev_ld`는 학습/eval 둘 다 lookdown이라 원래 정상.

추가로, BEV 투영에 쓰이는 `cam_pitch_deg`도 학습 시 항상 `0.0` 고정이었음(실제 프레임 촬영 각도인 `pitch_1`/`pitch_2`를 무시) — eval은 `bev_ld`일 때 lookdown offset을 반영하므로 이것도 프레임 실제 각도와 다른 값을 쓰는 기하학적 오류였음.

## 수정 내용

1. **`internnav/model/utils/unified_image_provider.py`**: `get_s2_extra()`에 optional `cam_pitch_deg: Optional[float] = None` 파라미터 추가 (없으면 기존처럼 `self.s2_pitch_deg` 사용 — 하위호환 100%), `processor.for_s2(...)` 호출에 전달.
2. **`internnav/dataset/internvla_n1_lerobot_dataset.py`**: `__getitem__`의 S2 이미지 주입 블록에서 `self._s2_provider.s2_view == 'bev_ld'`일 때만 기존 lookdown 소스(`lookdown_image`/`depth_image`, `cam_pitch_deg=pitch_2`)를 쓰고, 그 외(`bev`/`fpv`)는 FPV 소스(원본 pitch_1 프레임 `fpv_image_raw` + 별도 로드한 FPV depth, `cam_pitch_deg=pitch_1`)를 쓰도록 분기 추가. `pitch_1`/`pitch_2`는 이미 `__getitem__`에 샘플별로 있던 값(모델 쪽 S1 BEV 경로 `traj_cam_pitch_2`와 동일 패턴 재사용) — 새 CLI 인자나 config 필드 추가 없음.

기존 로직(`s2_combine_mode=='none'`이거나 `bev_ld` 자체)은 100% 그대로 — guard clause로 새 분기만 추가.

## 검증 커맨드

```
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/<config>.py --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/<name>/train
```
- `--machine 5090`: 단일 GPU 인프라 프리셋 / `--no-eval`: 학습만 / `--max-steps 3`: 3-step 스모크 / `--debug-dir`: 자동 디버그 이미지 저장 경로 (이번엔 처음부터 `.../train`로 지정해서 이전 세션의 `eval/` 잔여물과 안 섞이게 함)

## 결과 (수정 후 재실행, 5/5 PASS)

| config | S2 축 | loss (3 step) | 소스 이미지 확인 |
|---|---|---|---|
| `base_s1.fpv_s2.fpv.py` | none (baseline no-op) | 1.07→1.07→1.39 | S2 없음 — 영향 없음 (guard clause가 `combine=='none'`에서 조기 리턴) |
| `s1.fpv_s2.bev.rgb.concat.py` | bev/rgb/raw/concat | 1.12→1.16→1.10 | **수정 전엔 lookdown(복도) 프레임이었으나, 수정 후 FPV(욕실) 프레임으로 정상 변경** — eval과 동일 소스가 됨 |
| `s1.fpv_s2.bev.ld.rgb.concat.py` | bev_ld/rgb/raw/concat | 1.11→1.04→1.42 | 그대로 lookdown(복도) 프레임 유지 — bev와 이제 서로 다른 이미지 |
| `s1.fpv_s2.depth.colormap.concat.py` | fpv/depth/colormap/concat | 1.11→1.12→1.50 | FPV(욕실) 프레임으로 정상 변경, JET 컬러맵도 새 장면과 기하학적으로 일치 |
| `s1.fpv_s2.depth.normalized.concat.py` | fpv/depth/normalized/concat | 1.19→1.15→1.49 | FPV 소스로 정상 변경 |

5개 모두 Traceback 없이 3 step 완주, loss NaN 없음.

## 시각적 확인

- `s1.fpv_s2.bev.rgb.concat`(`train/s2_step_000000_1_fpv.jpg`)과 `s1.fpv_s2.bev.ld.rgb.concat`(`train/s2_step_000000_1_fpv.jpg`)를 나란히 비교 — **이제 서로 다른 장면**(전자: 욕실/샤워부스, 후자: 복도/기둥). 수정 전엔 두 파일이 픽셀 단위로 동일했음.
- `s1.fpv_s2.depth.colormap.concat`의 `_1_fpv.jpg`도 동일하게 욕실 장면으로 바뀌었고, `_2_colormap.jpg`(JET)가 그 장면의 근/원 구조(문=근, 벽=원)와 기하학적으로 일치함을 확인.

## 시각화 결과 경로

```
logs/input_v0.1/image_base/
  base_s1.fpv_s2.fpv/train/                 step_*_b0N_{1_fpv,2_gt_depth,5_gt_topdown,w5_gt_topdown_world}.jpg (S1만, S2 없음)
  s1.fpv_s2.bev.rgb.concat/train/           step_*(S1) + s2_step_*_{1_fpv,2_bev_gt,5_gt_topdown,w5_gt_topdown_world}.jpg(S2, FPV 소스)
  s1.fpv_s2.bev.ld.rgb.concat/train/        step_*(S1) + s2_step_*_{1_fpv,2_bev_gt}.jpg(S2, lookdown 소스)
  s1.fpv_s2.depth.colormap.concat/train/    step_*(S1) + s2_unified_step_*_{1_fpv,2_colormap}.jpg(S2, FPV 소스)
  s1.fpv_s2.depth.normalized.concat/train/  step_*(S1) + s2_unified_step_*_{1_fpv,2_normalized}.jpg(S2, FPV 소스)
```

## 변경 파일

- `internnav/model/utils/unified_image_provider.py` — `get_s2_extra()`에 optional `cam_pitch_deg` 파라미터 추가
- `internnav/dataset/internvla_n1_lerobot_dataset.py` — S2 소스 선택을 `s2_view=='bev_ld'` 기준으로 FPV/lookdown 분기, `cam_pitch_deg`를 프레임 실제 각도(`pitch_1`/`pitch_2`)로 전달

## 범위 밖

`s2_combine_mode='replace'`는 여전히 학습 데이터 미지원(`NotImplementedError`, 기존 그대로) — 이번 수정과 무관.

## Update: image_base/ 전체(7개, `_check` 제외) train+eval 시각화 디버깅

계획: `/root/.claude/plans/python-scripts-train-eval-qwenvl-train-r-majestic-sunbeam.md`

범위를 넓혀 `image_base/`의 나머지 config까지 **train(3-step) + eval(2-episode) 둘 다** 실행. `base_s1.fpv_s2.fpv.py`/`s1.bev.rgb.concat_s2.fpv.py`는 `logs/input_v0.1/image_base/`에 `_check` 폴더가 있어(다른 세션이 작업 중) 스킵.

### 실행 커맨드

```
# train (3-step smoke)
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/<name>.py --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/<name>

# eval (2-episode debug rollout, 기본 체크포인트로 시각화 검증용 — SR/SPL 무의미)
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/<name>.py --machine 5090 --no-train --model-path checkpoints/InternVLA-N1-DualVLN --max-steps 2 --debug-dir logs/input_v0.1/image_base/<name>
```
`--debug-dir`(서브폴더 없이 지정)은 `--no-eval`/`--no-train` 여부에 따라 runner가 자동으로 `.../train` 또는 `.../eval`을 붙임(다른 세션이 이번 실행 사이에 이 편의 기능을 추가함). `--model-path checkpoints/InternVLA-N1-DualVLN`은 이 config로 학습된 체크포인트가 아니라 기본 체크포인트 — S2 concat/S1 replace 모두 아키텍처를 바꾸지 않는 조합이라 크래시 없이 로드되며(`260708_habitat_eval_unified_provider_result.md`에서 이미 확인), 목적은 디버그 이미지 파이프라인 검증이지 SR/SPL이 아님.

### 결과 (7/7 config, train+eval 모두 PASS)

| config | 상태 | train 확인 | eval 확인 |
|---|---|---|---|
| `s1.bev.rgb.replace_s2.fpv` | 신규 실행 | loss 1.24→1.23→1.26, `unified` 없음(replace라 S1 자체 debug, `step_*_{1_fpv,2_gt_depth,3_bev_gt}.jpg`) | NE 3.77→1.91, SPL 0.356, SR 0.5 — `step_*_{1_fpv,3_bev_gt}.jpg` 육안 확인: 바닥/러그 전환부가 BEV와 기하학적으로 일치 |
| `s1.depth.colormap.replace_s2.fpv` | 신규 실행 | loss 1.03→1.01→1.03, `unified_step_*_2_depth_raw.jpg`가 실제로는 JET 컬러맵 이미지(내용은 맞음, **파일명만 항상 "depth_raw"로 하드코딩** — `internvla_n1_unified_provider.py:171`, cosmetic, 기능 버그 아님) | NE 6.95→4.31, SPL 0.331, SR 0.5 — eval 쪽은 mode 무관 공용 `step_*_{1_fpv,2_gt_depth,3_bev_gt}.jpg`만 생성(이전 세션에서 이미 문서화된 "eval 경로엔 S1 mode별 debug 없음" 제약, 버그 아님) |
| `s1.depth.normalized.replace_s2.fpv` | 신규 실행 | loss 1.17→1.16→1.45 | NE 7.20→3.89, SPL 0.418, SR 0.5 |
| `s1.fpv_s2.bev.rgb.concat` | train 기존 검증분 재사용, eval 신규 | (위 Update 이전 섹션 참고) | NE 7.11→5.40, SPL/SR 0.0(이 2-episode 샘플은 실패) — `s2_step_*_1_fpv.jpg`가 Habitat 라이브 FPV(우드패널 거실)이고 `_2_bev_gt.jpg`가 그 바닥/가구를 정확히 투영 — eval 쪽은 원래부터 올바르게 FPV를 썼으므로 회귀 없음 확인 |
| `s1.fpv_s2.bev.ld.rgb.concat` | train 기존 검증분 재사용, eval 신규 | (위 참고) | NE 7.19→5.23, SPL/SR 0.0 — Traceback 없이 완주만 확인(별도 시각 비교는 train 단계에서 이미 함) |
| `s1.fpv_s2.depth.colormap.concat` | train 기존 검증분 재사용, eval 신규 | (위 참고) | NE 7.11 — `s2_unified_step_*_{1_fpv,2_colormap}.jpg`: 우드패널 거실 FPV와 JET depth가 문/아치(근)·벽(원) 구조로 정확히 일치 |
| `s1.fpv_s2.depth.normalized.concat` | train 기존 검증분 재사용, eval 신규 | (위 참고) | NE 7.11 — Traceback 없이 완주 |

전부 Traceback/ChildFailedError 없이 완주. eval SR/SPL은 미학습 기본 체크포인트라 참고용(디버그 이미지 파이프라인 검증이 목적).

### 시각화 결과 경로 (신규 3개)

```
logs/input_v0.1/image_base/
  s1.bev.rgb.replace_s2.fpv/train/            step_*_b0N_{1_fpv,2_gt_depth,3_bev_gt}.jpg
  s1.bev.rgb.replace_s2.fpv/eval/             step_*_b00_{1_fpv,2_gt_depth,3_bev_gt,5_gt_cam,w3_bev_world_gt,w5_gt_topdown_world}.jpg (405장)
  s1.depth.colormap.replace_s2.fpv/train/     unified_step_*_{1_fpv,2_depth_raw(=실제 colormap),3_depth_adapted}.jpg
  s1.depth.colormap.replace_s2.fpv/eval/      step_*_{1_fpv,2_gt_depth,3_bev_gt}.jpg (S1 mode별 eval debug 없음, 기존 제약)
  s1.depth.normalized.replace_s2.fpv/train,eval/  위와 동일 패턴
  s1.fpv_s2.bev.rgb.concat/eval/              s2_step_*_{1_fpv,2_bev_gt,5_gt_cam,w2_bev_world_gt,w5_gt_topdown_world}.jpg
  s1.fpv_s2.bev.ld.rgb.concat/eval/           s2_step_*_{1_fpv,2_bev_gt}.jpg
  s1.fpv_s2.depth.colormap.concat/eval/       s2_unified_step_*_{1_fpv,2_colormap}.jpg
  s1.fpv_s2.depth.normalized.concat/eval/     s2_unified_step_*_{1_fpv,2_normalized}.jpg
```

### 발견한 사소한 이슈 (수정 안 함, 보고만)

- `internvla_n1_unified_provider.py:171`의 `_save_debug()`가 S1 depth 디버그 파일명을 `s1_image_mode` 값과 무관하게 항상 `_2_depth_raw.jpg`로 저장함 — 실제 저장되는 이미지 내용은 mode(raw/normalized/colormap)에 맞게 정확히 처리됨, 파일명만 안 맞는 cosmetic 이슈.

### 스킵한 config

`base_s1.fpv_s2.fpv.py`, `s1.bev.rgb.concat_s2.fpv.py` — `logs/input_v0.1/image_base/`에 `_check` 폴더 존재(다른 세션 작업 중), 사용자 지시대로 건드리지 않음.

## Update: `s1.fpv_s2.bev.rgb.concat` / `s1.fpv_s2.bev.ld.rgb.concat` train 폴더 소실 → 재실행

위 검증 직후 사용자가 두 config의 `train/` 폴더가 사라졌다고 보고. 확인 결과:

- **코드 버그 아님** — 당시 train은 정상 완료했고 이미지도 육안 확인까지 마쳤던 것(위 기록 참고).
- `base_s1.fpv_s2.fpv`, `s1.bev.rgb.replace_s2.fpv`, `s1.depth.colormap.replace_s2.fpv`, `s1.depth.normalized.replace_s2.fpv`는 폴더 전체가, `s1.fpv_s2.bev.rgb.concat`/`s1.fpv_s2.bev.ld.rgb.concat`는 `train/`만 사라짐 (`eval/`은 유지). `.Trash-1000`에도 없어 `rm`류로 삭제된 것으로 보임.
- `logs/input_v0.1/image_base/`에 이번 작업과 겹치는 config 이름의 `_check` 접미사 폴더(`s1.bev.rgb.replace_s2.fpv_check` 등)가 내 작업과 초 단위로 겹치는 시각에 새로 생성된 것을 발견 — 같은 파일시스템을 공유하는 다른 세션이 동시에 유사한 작업을 수행 중이었던 것으로 추정. 정확히 어떤 명령이 지웠는지는 특정 못함.
- 두 config 모두 3-step train 재실행 완료(`--no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/<name>`), Traceback 없이 정상, `train/s2_step_000000_1_fpv.jpg` 재확인 결과 이전과 동일하게 `bev`=욕실(FPV)/`bev_ld`=복도(lookdown)로 정상 구분됨 — 수정 사항 회귀 없음.

## Update: `bev_ld`에 S2 GT topdown 디버그 이미지가 없던 버그 발견·수정

재실행 확인 도중 사용자가 `s1.fpv_s2.bev.ld.rgb.concat`에 tdmap(GT topdown 참조 이미지)이 없다고 보고.

**원인**: `internvla_n1_lerobot_dataset.py:1278`의 S2 GT topdown 디버그 저장 조건이 `getattr(self._s2_provider, 's2_view', None) == 'bev'`로, `'bev_ld'`를 빠뜨리고 있었음. `unified_image_provider.py`의 `get_s2_extra()`는 이미 `if self.s2_view in ('bev', 'bev_ld'):`로 두 view를 동일하게 취급하는데, 이 topdown 저장 조건만 `'bev'` 하나만 하드코딩되어 있어 `bev_ld`는 BEV 이미지(`s2_step_*_2_bev_gt.jpg`)는 저장되면서 그 스케일/방향 비교용 GT topdown(`s2_step_*_5_gt_topdown.jpg`, `_w5_gt_topdown_world.jpg`)만 누락되는 상태였음.

**수정**: 조건을 `in ('bev', 'bev_ld')`로 변경 (기존 `unified_image_provider.py`의 동일 패턴과 일치시킴).

```python
# before
and self._s2_provider is not None and getattr(self._s2_provider, 's2_view', None) == 'bev'):
# after
and self._s2_provider is not None
and getattr(self._s2_provider, 's2_view', None) in ('bev', 'bev_ld')):
```

**검증**: `s1.fpv_s2.bev.ld.rgb.concat` 3-step train 재실행 — `s2_step_*_5_gt_topdown.jpg`/`_w5_gt_topdown_world.jpg` 각 14장 정상 생성 확인. `_w5_gt_topdown_world.jpg`의 forward 화살표와 `_2_bev_gt.jpg`의 V자 투영 방향이 기하학적으로 일치함을 육안 확인 — `bev` config와 동일한 패턴으로 정상 동작.

**변경 파일**: `internnav/dataset/internvla_n1_lerobot_dataset.py` (조건 1곳만 수정, guard clause 로직 자체는 그대로).
