# 결과 보고: UnifiedImageProvider 전체 조합 3-step 스모크 테스트

관련 계획/이전 결과: [260707_s1s2_bev_depth_panorama.md](260707_s1s2_bev_depth_panorama.md), [260707_s1s2_bev_depth_panorama_result.md](260707_s1s2_bev_depth_panorama_result.md)

## Update (같은 날, 사용자 피드백 반영 후)

최초 실행(20개 중 13 PASS / 7 FAIL) 이후 사용자 피드백을 받아 아래 4개를 추가로 수정함:

1. **`s1.depth_colormap_s2.fpv` (bf16→numpy 버그) 수정 완료, PASS로 전환.** `unified_image_provider.py::_depth_to_colormap`에서 `.cpu().numpy()` 전에 `.float()` 캐스팅 추가. 재실행 확인: 72장 이미지 생성, loss 정상.
2. **bev-view + `concat` 6개: 실제 T-alignment 재설계 대신, `fpv+depth+concat`과 동일한 `NotImplementedError` 가드만 추가**(사용자 확인 후 이 방식으로 진행 결정). `nextdit_async` 헤드에서 `traj_hidden_states`는 `traj_poses`의 T로 repeat되는데 `memory_tokens`는 `traj_images`의 (concat으로 2배 된) T를 쓰기 때문에 `torch.cat` batch 차원이 어긋남 — `internvla_n1.py` forward 핵심 로직을 재설계해야 하는 범위 밖 작업이라 크래시 대신 명확한 에러 메시지로 문서화. 재실행 확인: 이제 크래시 대신 깔끔한 `NotImplementedError`로 실패.
3. **S1 `depth_raw`/`depth_adapted` 디버그 이미지가 흰색으로 washed-out되던 문제 수정.** `internvla_n1_unified_provider.py::_save_debug`에서 저장 직전에만 per-frame `/max()` 정규화 추가 (모델이 실제로 받는 텐서는 그대로, 디버그 jpg만 보기 좋게).
4. **S2 디버그 이미지에 원본 프레임이 없어 S1과 비교 불가능하던 문제 수정.** `unified_image_provider.py::get_s2_extra`에서 `s2_unified_step_XXXXXX_1_fpv.jpg`(원본, 신규) + `_2_{mode}.jpg`(변환 결과, raw 모드는 표시용으로만 정규화)를 S1과 동일한 네이밍으로 저장. 실제 모델에 들어가는 `img_np`/반환 이미지는 변경 없음.

수정 후 재확인 결과: 4개 모두 재실행하여 시각적으로 확인 완료 (`logs/input_v0.1/s1.depth_raw_s2.fpv_v2/`, `logs/input_v0.1/s1.fpv_s2.depth_raw_v2/`, `logs/input_v0.1/s1.depth_colormap_s2.fpv/`, `logs/input_v0.1/s1.bev_rgb_concat_s2.fpv/`). bev-S1 나머지 5개 occ concat 조합은 코드 리뷰로 충분하다는 확인을 받아 재실행하지 않음 — 동일 가드 로직이라 동일하게 동작함.

5. **S2 `bev` (`s1.fpv_s2.bev_rgb`)에 디버그 이미지가 아예 없던 것 수정.** 기존 `BEVImageProvider.get_s2_extra`(`visual_input_provider.py`, S1의 `get_s1_input`이 이미 쓰는 `debug_dir` 컨벤션과 동일)에 디버그 저장을 추가: `s2_step_XXXXXX_1_fpv.jpg`(원본 lookdown 프레임) + `_2_bev_{depth_src}.jpg`(BEV 투영 결과) — `get_s1_input`의 `_1_fpv.jpg`/`_N_bev_{src}.jpg` 네이밍과 동일하게 맞춤. `UnifiedImageProvider`(view='bev' 위임)와 기존 legacy BEV 파이프라인 양쪽에 공통 적용됨(공유 베이스 클래스 수정, 순수 추가라 기존 동작엔 영향 없음 — `debug_dir=None`이면 아무 것도 안 함).

재실행 확인 (`logs/input_v0.1/s1.fpv_s2.bev_rgb_v2/`): loss 정상(1.00→1.19), 14 step마다 `_1_fpv.jpg`+`_2_bev_gt.jpg` 쌍 생성. 육안 확인: 원본 프레임(복도, 가운데 기둥/벽)과 BEV 결과(카메라 FOV 콘 모양의 V자 투영)가 기하학적으로 일치함 — 정상 동작 확인.

6. **S2 bev 디버그에 GT topdown 참조 이미지 추가** (scale/방향 디버깅용, 기존 S1 쪽 `step_XXXXXX_bNN_5_gt_topdown.jpg`와 동일한 소스). `save_train_tdmap_debug`(`visual_input_provider.py`)에 optional `prefix` 파라미터 추가(기존 S1 호출부는 인자 안 주면 기존 그대로 동작), `internvla_n1_lerobot_dataset.py`의 `traj_tdmaps` 계산 블록에서 `s2_view=='bev'`일 때 `sampled_ids[0]`(== `start_frame_id`, S2 주입에 쓴 것과 동일 프레임)의 topdown을 S2와 같은 step 번호로 저장하도록 연결. 재실행 확인(`logs/input_v0.1/s1.fpv_s2.bev_rgb_v3/`): `s2_step_XXXXXX_5_gt_topdown.jpg` + 보너스로 `_w5_gt_topdown_world.jpg`(forward 화살표 포함, world-frame 정렬)까지 생성됨. 육안 확인: BEV의 V자 투영 방향과 topdown의 forward 화살표가 일치.

7. **S2 occupancy map (depth 기반) 신규 지원.** 기존엔 `BEVProcessor.for_s2`가 무조건 `bev_chw`(컬러 BEV)만 호출해서 S2는 occ 계열을 전혀 만들 수 없었음(S1의 `for_s1`/`_bev_batch`는 이미 `occ`/`occ.binary`/`occ.prob`/`occ.dist`/`occ.dist.sep` 지원). `for_s2`에 optional `image_type` 파라미터 추가해 `_bev_batch`로 라우팅(raw depth는 metric으로 스케일 보정 후 전달), `BEVImageProvider.get_s2_extra`에도 optional `image_type` 파라미터 추가(기본값 None → 기존 `self.image_type` 그대로,하위호환). `UnifiedImageProvider.get_s2_extra`에서 `s2_image_mode`로부터 **S2 전용** occ 타입을 계산해서 전달(기존엔 S1의 `image_type`을 그대로 재사용하던 버그성 커플링이 있었음 — S2가 독립적으로 occ 모드 선택 가능하도록 수정). 신규 config `image/s1.fpv_s2.bev_occ.py`로 검증: `s2_step_XXXXXX_2_bev_occ_gt.jpg` 생성, loss 정상, 육안 확인 — occ map이 컬러 BEV와 동일한 FOV 콘 형태로 올바르게 투영됨.

**최종 상태 (2차 업데이트)**: 20개 조합 + S2 bev topdown 디버그 + S2 occ map 신규 지원까지 전부 확인 완료.
**최종 상태**: 20개 조합 중 **14개 실사용 가능**(PASS, 시각화 정상 — S2 bev 포함), **6개(bev-view+concat) 는 의도적으로 미지원**(깔끔한 `NotImplementedError`로 문서화, 크래시 아님).

## 조합 enumeration (코드 근거: `internnav/model/utils/unified_image_provider.py`)

축은 S1/S2 각각 `view(fpv|bev) x type(rgb|depth|panorama) x mode x combine(none|replace|concat)`.

- `panorama`: 항상 `NotImplementedError` 스텁 → 테스트 대상에서 제외.
- `combine='none'`: view/type/mode 무관하게 완전 no-op (`S1VisualInput()` / `[]` 즉시 반환) → S1/S2 각각 대표 1개(baseline)로만 취급.
- S2 `view='bev'`: 내부적으로 항상 `BEVProcessor.for_s2()`(컬러 BEV만) 호출 — `s2_image_type`/`s2_image_mode` 값은 실제로 무시됨. 대표 1개 조합으로 축소.
- S2 `combine='replace'` vs `'concat'`: 현재 구현상 동작 차이 없음(`internvla_n1_lerobot_dataset.py`에서 항상 "append" 방식) — `replace` 하나만 테스트.
- S1 `view='fpv'`: `type='rgb'`는 `combine!=none`이면 `ValueError`(구현상 depth만 지원) → 제외.

**유효 조합: S1 16개(no-op 포함) x S2 5개(no-op 포함)** 중, 이번 라운드는 "S1 축만 변경(S2=no-op)" 15개 + "S2 축만 변경(S1=no-op)" 4개 + baseline 1개 = **총 20개**를 config로 만들어 실행. (S1×S2 풀 크로스 80개는 두 축이 서로 독립적으로 동작함을 코드로 확인했으므로 생략.)

## 실행 커맨드

```
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image/{config}.py --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/{name}
```
- `--max-steps 3`: 3-step 스모크 테스트로 제한 (신규: `Params.max_steps`, 기본 -1=미설정 → 기존 정식 학습 경로 100% 동일).
- `--debug-dir`: 이제 `--debugpy` 없이도 단독으로 동작 (기존 버그 수정, 아래 참고).

## 이번에 고친 버그 3개 (본 실행 전 발견)

1. **`internvla_n1_trainer.py::train()`가 `max_steps`/`save_strategy`와 무관하게 항상 `trainer.save_state()` + 전체 모델 저장(`safe_save_model_for_hf_trainer`)을 호출** → 3-step 스모크 테스트인데 매번 16GB 체크포인트가 `checkpoints/image/*`에 저장됨. `if training_args.max_steps <= 0:` guard 추가(기본값 -1이라 기존 학습 미영향).
2. **`train_argv`가 smoke 여부와 무관하게 항상 `--report_to wandb`** → 매번 wandb run 생성. smoke(`max_steps>0`)일 때 `--report_to none`.
3. **`runner.py`: `--debug-dir`이 `--debugpy`와 같이 줄 때만 `params.debug_dir`에 반영되는 버그** (`--debugpy` 없이 `--debug-dir`만 주면 조용히 무시됨 → "no such file or directory") → 이제 `--debug-dir` 단독으로도 반영. 또한 smoke 실행(`max_steps>0`)은 `output_dir`를 `checkpoints_root` 대신 `debug_dir`로 라우팅해 `checkpoints/`에 아무것도 안 남도록 함.

이 수정 전 실행분(`checkpoints/image/*`, ~32GB)은 사용자 확인 후 삭제 완료. 이번 재실행 후 `checkpoints/image/`에는 (수정 적용 전 검증용으로 실행한) 180K짜리 폴더 1개만 남아있음 — 원하시면 `rm -rf checkpoints/image`로 정리 가능.

## 결과 요약 (20개 중 13 PASS / 7 FAIL)

`runner.py`는 학습 서브프로세스 실패를 자기 exit code에 반영하지 않으므로 exit code가 아니라 각 로그의 `Traceback`/`ChildFailedError` 유무로 직접 확인함.

### PASS — S1 축 (S2 = fpv no-op)

| config | S1 view/type/mode/combine | 결과 | 시각화 |
|---|---|---|---|
| `base_s1.fpv_s2.fpv` | fpv/rgb/raw/none | baseline no-op, loss 1.02→1.24 | 이미지 없음 (no-op이라 정상) |
| `s1.bev_rgb_replace_s2.fpv` | bev/rgb/raw/**replace** | loss 정상 | `logs/input_v0.1/s1.bev_rgb_replace_s2.fpv/step_*_{1_fpv,2_gt_depth,3_bev_gt,6_ros2_occ_gt,7_ros2_pkg_occ_gt}.jpg` (120장) |
| `s1.bev_occ_replace_s2.fpv` | bev/depth(occ)/raw/**replace** | loss 정상 | 위와 동일 + `3_train_occ_gt.jpg` (144장) |
| `s1.bev_occ.binary_replace_s2.fpv` | bev/depth/binary/replace | loss 정상 | 144장 |
| `s1.bev_occ.prob_replace_s2.fpv` | bev/depth/prob/replace | loss 정상 | 144장 |
| `s1.bev_occ.dist_replace_s2.fpv` | bev/depth/dist/replace | loss 정상 | 120장 (`3_train_occ` 없음 — dist/dist_sep은 시각화 분기 미지원, 기존 코드 그대로) |
| `s1.bev_occ.dist.sep_replace_s2.fpv` | bev/depth/dist_sep/replace | loss 정상 | 120장 |
| `s1.depth_raw_s2.fpv` | fpv/depth/**raw**/replace | loss 정상 | `unified_step_*_{1_fpv,2_depth_raw,3_depth_adapted}.jpg` (72장) |
| `s1.depth_normalized_s2.fpv` | fpv/depth/**normalized**/replace | loss 정상 | 72장 |

### PASS — S2 축 (S1 = fpv no-op)

| config | S2 view/type/mode/combine | 결과 | 시각화 |
|---|---|---|---|
| `s1.fpv_s2.bev_rgb` | bev/rgb/raw/replace | loss 정상 | 이미지 없음 — **기존 `BEVImageProvider.get_s2_extra`에 원래 디버그 저장 기능이 없음** (이전 세션에서도 확인된 기존 제약, 버그 아님) |
| `s1.fpv_s2.depth_raw` | fpv/depth/raw/replace | loss 정상 | `logs/input_v0.1/s1.fpv_s2.depth_raw/s2_unified_step_*.jpg` (14장, 흰색에 가까움 — raw는 정규화 안 함, 의도된 동작) |
| `s1.fpv_s2.depth_normalized` | fpv/depth/normalized/replace | loss 정상 | 14장 (그레이스케일 정규화) |
| `s1.fpv_s2.depth_colormap` | fpv/depth/colormap/replace | loss 정상 | 14장 (JET 컬러맵, 육안 확인 가장 쉬움) |

### FAIL (7개)

| config | 조합 | 에러 |
|---|---|---|
| `s1.bev_rgb_concat_s2.fpv` | bev/rgb/raw/**concat** | `RuntimeError: Sizes of tensors must match ... Expected size 48 but got size 24` at `internvla_n1.py:255 torch.cat([memory_tokens, traj_hidden_states])` |
| `s1.bev_occ_concat_s2.fpv` | bev/depth(occ)/raw/concat | 동일 |
| `s1.bev_occ.binary_concat_s2.fpv` | bev/depth/binary/concat | 동일 |
| `s1.bev_occ.prob_concat_s2.fpv` | bev/depth/prob/concat | 동일 |
| `s1.bev_occ.dist_concat_s2.fpv` | bev/depth/dist/concat | 동일 |
| `s1.bev_occ.dist.sep_concat_s2.fpv` | bev/depth/dist_sep/concat | 동일 |
| `s1.depth_colormap_s2.fpv` | fpv/depth/**colormap**/replace | `TypeError: Got unsupported ScalarType BFloat16` at `unified_image_provider.py:84 _depth_to_colormap` (`.cpu().numpy()`가 bf16 텐서를 못 받음) |

**원인 분석 (수정은 안 함 — 보고만):**
- **bev + concat 6개 전부 동일 원인**: `s1_combine_mode='concat'`은 T를 2배로 늘리는데(`fpv_bev` 모드), `nextdit_async` S1 head가 `traj_hidden_states`를 `traj_poses`의 T(원본)로 repeat하는 반면 이미지 쪽 T는 doubled라서 `torch.cat([memory_tokens, traj_hidden_states])`에서 크기가 안 맞음. 개발 계획 문서(`260707_s1s2_bev_depth_panorama.md` 리스크 #2)에 "concat은 fpv+depth만 미구현이고 view=bev의 기존 `fpv_bev` 경로는 이미 검증됨"이라고 적혀 있었지만, 실제로 `scripts/train_eval/qwenvl_train/bev/*.py` 중 `bev_s1_mode='fpv_bev'`를 쓰는 config는 하나도 없었음(전부 `'bev'`) — 즉 **한 번도 실행된 적 없는 조합**이었고, 이번에 처음 돌려보고 깨진 걸 확인함.
- **depth colormap 1개**: `_depth_to_colormap`이 bf16 텐서를 `.numpy()`로 바로 변환하려다 실패. `raw`/`normalized` 모드는 numpy 변환 없이 텐서만 다루기 때문에 안 걸리고, `colormap`만 cv2 경유라서 걸림.

## 시각화 결과 경로

```
logs/input_v0.1/
  base_s1.fpv_s2.fpv/                     (이미지 없음, no-op 확인용)
  s1.bev_rgb_replace_s2.fpv/               step_*_{1_fpv,2_gt_depth,3_bev_gt,6_ros2_occ_gt,7_ros2_pkg_occ_gt}.jpg
  s1.bev_occ_replace_s2.fpv/               + step_*_3_train_occ_gt.jpg
  s1.bev_occ.binary_replace_s2.fpv/        동일 패턴
  s1.bev_occ.prob_replace_s2.fpv/          동일 패턴
  s1.bev_occ.dist_replace_s2.fpv/          3_train_occ 없음
  s1.bev_occ.dist.sep_replace_s2.fpv/      3_train_occ 없음
  s1.depth_raw_s2.fpv/                     unified_step_*_{1_fpv,2_depth_raw,3_depth_adapted}.jpg
  s1.depth_normalized_s2.fpv/              동일 패턴 (정규화됨)
  s1.fpv_s2.bev_rgb/                       이미지 없음 (기존 제약)
  s1.fpv_s2.depth_raw/                     s2_unified_step_*.jpg (흰색조)
  s1.fpv_s2.depth_normalized/              s2_unified_step_*.jpg (그레이스케일)
  s1.fpv_s2.depth_colormap/                s2_unified_step_*.jpg (JET 컬러, 가장 보기 좋음)
```

## 변경 파일

- 수정: `internnav/trainer/internvla_n1_trainer.py` (max_steps>0 시 저장 skip guard)
- 수정: `scripts/train_eval/qwenvl_train/default_config.py` (`Params.max_steps` 필드, smoke 시 `report_to=none`/`eval_strategy=no`/`save_strategy=no`/`load_best_model_at_end=False`)
- 수정: `scripts/train_eval/qwenvl_train/runner.py` (`--max-steps` CLI, `--debug-dir` 단독 동작 버그 수정, smoke 시 output_dir을 debug_dir로 라우팅, smoke/debugpy 시 wandb.login 스킵)
- 신규: `scripts/train_eval/qwenvl_train/image/*.py` 17개 (기존 3개 + 신규 17개 = 총 20개 조합 config)

## 남은 것 (보고만, 미수정 — 사용자 지시대로 에러만 리포트)

- bev-view `concat` 조합 6개: T-doubling ↔ traj_poses 정렬 버그로 사용 불가.
- fpv+depth `colormap` 조합 1개: bf16→numpy 캐스팅 버그로 사용 불가. (**이후 세션에서 수정 완료** — 아래 참고)

## Update 2 (2026-07-08): 학습 검증 후속 수정 + Habitat eval 검증 (진행 중)

### 이전 세션에서 학습 쪽 마무리한 것
- `s1.depth_colormap_s2.fpv` bf16 버그 수정 → PASS 전환.
- bev-view `concat` 6개: 실제 재설계 대신 `fpv+depth+concat`과 동일한 `NotImplementedError` 가드로 문서화(사용자 확인).
- S1 `depth_raw` 디버그 이미지 washed-out 문제, S2 디버그에 원본 프레임 없던 문제 수정.
- S2 bev 디버그에 이미지가 아예 없던 문제 수정 (`visual_input_provider.py::get_s2_extra`에 `_1_fpv.jpg`+`_2_bev_{src}.jpg` 저장 추가) + gt_topdown 참조 이미지 추가(`save_train_tdmap_debug`에 `prefix` 파라미터 추가해 S2에서도 재사용).
- **S2 occupancy map(depth 기반) 신규 지원**: `BEVProcessor.for_s2`에 `image_type` 파라미터 추가(기존엔 컬러 BEV만 가능했음), S2 전용 occ 모드를 S1과 독립적으로 계산하도록 수정. 신규 config `image/s1.fpv_s2.bev_occ.py`.
- 최종: 20개 조합 중 14개 PASS + S2 bev/occ 신규 기능까지 전부 시각적으로 확인 완료(학습 단계에서만).

### Habitat eval(rollout) 검증 — 완료 (2026-07-08)

GPU 복구 후 10개 조합 전부 2 episode씩 실행. **10/10 PASS** — Traceback 없음.

상세 결과: [260708_habitat_eval_unified_provider_result.md](260708_habitat_eval_unified_provider_result.md)

**다음 단계**: Isaac Sim h1 환경 검증 (별도 진행 예정).
