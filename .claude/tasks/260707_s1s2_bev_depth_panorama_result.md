# 결과 보고: S1/S2 통합 Image Provider (view x type x mode x combine)

계획 파일: `/root/.claude/plans/tasks-260707-s1s2-bev-depth-panorama-md-ticklish-lollipop.md`
브랜치: `feature/input_v0.1` (`feature/bev_v0.1`에서 분기)

## 이전과 달라진 점

- 기존 `bev`(boolean)/`bev_s1_mode`/`bev_image_type` 파이프라인은 **완전히 그대로 유지** (필드명/동작 변경 없음, 기존 `scripts/train_eval/qwenvl_train/bev/*.py` 전부 그대로 동작).
- 새 축 4개를 S1/S2 각각 독립적으로 추가: `s{1,2}_image_view`(fpv|bev), `s{1,2}_image_type`(rgb|depth|panorama), `s{1,2}_image_mode`(값 처리 방식), `s{1,2}_combine_mode`(none|replace|concat). 마스터 토글은 `image_provider: bool`.
- **S1 depth 입력**(신규): `view=fpv, type=depth`일 때 GT/추정 depth를 raw(1ch, meter 그대로)/normalized(1ch, [0,1])/colormap(3ch, JET) 이미지로 변환해 S1이 보게 함. 1ch → 3ch 변환은 `ImageInputAdapter`(기본 `repeat`, 무파라미터).
- **S2 depth/BEV 입력**(신규, 학습 시점까지 포함): 기존에는 BEV가 eval(rollout) 시점에만 S2에 주입 가능했으나, `NavPixelGoalDataset`이 이미 가변 개수 멀티이미지 프롬프트 구조(history/look-down 이미지)를 갖고 있다는 걸 확인하고 학습 데이터셋 레벨에도 동일한 패턴으로 depth/BEV 이미지를 주입하도록 확장함.
- `view=bev`는 기존 `BEVImageProvider` 로직에 **100% 위임**(상속으로 재사용, 로직 중복 없음). `type=panorama`는 스텁으로 `NotImplementedError`.
- 이전 세션에서 만들었던 좁은 범위의 파일 3개(`image_provider.py` 등)는 삭제하고 통합 설계로 대체.

## 새/수정 파일

- 신규: `internnav/model/utils/unified_image_provider.py` (`UnifiedImageProvider`, `ImageInputAdapter`, `create_unified_provider`)
- 신규: `internnav/model/basemodel/internvla_n1/internvla_n1_unified_provider.py` (S1 forward 연결)
- 신규: `internnav/trainer/internvla_n1_unified_provider_trainer.py` (모델 클래스 monkey-patch, 기존 `internvla_n1_bev_provider_trainer.py`와 동일 패턴)
- 신규: `scripts/debugging/unit_test/test_unified_image_provider.py` (9개 테스트)
- 신규: `scripts/train_eval/qwenvl_train/image/{base_s1.fpv_s2.fpv, s1.depth_raw_s2.fpv, s1.fpv_s2.depth_colormap}.py`
- 수정(추가만, guard-clause): `internnav/model/utils/visual_input_provider.py`(factory에 elif 1개), `internnav/trainer/internvla_n1_argument.py`(S2용 optional 필드 5개), `internnav/dataset/internvla_n1_lerobot_dataset.py`(S2 이미지 주입 지점 1곳), `internnav/habitat_extensions/vln/habitat_vln_evaluator_bev.py`(라벨 하드코딩 제거), `scripts/train_eval/qwenvl_train/default_config.py`(Params 필드 + trainer/train_argv/build_habitat_eval_cfg 분기)

## 실행한 명령어와 argument 의미

```bash
# 유닛 테스트 (32개 전부 통과: 신규 9개 + 기존 BEV 23개)
python -m pytest scripts/debugging/unit_test/test_unified_image_provider.py scripts/debugging/unit_test/test_visual_input_provider.py -q
```

```bash
# S1 depth(raw) 학습 smoke test — 2 step 실제 GPU 학습
torchrun --nproc_per_node=1 internnav/trainer/internvla_n1_unified_provider_trainer.py \
  --s1_image_view fpv --s1_image_type depth --s1_image_mode raw --s1_combine_mode replace --bev_depth_source gt \
  --deepspeed scripts/train/qwenvl_train/zero2.json --model_name_or_path checkpoints/InternVLA-N1-System2 \
  --vln_dataset_use "r2r_125cm_0_30%30,r2r_60cm_15_15%30" --data_root data/InternData-N1-v0.5-mini/vln_ce \
  --max_steps 2 --per_device_train_batch_size 1 --debug_dir output/smoke_debug_depth_raw ...
```
- `--s1_image_view fpv --s1_image_type depth --s1_image_mode raw --s1_combine_mode replace`: S1이 원본 FPV 대신 raw depth(1채널, meter 값 그대로) 이미지를 보도록 교체.
- `--bev_depth_source gt`: depth는 dataset의 GT depth 사용 (dav2/udv2 추정도 가능하나 이번엔 GT로 검증).
- `--debug_dir`: 매 step마다 `unified_step_XXXXXX_bNN_{1_fpv,2_depth_raw,3_depth_adapted}.jpg` 저장.

```bash
# S2 depth(colormap) 학습 smoke test — dataset 레벨 이미지 주입 검증
torchrun --nproc_per_node=1 internnav/trainer/internvla_n1_unified_provider_trainer.py \
  --s2_image_view fpv --s2_image_type depth --s2_image_mode colormap --s2_combine_mode replace \
  --deepspeed scripts/train/qwenvl_train/zero2.json --model_name_or_path checkpoints/InternVLA-N1-System2 \
  --vln_dataset_use "r2r_125cm_0_30%30,r2r_60cm_15_15%30" --data_root data/InternData-N1-v0.5-mini/vln_ce \
  --max_steps 1 --per_device_train_batch_size 1 --debug_dir output/smoke_debug_s2_depth_v2 ...
```
- `--s2_combine_mode replace`: S2 프롬프트의 look-down FPV 이미지 자리를 depth colormap 이미지로 교체.
- 내부적으로 `NavPixelGoalDataset.__getitem__`이 `<image>` 태그 하나를 추가하고 `process_image_unified()`로 동일하게 토크나이즈 — RoPE/collator 수정 없이 그대로 동작 확인.

```bash
# 회귀 확인: 기존 BEV 경로(코드 미수정) 그대로 동작하는지
torchrun --nproc_per_node=1 internnav/trainer/internvla_n1_bev_provider_trainer.py \
  --bev_s1_mode bev --bev_image_type rgb --bev_depth_source gt ...
```

## 시각화 및 디버깅 결과

- S1 depth(raw) triptych (`output/smoke_debug_depth_raw/unified_step_000000_b00_*.jpg`): FPV는 정상적인 실내 씬. depth_raw는 거의 흰색으로 보이는데, 이는 **의도된 동작**임 — "raw"는 [0,1] 정규화를 하지 않고 meter 값을 그대로 저장하므로 1m 이상 거리는 전부 흰색으로 clamp됨. 육안 확인용 이미지가 필요하면 `s1_image_mode=colormap`을 쓰면 됨(정규화 후 JET colormap).
- S2 depth(colormap) 이미지 (`output/smoke_debug_s2_depth_v2/s2_unified_step_000000.jpg`): 복도 씬이 빨강(가까움)~파랑(멀음) JET colormap으로 정확히 시각화됨 — depth 값이 실제로 읽히고 있고 변환이 올바르다는 것을 확인.

## 검증 결과 요약

| 항목 | 결과 |
|---|---|
| 유닛 테스트 32개 | 전부 통과 |
| S1 fpv+depth(raw, replace) 실제 GPU 학습 2 step | loss 1.32→1.24, gradient 정상 흐름 |
| S2 fpv+depth(colormap, replace) 실제 GPU 학습 | loss 정상 범위, 디버그 이미지로 시각 확인 |
| S2 bev(rgb, replace) 학습 | **최초 실행 시 크래시 발견 후 수정** — 아래 "발견된 버그" 참고, 수정 후 정상 동작 확인 |
| `image_provider=True` + 모든 축 기본값 (no-op) | 정상 동작 (에러 없음) |
| 기존 BEV 경로 (코드 미수정) 회귀 확인 | 정상 동작, loss 1.38→1.24 (기존과 동일한 패턴) |
| 3개 신규 experiment config 로드 | PARAMS/argv 모두 기대한 값으로 정확히 resolve됨 |

## 발견된 버그 (사용자 질문으로 실제 실행해보고 발견 → 수정)

최초 구현에서는 `s2_image_view='bev'`를 실제로 돌려보지 않고 코드만 보고 "동작할 것"이라 판단했는데, 사용자가 "S2에서 depth/bev 정말 되냐"고 물어봐서 직접 실행해보니 **크래시**했습니다.

- 원인: `NavPixelGoalDataset.__getitem__`에서 `depth_image`는 S1용으로 이미 224x224로 리사이즈된 상태인데, S2 BEV 주입 코드는 원본 해상도(640x480)의 `lookdown_image`를 그대로 넘겨서 `BEVProcessor.for_s2()` 내부의 rgb/depth reshape가 깨짐 (`RuntimeError: shape '[1, 50176, 3]' is invalid for input of size 921600`).
- 수정: `internnav/dataset/internvla_n1_lerobot_dataset.py`에서 S2 이미지 생성 직전에 `lookdown_image`를 `depth_image`와 같은 해상도로 리사이즈하도록 1줄 추가.
- 수정 후 재실행하여 정상 동작(loss 1.17) 확인함. S2 depth(GT) 경로는 이 버그의 영향을 받지 않았음 (rgb 인자를 쓰지 않는 코드 경로라 우연히 문제 없었음).
- 참고: S2 BEV 경로는 기존 BEV 코드(`BEVImageProvider.get_s2_extra`)에 디버그 이미지 저장 기능이 원래 없어서, 이번에도 시각 확인용 디버그 이미지는 남지 않음 (기존 동작과 동일).

## 이번 pass에서 미룬 것 (계획서에 명시)

- Panorama 실제 구현 (데이터 없음 — 스텁만, 호출 시 `NotImplementedError`)
- `combine='concat'`은 `view=fpv,type=depth`에서 미지원 (T-doubling이 `traj_poses` 정렬을 깨뜨림 — 기존 `fpv_bev` 경로도 동일한 잠재적 이슈가 있음을 확인했으나 손대지 않음)
- Isaac Sim / real-world eval 쪽 새 축 연결 (Habitat만 이번 범위)
- S2 학습 이미지는 GT depth만 지원 (dav2/udv2 추정은 dataloader worker에서 모델 추론이 필요해 범위 밖 — 요청 시 명확한 에러)
- S1 eval(rollout) 시점 depth 지원은 아직 없음(학습만) — Habitat eval은 `visual_provider='unified_image'`로 factory까지는 연결해뒀으나 실제 rollout 검증은 안 함
- S2 학습에서 `bev_depth_source='dav2'/'udv2'`(추정 depth)는 아직 미지원 (dataset construction 시점에 명확한 에러). `dataloader_num_workers=0`이면 기술적으로 어렵지 않음 — 필요하면 후속 작업으로 가능

## 남은 스크래치 아티팩트

`output/smoke_test_*`, `output/smoke_debug_*`는 이번 검증용으로 만든 산출물이며 git에 포함되지 않음(untracked). 필요 없으면 삭제해도 무방.
