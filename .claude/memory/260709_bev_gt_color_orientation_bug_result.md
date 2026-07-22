# s1.bev.rgb.concat_s2.fpv (Isaac/H1) — bev_gt 컬러/방향 불일치 조사 결과

## 요청

`scripts/train_eval/qwenvl_train/image_base/`의 4개 config(`s1.bev.rgb.concat_s2.fpv.py`, `s1.bev.rgb.replace_s2.fpv.py`, `s1.fpv_s2.bev.ld.rgb.concat.py`, `s1.fpv_s2.bev.rgb.concat.py`) 시각화 디버깅 + `s1.bev.rgb.concat_s2.fpv`에서 `bev_gt`가 실제 RGB BEV가 아니고 `gt_topdown`과 방향이 안 맞는 문제 조사.

**진행 상황**: 보고된 버그(`s1.bev.rgb.concat_s2.fpv`, Isaac/H1 경로)의 원인을 상당 부분 좁혔으나 최종 확정은 못했음. 나머지 3개 config 및 Habitat 경로 시각화 디버깅은 아직 미착수 — 아래 결과를 공유하고 다음 방향을 확인한 뒤 이어감.

## 확인된 사실

`logs/input_v0.1/image_base/s1.bev.rgb.concat_s2.fpv/eval_isaac/`(2026-07-09 13:11 실행, `--model_path checkpoints/image_base/s1.bev.rgb.concat_s2.fpv/checkpoint-1` — 이 config로 실제 학습된 체크포인트, config.json도 `s1_image_type=rgb, s1_view=bev, s1_combine_mode=concat` 일치 확인)의 `step_000000_b00_3_bev_gt.jpg`를 열어보면 실제로 컬러가 아니라 회색조 + 흰색 해칭(hachure) 패턴 — occupancy 시각화와 흡사함.

정량 확인 (마스킹된 non-black 픽셀만, per-pixel max-min 채널 스프레드):
| | isaac bev_gt (164 스텝 샘플) | habitat bev_gt (`_check` 폴더, 검증된 정상 케이스) |
|---|---|---|
| 스프레드 평균 | 4~7 / 255 | 7~32 / 255 |

→ isaac 쪽만 전체 스텝에서 일관되게 거의 무채색. 환경 자체가 원래 회색(하지만 fpv.jpg는 나무색 캐비닛/액자 등 색이 뚜렷함)이라 그런 게 아니라 **BEV에 색이 거의 실리지 않는 시스템적 문제**로 판단.

## 배제한 가설 (코드/실측으로 검증 완료)

1. **`unified_image_provider.py`/`depth_rgb_to_bev_torch.py`의 RGB 투영 로직 자체 버그** — 배제. `create_unified_provider`로 실제 isaac 파라미터(fx=585, cam_height=1.25, pitch=30, ref 640x480) 구성 후 실제 fpv.jpg + 합성 depth로 직접 호출 → 정상적으로 다양한 색 출력 확인(`bev[0,0,::40,::40]` 샘플에 0.78/0.46 등 채널별로 다른 값 다수). `image_type` dispatch도 `s1_type='rgb'`일 때 `_bev_batch`가 정확히 `bev_chw`(occ 아님) 분기로 감. 회색조가 "정상적으로 튀어나온 tilt된 cone 모양" 자체는 실제로 정상 — Habitat `_check` 폴더의 검증된 정상 케이스도 cone이 대각선으로 tilt돼 있음 (레거시 `logs/bev_debug/bev_debug_000000.jpg`도 동일 패턴), 이 tilt은 버그가 아님.
2. **depth 단위/스케일(`internvla_n1_agent.py`의 `depth * 10.0`) 버그** — 배제. `_get_s1_bev_replace`에 임시 진단 print를 넣고 실제 Isaac eval 1회 실행(`--machine h1 --model-path checkpoints/image_base/s1.bev.rgb.concat_s2.fpv/checkpoint-1 --max-steps 2`) → depth 값이 `min=0.28~0.53, max=2.4~5.0(threshold clamp), mean=1.0~2.0` 로 실내 장면에 맞는 정상적인 미터 단위 range. 만약 스케일이 10배 잘못됐다면 거의 모든 프레임이 5.0(threshold)에 clamp돼야 하는데 실제로는 절반 정도만 clamp되고 나머지는 2~3m대의 다양한 값 — 정상. (진단 print는 확인 후 원복함, `git diff` 깨끗함)

## 유력하게 남은 가설 (미확정 — 추가 조사 필요)

**`obs['rgb']`와 `obs['depth']`가 같은 프레임을 가리키지 않는 것으로 보임.**

같은 실행에서 `step_000000_b00_1_fpv.jpg`(RGB)와 `step_000000_b00_2_gt_depth.jpg`(depth, 같은 t=0 인덱스에서 저장 — `_save_s1_debug`의 `_save_frame`이 `rgb[b,t]`와 `depth[b,t,...,0]`을 같은 t로 저장하므로 코드상 페어링은 맞음)를 나란히 보면:
- fpv.jpg: 복도, 오른쪽에 흰 캐비닛, 벽에 액자 여러 개, 나무색 서랍
- gt_depth.jpg: 전혀 다른 구도 — 왼쪽에 밝은 문/창문, 가운데~오른쪽에 어두운 가구 실루엣

step 2(`step_000002_*`)에서도 마찬가지로 fpv(창문 있는 방, 테이블)와 depth(복도+박스 형태)가 서로 다른 장면. **2번의 별도 Isaac 실행(13:11, 이번 13:57) 모두 이 패턴이 재현됨** — 우연이 아니라 구조적 문제로 보임.

RGB·depth가 이렇게 어긋난 상태로 BEV의 "depth로 3D 위치 결정 + 그 위치에 RGB 색 샘플링" 투영을 하면, 색이 실제 지오메트리와 무관하게 섞여 들어가 회색조로 뭉개지고, `gt_topdown`(별도의 topdown 카메라 GT)과도 구조가 안 맞아 보이는 게 자연스럽게 설명됨 — 즉 보고된 두 증상(컬러 이상 + 방향 불일치) 모두 이 한 가지 원인으로 설명 가능.

**추적한 소스 코드 위치**: `internnav/env/utils/internutopia_extension/tasks/vln_eval_task.py`의 `get_rgb_depth()` (L110-137). `camera.get_data()` 한 번 호출한 `cur_obs`에서 `rgba`와 `depth`를 같이 꺼내므로 코드상으로는 동일 카메라·동일 시점처럼 보이지만, `env_id==0`일 때만 실행되는 `rep.orchestrator.step(rt_subframes=2, delta_time=0.0, ...)`(L116-117, replicator 렌더 강제 진행 — flash/teleport 이동 직후 렌더 버퍼를 따라잡기 위한 것으로 추정)가 RGB pass와 depth pass를 동일하게 수렴시키지 못했을 가능성을 의심 중 (RTX 렌더러에서 AOV별 수렴 속도가 다를 수 있음, `robot_flash=True`로 텔레포트 이동을 쓰는 이 태스크 설정과 결합하면 순간이동 직후 depth와 rgba가 서로 다른 서브프레임 상태를 반영할 수 있음). **이 부분은 가설이며 확정하지 못함** — 실제로 `rt_subframes`를 늘리거나 rgba/depth를 같은 시점에 강제 동기화해보는 실험, 혹은 `camera.get_data()`가 내부적으로 프레임을 어떻게 캐싱하는지 InternUtopia 쪽 코드를 더 봐야 함.

## 다음 단계 제안

1. `get_rgb_depth()`의 `rt_subframes` 값을 늘리거나, rgba/depth 획득 사이에 추가 렌더 스텝을 넣어 실제로 두 장면이 일치하는지 재현 실험.
2. 또는 `camera.get_data()`가 rgba/depth를 GPU에서 언제 스냅샷하는지(InternUtopia/omni.replicator 쪽) 확인.
3. 확정되면 수정 + 재검증(동일 `--max-steps 2` 커맨드로 fpv.jpg와 gt_depth.jpg 장면이 일치하는지 육안 확인) 필요.
4. 원래 요청이었던 4개 config train/eval 시각화 디버깅(Habitat 경로 포함)은 이 조사 때문에 아직 미착수 — 이어서 진행할지, 이 버그 확정/수정을 먼저 할지 방향 확인 필요.

## 실행 커맨드 (재현용)

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/image_base/s1.bev.rgb.concat_s2.fpv.py --machine h1 --no-train --model-path checkpoints/image_base/s1.bev.rgb.concat_s2.fpv/checkpoint-1 --max-steps 2 --debug-dir <dir>
```
- `--machine h1`: Isaac Sim 경로 (단일 프로세스, `python.sh` 필요)
- `--no-train`: eval만 실행
- `--model-path`: 이 config로 학습된 체크포인트 (아키텍처 일치 확인됨)
- `--max-steps 2`: eval 2 episode로 제한 (episode당 최대 `max_step=1000` 스텝까지 갈 수 있어 실행 시간이 김 — 이번 조사에서 1 episode당 몇 분 소요)
- `--debug-dir`: `step_*_{1_fpv,2_gt_depth,3_bev_gt,5_gt_cam,w5_gt_topdown_world}.jpg` 등이 저장됨
