# instruction_labeling P1 — habitat 렌더러 검증 결과 (2026-10-13, KST)

코드: `scripts/dataset_converters/instruction_labeling/render/` (README 참고). 설정: `configs/render.yaml`.
계획: `.claude/tasks/instruction_labeling.md`. 로그: `logs/instruction_labeling/{p1r,p1a}/`.

## 용어
| 용어 | 뜻 |
|---|---|
| 릴리스 | InternData-N1-v0.5-mini `vln_ce/traj_data/r2r` (train 만 이미지 있음) |
| P1-R (렌더러 검증) | 릴리스 train 의 **action 을 그대로** 재생해서 렌더 → pose·이미지를 릴리스와 비교. 렌더러 오차만 잰다 |
| P1-A (생성기 검증) | val 에는 action 이 없다 → reference_path 에서 action 을 만드는 생성기가 릴리스 action 을 얼마나 재현하나 |
| greedy_wp | habitat `GreedyGeodesicFollower`(= habitat-lab `ShortestPathFollower`)로 reference_path 정점을 차례로 따라감 |
| MAE | 픽셀값 차 절댓값 평균 (0~255 척도) |
| SSIM | 구조 유사도, 1 이면 동일 |
| JPEG 잡음 바닥 | 우리 렌더 원본 vs 그것을 JPEG 로 저장한 것의 MAE. 압축만으로 생기는 차이 |

## P1-R 렌더러: **통과 (사용자 기준 합의 대기)**
train 3 scene (17DRP5sb8fy, 1LXtFkjw3qL, 29hnd4uzFmX), 429 episode 전부 매칭, 21,720 프레임 (픽셀은 5 프레임마다 + 마지막).

| 항목 | 125cm_0deg | 125cm_30deg |
|---|---|---|
| 위치 오차 p99 / max | 0.07 / 0.08 mm | 같음 |
| 회전 오차 max | 0.0007° | 0.0022° |
| RGB MAE 평균 (worst episode) | 1.48 (2.28) | 1.50 (2.04) |
| JPEG 잡음 바닥 MAE | 2.51 | 2.65 |
| SSIM 평균 / 최저 | 0.978 / 0.973 | 0.977 / 0.973 |
| PSNR 최저 | 38.1 dB | 34.8 dB |
| 평균 밝기 차 (ours−release) | +0.23 | +0.22 |
| depth mm 완전일치 | 99.6 % | 99.3 % |

- **기하는 완전히 같다**: 위치 0.1 mm 미만, depth 가 mm 단위로 99 % 이상 일치 → 카메라 위치·방향·hfov·높이·pitch 동일.
- **RGB 차이는 JPEG 압축 잡음보다 작다** (1.5 < 2.5). 차이 이미지에 구조가 없고 가장자리·텍스처에 흩어진 잡음뿐
  (`logs/instruction_labeling/p1r/viz/*.jpg`). 남는 원인 후보: 릴리스 렌더 GPU/드라이버의 래스터화 차, 인코더 차.
- **md5(바이트) 일치는 불가능**: 릴리스 jpg 의 chroma 양자화 테이블 첫 값이 8 인데 이 머신 인코더(PIL/cv2)는 Q75 에서 9.
  → 저장 시 **릴리스 jpg 의 양자화 테이블을 그대로** 쓴다 (`image.jpeg_qtables_ref`).
- depth 저장 규약: `floor(m*1000)` uint16 (round 로 하면 50 %만 일치 — floor 가 정답).
- 처음 측정에서 회전 오차가 0.44° 로 보였던 것은 릴리스 pose 가 float32 라서 `arccos` 가 불안정했던 탓. rotvec 으로 재면 0.002°.

## P1-A action 생성기: **완전 재현 실패 — greedy_wp 를 근사로 채택 제안**
train 61 scene, 3,558 trajectory. 렌더 없이 sim step 만.

| 후보 | 완전일치 | 편집거리/길이 | 프레임 수 차 \|평균\| | 경로 평균 차 | 끝점 차 p95 |
|---|---|---|---|---|---|
| **greedy_wp_r0.25** | 0.3 % | 0.27 | 9.0 | 0.19 m | **0.23 m** |
| greedy_wp_r0.5 | 0.3 % | 0.24 | 10.5 | 0.18 m | 0.45 m |
| greedy_goal_r0.25 | 0.3 % | 0.29 | 13.1 | 0.32 m | 0.23 m |
| greedy_dense0.5_r0.25 | 0.1 % | 0.55 | 25.9 | 0.23 m | 0.27 m (follower 오류 287) |

- 시험했지만 일치하지 않은 것: VLN-CE 공식 `train_gt.json.gz` action (27/10,684 일치), 방위각 규칙(`vlnce_discretize` 식),
  정점 보간, navmesh agent radius 0.15~0.25 재계산.
- greedy_wp_r0.25 분포: 경로 평균 차 중앙 0.13 m, 프레임 수 차 중앙 −1, |차| ≤ 3 프레임 46 %.
- **릴리스 쪽 이상치 8.3 %**: 릴리스가 reference_path 를 크게 벗어나 돌아간다 (프레임 수가 20 이상 많음, 7 % 는 경로 최대 차 2 m 초과).
  예: `dhjEzFoUFzH` traj 4467 (릴리스 244 프레임 vs reference 86), `Vvot9Ly1tCj` traj 371 — 그림 `logs/instruction_labeling/p1a/topdown/`.
  지시문은 reference_path 를 설명하므로 **릴리스 train 의 이 8 % 는 이미지와 지시문이 어긋날 가능성**이 있다 (별도 확인 필요).
- 결론: 릴리스 action 생성 규칙은 공개되지 않은 자체 플래너로 보인다. val 은 greedy_wp_r0.25 로 만든다.
  이 생성기는 InternNav habitat evaluator 가 쓰는 `ShortestPathFollower(sim, 0.25)` 와 같은 구현이다.
  주장 범위: "val 이미지는 train 과 **같은 카메라·렌더러**, 경로는 reference_path 를 따르는 근사 (평균 0.19 m 차)".

## P1-E val_unseen: 코드 완료, 스모크만 실행 (전체 렌더는 사용자 확인 후)
- `render_split.py` 2 trajectory 스모크 (2azQ1b91cZZ): 6 episode, `verify_split.py` 의 스키마·이미지·사본 동일성·지시문·
  05b V5(로더 재현 로드)/V6(pose 불변식) 전부 통과 (개수 불일치는 일부만 렌더한 탓).
- 출력 예정 위치: `data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r/val_unseen/<scene>/`.

## 명령 (한 줄)
```
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1r_replay_verify.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes 17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX --gpu 0 --out logs/instruction_labeling/p1r
/usr/bin/python scripts/dataset_converters/instruction_labeling/render/p1a_action_gen.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --scenes all --gpu 1 --out logs/instruction_labeling/p1a
```
| 인자 | 뜻 | 왜 이 값 |
|---|---|---|
| `--scenes` (p1r) | 대조할 train scene | 기존 gs_vlnpe self-test 와 같은 3 scene |
| `--pixel_every` (기본 5) | 픽셀 비교 간격 | 전 프레임은 느리고 연속 프레임은 거의 같음 |
| `--scenes all` (p1a) | 61 scene 전체 | 렌더 없이 sim step 만 해서 20 분 |
| `--gpu` | EGL 렌더 GPU | GPU 메모리 대부분이 다른 작업에 쓰이는 중, 렌더는 소량 |

## P1-V val_unseen 재현 수치 (GT waypoint 를 공통 자로) — 추가
val 에는 릴리스 이미지·pose 가 없어 릴리스 대조가 불가능 → 모든 split 에 있는 reference_path(GT waypoint)에 대한 이탈을 같은 지표로 잼
(`render/p1v_path_stats.py`, `logs/instruction_labeling/p1v/`).
dev = 경로 점 → reference_path 선분 거리, wp_miss = GT waypoint → 경로 최단거리, end_to_goal = 끝점 → goal.

| 지표 | train 릴리스 | train 우리 (greedy_wp) | **val_unseen 우리** |
|---|---|---|---|
| trajectory 수 | 3,558 | 3,558 | 613 |
| dev 평균 (중앙) | 0.25 m (0.14) | 0.08 m (0.047) | **0.07 m (0.046)** |
| dev 최대 p95 | 2.90 m | 0.70 m | **0.56 m** |
| dev 최대 > 1 m 비율 | 14.3 % | 2.4 % | **2.9 %** |
| wp_miss 최대 중앙 / p95 | 0.24 / 0.50 m | 0.17 / 0.24 m | **0.18 / 0.24 m** |
| end_to_goal 중앙 / p95 | 0.12 / 0.25 m | 0.13 / 0.24 m | **0.12 / 0.24 m** |
| 끝점 3 m 이내 | 99.8 % | 100 % | **100 %** |
| 평균 프레임 수 | 68.3 | 62.5 | 61.9 |

- val_unseen 의 우리 경로는 train 의 우리 경로와 분포가 거의 같다 → 생성기가 split 에 관계없이 같은 품질.
- 우리 경로는 GT waypoint 를 **릴리스보다 더 가깝게** 따른다 (dev 평균 0.08 vs 0.25 m). 릴리스는 14 % 가 waypoint 선에서 1 m 이상 벗어난다.

## P1-V2 경로 3종 쌍별 거리 — 추가
- 데이터 출처: InternData `vln_ce/raw_data` json = VLN-CE 공식 `R2R_VLNCE_v1-3_preprocessed` 와 경로·시작·지시문 100 % 동일 (train 10,819 / val_unseen 1,839).
- VLN-CE GT 궤적 = 공식 `*_gt.json.gz` 의 `locations` (val_unseen 에도 있음). InternData train 릴리스 action 과는 27/10,684 만 일치 → InternData 는 이것을 쓰지 않았다.
- 거리 = 대칭 평균 최근접 거리 (A 각 점→B 최근접, B→A 를 평균), 최대 = Hausdorff. 스크립트 `render/p1v_vs_vlnce_gt.py`.

| 쌍 | split | 평균 거리 중앙 (p95) | 최대 거리 중앙 (p95) | 최대 > 1 m |
|---|---|---|---|---|
| 우리 ↔ VLN-CE GT | train | 0.07 (0.12) m | 0.25 (0.44) m | 0.1 % |
| 우리 ↔ VLN-CE GT | **val_unseen** | **0.07 (0.12) m** | **0.27 (0.45) m** | **0 %** |
| 릴리스 ↔ VLN-CE GT | train | 0.14 (0.63) m | 0.39 (2.76) m | 10.9 % |
| 우리 ↔ 릴리스 | train | 0.13 (0.59) m | 0.36 (2.74) m | 10.6 % |

## P1-E val_unseen 렌더 완료 — 추가 (2026-10-13 23시 KST)
- 11 scene, 613 trajectory, 1,839 episode. `verify_split.py` **전 scene 통과** (개수, 스키마, 이미지 형식, 같은 trajectory 사본 바이트 동일,
  지시문 일치, 05b V5 로더 재현 로드, V6 pose 불변식).
- 출력 `data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r/val_unseen/` 47 GB, 캐시 `_traj_cache/val_unseen/` 13 GB.
- 소요: scene 당 1 프로세스 병렬, 최장 scene 약 5 분.
- **장애**: GPU 0 에 프로세스 2개(zsNo4HB9uLZ, x8F5xyUWy9e)를 올렸더니 둘 다 EGL 단계에서 core dump.
  GPU 를 나눠 재실행하니 정상. GPU 당 렌더 프로세스는 1개로 둔다 (GPU 메모리 대부분을 vLLM 이 점유 중).
- **방향 확인**: 전진 23,261 스텝 중 카메라 전방과 이동 방향이 어긋난 것(cos<0.9) 15 건 (벽 미끄러짐). 렌더 방향 정상.
- **viz 버그 수정**: `viz_topdown.py` 의 quiver 가 y축 반전을 따르지 않아(`angles='uv'` 기본값) 남북 방향 화살표가 거꾸로 그려졌다.
  `angles='xy', scale_units='xy'` 로 수정하고 그림 전부 재생성 (`logs/instruction_labeling/topdown_val_unseen/`, `p1a/topdown/`).
  → 그림으로 방향을 검사할 때는 그림 코드 자체도 수치(이동 방향·카메라 전방 내적)로 교차검증한다.
