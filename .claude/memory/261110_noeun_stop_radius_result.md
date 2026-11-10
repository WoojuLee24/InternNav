# 2026-11-10 노은역 S2 반경 STOP 라벨 (noeun_stop_radius)

## 용어
| 용어 | 뜻 |
|---|---|
| STOP 샘플 | 정답이 `STOP` 인 학습 샘플. 튜플 `(…, (f, f+1), 0, None)` |
| 기존 STOP | 에피소드마다 마지막 프레임 1 장 × `stop_weight`(5) |
| `stop_radius_m` (r) | 반경 STOP 반경. 마지막 프레임까지 xy 거리 ≤ r 이고 남은 행동이 전진뿐인 프레임을 모두 STOP 으로. 0 = 끔 |
| 조건 (a) | `poses_60cm_15deg` 의 xy 로 잰 마지막 프레임까지 거리 ≤ r |
| 조건 (b) | 그 프레임부터 끝까지 남은 action(시프트된 것)이 전진(1)뿐 — 아직 돌아야 하면 STOP 금지 |
| 고유 개수 | 가중치(stop_weight 5 · wait_weight 5) 곱하기 전 개수 = 기존 `rank0_print(turn, pixel, stop)` 와 같은 집계 |

## 바뀐 점 (기본값이면 기존과 100 % 동일)
| 파일 | 변경 |
|---|---|
| `internnav/dataset/internvla_n1_lerobot_dataset_stop_radius.py` (새) | 부모가 만든 목록에 반경 STOP 추가 · 같은 프레임 pixel/turn 제거. `NavPixelGoalStopRadiusDataset`(NavPixelGoalDataset 상속), `NavPixelGoalWaitStopRadiusDataset`(WAIT 데이터셋 상속), `STOP_RADIUS_DATASETS` 이름 표 |
| `internnav/trainer/internvla_n1_stop_radius_trainer.py` (새) | WAIT trainer 와 같은 방식으로 데이터셋 이름을 바꿔 끼움. `--wait_mode` 있으면 WAIT+반경 |
| `internnav/trainer/internvla_n1_argument.py` | `DataArguments.stop_radius_m: float = 0.0` 한 줄 |
| `scripts/train_eval/qwenvl_train/default_config.py` | `Params.stop_radius_m = 0.0`, `trainer` 맨 앞 guard (`> 0` 일 때만 새 trainer), `> 0` 일 때만 `--stop_radius_m` emit |
| `relabel_half_noeun/noeun_base.py` | `STOP_RADII=(0.5, 1.0)`, `stage1_stop_radius()` = `stage1()` + `stop_radius_m` |
| `relabel_half_noeun/make_cells.py` | 셀 2 개 생성: `s1.noeun-both-b1_wait-wait_stopr0.5.py`, `_stopr1.0.py` |
| `relabel_half_noeun/check_stop_radius.py` (새) | 학습 전 검증 (아래) |

## 학습 전 검증
명령: `python3 scripts/train_eval/qwenvl_train/relabel_half_noeun/check_stop_radius.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-both-b1_wait-wait.py --out logs/noeun_stop_radius/check`
(데이터 인자는 셀의 실제 학습 argv 를 DataArguments 로 파싱. 데이터 noeun-both-b1, r2r_60cm_15_15, 에피소드(지시문 단위) 1,161)

### 기본 경로 불변
- r=0 샘플 목록 == 기존 `NavPixelGoalWaitDataset`: **튜플 단위 동일** (56,199 개). 기본 로더 `NavPixelGoalDataset` 과도 동일 (55,047).
  - 56,199 ≈ v01 학습 실측 56,192 (= 439 step × 128) → v01 과 같은 데이터
- 기존 config 48 개 (noeun 42 + relabel_half 6) 의 train argv · trainer 가 origin/main 과 **바이트 동일**.
- 셀 argv 차이 (기준 대비): `--run_name` 과 `--stop_radius_m` 뿐.
- GPU 스모크 (stopr1.0, 4 GPU, 2 step): `NavPixelGoalWaitStopRadiusDataset` 로드, loss 1.67 → 1.55, rc=0.

### 샘플 수 (turn / pixel / stop / wait = 고유 개수, total = 가중치 곱한 학습 샘플 수)
| r (m) | turn | pixel | stop | wait | total | 1 epoch step (÷128) | 반경 프레임 | 뺀 pixel |
|---|---|---|---|---|---|---|---|---|
| 0 (v01) | 3,438 | 45,741 | 1,161 | 243 | 56,199 | 439 | 0 | 0 |
| 0.5 | 3,438 | 45,741 | 2,559 | 243 | 63,189 | 494 | 1,398 | 0 |
| 1.0 | 3,438 | 45,699 | 4,737 | 243 | 74,037 | 579 | 3,576 | 42 |
| 1.5 | 3,438 | 45,174 | 6,750 | 243 | 83,577 | 653 | 5,589 | 567 |

- 학습 샘플 중 STOP 비율: r=0 10.3 % → 0.5 20.2 % → 1.0 32.0 % → 1.5 40.4 %
- r=0.5 에서 뺀 pixel 0: 마지막 ~0.5 m 는 원래 라벨이 없던 구간 (끝 직전 프레임 건너뜀 + goal_len<3 버림)

### 경로 길이 구간별 STOP 고유 개수
| r (m) | 0-10 m | 10-30 m | 30-60 m | 60+ m |
|---|---|---|---|---|
| 에피소드 수 | 174 | 378 | 294 | 315 |
| 0 | 174 | 378 | 294 | 315 |
| 0.5 | 387 | 834 | 642 | 696 |
| 1.0 | 726 | 1,572 | 1,191 | 1,248 |
| 1.5 | 1,053 | 2,280 | 1,707 | 1,710 |

- 조건 (b) 로 빠진 프레임이 있는 에피소드 (r=1.0): 132 / 1,161 (11.4 %)

### 시각화 (`logs/noeun_stop_radius/check/`, 끝 10 프레임, 빨간 테두리 = STOP)
| 파일 | 에피소드 | 반경 STOP 프레임 (남은 거리 m), r=1.0 |
|---|---|---|
| `short_v6_s0_ep146_r1.0.jpg` | 3.2 m 경로 | 12 (0.76), 13 (0.51), 14 (0.26). 11 (1.02) 은 반경 밖 |
| `most_hunav_s0_ep152_r1.0.jpg` | 3.5 m | 11 (0.99), 12 (0.77), 13 (0.50), 14 (0.25) |
| `turn_in_radius_hunav_s0_ep0_r1.0.jpg` | 48 m | 204 (0.80), 205 (0.49), 206 (0.27). 203 (0.78 m) 은 회전이라 (b) 로 제외 |

## 학습
(진행 후 기록)
