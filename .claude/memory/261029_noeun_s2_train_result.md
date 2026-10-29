# 노은역 S2 학습 (relabel_half_noeun-v01) — WAIT 행동 포함 · 진행 기록 2026-10-29 KST

계획: `/home/irteam/.claude/plans/crispy-wobbling-thimble.md` (사용자 승인).

## 용어
- **WAIT 샘플** = S2 정답 텍스트가 `WAIT` 인 학습 샘플. hunav 에서 로봇이 사람 때문에 멈춘 프레임 (parquet action 4, 로더 시프트 후 `actions[t]==4`).
- **wait_mode**: `none` = 기본 trainer·로더 (변경 없음) · `wait` = 대기 프레임마다 WAIT 샘플 x `wait_weight`(5) · `exclude` = 대기 프레임 샘플 없음 (WAIT 없는 대조군).
- ⚠️ habitat 평가기에서 action 4 = LOOKUP. WAIT 는 텍스트로 따로 처리해야 한다.

## 실험 10 개 (공개 InternVLA-N1-System2 출발, `r2r_60cm_15_15`, 1 epoch, `--no-eval`)
| 셀 (`relabel_half_noeun/s1.<label>_wait-<mode>.py`) | 노드 |
|---|---|
| noeun-both-{b1,post} wait · noeun-v6-b1 wait | node2 |
| noeun-both-{b1,post} exclude · noeun-v6-post wait | node3 |
| noeun-hunav-{b1,post} wait · exclude | node4 |
데이터 = 경로(v6 200 / hunav 187 / both 387) x 라벨(b1 / post) x 형제 3 문장 (scene = `<ds>_s<k>`).

## 코드 (신규 + default_config 필드 추가만)
| 파일 | 하는 일 |
|---|---|
| `internnav/dataset/internvla_n1_wait_lerobot_dataset.py` | `NavPixelGoalWaitDataset` — 부모 샘플 목록 후처리 (대기 프레임 시작 pixel/turn 제거, turn 은 4 에서 자름, wait 면 WAIT 샘플 추가) |
| `internnav/trainer/internvla_n1_wait_trainer.py` | `--wait_*` 떼고 데이터셋 이름 바꿔 `_base.train()` |
| `scripts/train_eval/qwenvl_train/default_config.py` | `wait_mode`·`wait_weight` + trainer guard + argv (none 이면 그대로) |
| `scripts/train_eval/qwenvl_train/relabel_half_noeun/` | `noeun_base.py` · `make_cells.py` · 셀 10 개 |
| `scripts/dataset_converters/nav3d/noeun/64_build_train.py` | 학습·val 데이터셋 12 개 (`<mini>/vln_ce_noeun-<paths>-<label>[-val]`, 원본 `vln_ce_noeun_src`) |
| `scripts/eval/noeun_openloop_wait.py` | 열린 루프 WAIT recall · 오발 (habitat 평가기와 같은 입력) |
| `scripts/eval/configs/noeun/noeun_val_seen_v01.yaml` | 다른 노드 Isaac 평가용 정보 (에피소드 · 카메라 · 출력 토큰 · 지표) |

## 검증 (CPU)
- 기존 셀 4 개 (relabel_half s1·s2, baseline stage1) `train_argv` · trainer 변경 전후 동일.
- v6 (대기 없음): wait 로더 샘플 목록 = 기본 로더 (pickle 동일, 27,273).
- hunav-b1: 기본 로더는 turn 목록에 4 가 든 샘플 69 개 (학습 중 KeyError 날 것). wait: 대기 프레임 243 (= 81 x 3), WAIT 샘플 1,215 (= 81 x 3 x 5), 제거 63 · 자름 6, 4 남은 turn 0. exclude: WAIT 0. pixel 23,643 · stop 2,805 는 그대로 (대기 프레임엔 보이는 목표가 없어 원래 turn 샘플).
- `__getitem__` (공개 System2 processor): WAIT 샘플 정답 토큰 `WAIT<|im_end|>`, STOP 샘플 `STOP<|im_end|>`.
- ⚠️ 대기 프레임 20 장 표본 중 약 절반만 사람이 60cm_15deg 정면에 보인다 (나머지는 시야 밖) — 이미지로 근거가 없는 WAIT 정답이 섞여 있다.

## 남은 것
- GPU smoke (node1 job 끝난 뒤), 학습 10 개, Drive 업로드, 열린 루프 평가 결과.
- 나중에: (가) 대기 문장 라벨, vln_pe 로더.
