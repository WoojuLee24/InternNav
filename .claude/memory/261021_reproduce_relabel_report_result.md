# VLN 파이프라인 ③ — full 데이터 재현과 relabel 2x2 (2026-10-21 KST)

5090 서버의 파이프라인 report ① 환경과 경로 데이터셋, ② labeling 학습 평가 에 이어지는 편이다.
두 질문에 답한다: (1) 공개 InternVLA-N1 성능을 우리 손으로 재현했는가, (2) 사람 문장 없이 자동 생성 문장(label)으로 학습해도 되는가.
HTML 판: [VLN 파이프라인 ③ 재현과 relabel](https://claude.ai/artifact/S51KpgTbDm4swoTb6eSiSN) (비공개, 페이지 Share 메뉴로 공유), repo 사본 `.claude/memory/261021_reproduce_relabel_report.html`. 모든 시각 KST.

## 0. 용어
| 용어 | 뜻 |
|---|---|
| stage1 (s1) | System2 학습. Qwen2.5-VL-7B 에서 시작해 VLM 전체를 학습, "이미지 속 다음 목표 픽셀" 예측. 평가는 S2 + ShortestPathFollower |
| stage2 (s2) | stage1 위에 System1(로컬 정책 `nextdit_async`) 을 붙여 학습 (VLM 고정). 평가는 dual system |
| SR | 성공률: 멈춘 위치가 목표 3 m 안. **주 지표** |
| OS | 경로 중 한 번이라도 목표 3 m 안에 들어간 비율 |
| ld30 | 평가 때 아래로 30° 보기 (`tilt_angle: 15` × 2). 학습 데이터 pitch 와 같은 정본 |
| scalevln (sv) | 공개 학습 데이터 중 하나. upstream `train_system2.sh` 에 주석 처리되어 있다 |
| full / mini | full = 공개 학습 데이터 전체, mini = InternData-N1-v0.5-mini (R2R 61 scene) |

## 1. Full 데이터 재현 — 공개 성능을 우리 손으로 만들 수 있는가

**목적.** 공개 InternVLA-N1(DualVLN) 학습을 Qwen2.5-VL-7B 부터 다시 해서 README 수치(R2R val_unseen)를 재현한다.
평가: R2R val_unseen 1,839 episode, `vln_r2r_full_ld30.yaml`, H200 8장.

### 1-1. 결과 (SR / SPL / OS / NE)
공식 수치는 두 가지다: **wo-dagger** (DAgger 데이터 없이 학습, S2+SPF 55.4 / Dual NavDP 58.2) 와 **최종** (DAgger 포함, S2+SPF 60.9 / Dual DualVLN 64.3).
DAgger = 학습한 모델을 돌려 틀린 지점의 정답 행동을 다시 모아 추가 학습하는 방식. 그 데이터는 공개되지 않아 **우리 stage1 의 맞는 비교 대상은 wo-dagger** 다.

| 무엇 | 우리 | vs wo-dagger (DAgger 없음) | vs 최종 (DAgger 포함) |
|---|---|---|---|
| 공개 System2 ckpt 를 우리 평가로 | **61.7** / 55.7 / 68.2 / 4.18 | — | 60.9 → +0.8 (평가 정상) |
| 공개 DualVLN ckpt 를 우리 평가로 | **63.7** / 58.1 / 69.4 / 4.12 | — | 64.3 → −0.6 (평가 정상) |
| stage2: 공개 System2(DAgger 포함) 위에서 우리가 학습 | **62.9** (08-20) / **63.7** (09-30) | — | 64.3 → −0.6~−1.4 (재현) |
| stage1: upstream 그대로 (scalevln 없음, 2 epoch) | 44.5 / 40.1 / 56.0 / 6.45 | 55.4 → **−10.9** | 60.9 → −16.4 |
| stage1 + scalevln, 1 epoch (최종) | 49.2 / 44.1 / 60.3 / 5.84 | 55.4 → **−6.2** | 60.9 → −11.7 |
| stage1 + scalevln, 2 epoch (최종) | 49.4 / 44.6 / 61.7 / 6.06 | 55.4 → **−6.0** | 60.9 → −11.5 |
| stage1 + scalevln 1 epoch 중간 ckpt-24000 (최고) | **52.6** / 49.2 / 60.4 / 5.28 | 55.4 → **−2.8 (근접)** | 60.9 → −8.3 |
| 전체 체인: 우리 stage1(44.5) → stage2 | 46.8 | 58.2* → −11.4 | 64.3 → −17.5 |
| 전체 체인: 우리 stage1(sv 1 epoch, 49.2) → stage2 | **52.4** / 46.9 / 62.7 / 5.53 | 58.2* → −5.8 | 64.3 → −11.9 |

\* Dual 의 wo-dagger 공식 수치는 NavDP(58.2) 만 있고 DualVLN 은 없어 참고값.

### 1-2. 학습 run
| run | 내용 | 기간 (KST) | 상태 |
|---|---|---|---|
| `baseline/dualvln_stage2_full_20260818_101948` | stage2, 공개 System2 에서 시작, 18,321 step | 08-18 → 08-20 | 완료 (62.9) |
| `baseline_ablation/dualvln_stage2_full_cl_20260928_120708` | 같은 stage2, channels_last | 09-28 → 09-30 | 완료 (63.7) |
| `baseline_ablation/dualvln_stage1_full_gemm_20260825_041953` | stage1, upstream 데이터 8종, 2 epoch (48,974 step) | 08-25 → 08-30 | 완료 (44.5, 과적합 loss 0.04) |
| `baseline_ablation/dualvln_stage2_from_stage1_gemm_20260908_160005` | 위 stage1 → stage2 | 09-09 → 09-10 | 완료 (46.8) |
| `baseline_ablation/dualvln_stage1_sv_ep1_..._20260929_053543` | stage1 + scalevln 2종, 1 epoch (47,249 step) | 09-29 → 10-04 | 완료 (49.2, 중간 52.6) |
| `baseline_ablation/dualvln_stage1_sv_ep2_..._20260929_053803` | stage1 + scalevln, 2 epoch (94,498 step) | 09-29 → 10-09 | 완료 (49.4) |
| `baseline/dualvln_stage2_from_stage1_20261006_071041` | sv 1 epoch stage1 → stage2 | 10-06 → 10-08 | 완료 (52.4) |

### 1-3. 핵심
1. **평가와 stage2 는 재현된다.** 공개 ckpt 를 우리 평가로 재면 공식과 ±1 이내, 공개 System2(DAgger 포함) 위의 stage2 학습도 64.3 대비 62.9~63.7.
2. **stage1 은 맞는 기준(wo-dagger 55.4)에 근접했지만 아직 못 미친다.** 최고 중간 ckpt 52.6 (−2.8), 최종 ckpt 49.2~49.4 (약 −6).
   "거의 재현" 이라 하려면 최종 ckpt 가 ±1.5 안에 들어와야 한다.
3. **scalevln 을 넣으면 +5** (44.5 → 49.2). 2 epoch 는 과적합 (loss 0.04) 이라 1 epoch 중간 ckpt 가 가장 좋았다.
4. **공식 최종과의 나머지 차이 (55.4 → 60.9, 5.5 %p) 는 DAgger 데이터 몫** 으로 본다. README 에 두 수치가 따로 있어 크기를 알 수 있고, 우리는 그 데이터가 없다.
5. **stage2 는 약한 stage1 을 메우지 못한다** (체인 52.4).
6. 그래서 instruction labeling 실험(2장)은 full 이 아니라 **mini 데이터 + stage1 1 epoch** 로, GT 학습과 label 학습을 같은 조건에서 비교한다.

정정: `.claude/memory/dualvln_stage1_stage2_repro_result.md` 의 "stage2_full_20260818 이 step 4993 에서 사망" 은 틀렸다 — 18,321 step 까지 완료하고 평가됐다.

## 2. Relabel 2x2 — 사람 문장 없이 학습해도 되는가

**목적.** R2R 사람 지시문(GT) 대신 자동 생성한 지시문(label)으로 학습했을 때, 사람 문장으로 평가한 성능이 GT 학습에 얼마나 가까운가.
episode·경로·이미지는 GT 와 같고 **문장만** 바꾼다.

### 2-1. 실험 설정 (relabel_half)
| 항목 | 값 | 왜 |
|---|---|---|
| 데이터 | InternData-N1-v0.5-mini `vln_ce` R2R train 61 scene / 10,684 episode | GT 학습과 같은 이미지·궤적. 문장만 교체 (`build_relabel_dataset.py`) |
| 모델 | s1 (System2 단독, Qwen2.5-VL-7B 시작, VLM 전체 학습) **1 epoch** | mini 에서 2 epoch 는 과적합 (GT 39.4 < 1 epoch 42.5) |
| 평가 | val_unseen 1,839 episode, `vln_r2r_mini_ld30.yaml` (아래 30°) | 모든 셀이 문장 파일 한 줄만 다르다 |
| 셀 | label 마다 3칸: label→GT (학습+평가), label→label, GT→label | 셀당 약 22 h (H200 8장) |
| 오차 | 학습 1회, 같은 episode 짝 비교 95 % 구간 약 ±2 %p | ±2 %p 안의 차이로는 결론 내지 않는다 |

### 2-2. label 계보
| label | 생성기 | 내용 |
|---|---|---|
| v278 | Kemal v278: gemma-4-31B AWQ 2단계 (프레임별 장면 요약 글 → 글만 보고 작성) | **2x2 학습 안 함** (버그 3개 상태) |
| v278fix | v278 + 프레임 버그 3개 수정 (waypoint 기준 key frame, HFOV 79, 검은 프레임 제거), seed 고정 | 이전 기준선 |
| **b0** | 이 repo `instruction_labeling` e2e (이미지 여러 장 + 경로 단어 → VLM 1회), gemma-4-31B bf16 | InternNav 학습·평가와 **같은 이미지**, 올바른 카메라 방향 (v278 계열은 회전·도착 프레임이 거울상) |
| **b1** | b0 + 살짝 회전 생략 + GT 문체 예시 3개 + 모든 waypoint 프레임 + 방 이동 줄(MP3D .house) | **현재 기준** (2026-10-20 지정) |

### 2-3. 결과 — 2x2 (SR / OS %, val_unseen 1,839)
- label→GT = label 로 학습, 사람 문장으로 평가. **주 지표**.
- label→label = 자기 문장으로 평가. 문체를 외우면 부풀려진다.
- GT→label = 사람 문장으로 학습한 모델에 label 문장. **문장에 경로 정보가 충분한가** (정보량).

| 학습 label | label→GT | label→label | GT→label |
|---|---|---|---|
| GT (상한) | **42.5** / 52.5 | — | — |
| **b1** | **37.4** / 50.1 | 52.9 / 66.0 | **44.0** / 56.8 |
| b1 − fewshot (b1_fewshot-0) | 35.3 / 49.4 | 53.2 / 65.5 | 44.2 / 57.0 |
| b1 + 정지 단서(물체) | 35.9 / 47.7 | 49.4 / 64.8 | 42.0 / 54.3 |
| C+방v2 (b1 − 모든 waypoint) | 35.9 / 48.9 | 51.9 / 64.1 | 41.4 / 53.6 |
| 정지 단서 고르기 vlm / rotate / A-grounded (b0 위) | 34.6 ~ 35.0 | 51.5 ~ 51.9 | 40.1 ~ 41.4 |
| C+방v2, Qwen3-VL-32B 로 생성 | 33.8 / 48.3 | 47.8 / 61.0 | 37.4 / 52.1 |
| b0_delivery-2stage-v1 (b0 를 v278 식 2단계로) | 33.8 / 52.5 | 54.7 / 68.0 | 38.1 / 52.9 |
| **b0** | 32.1 / 49.4 | **56.3** / 67.9 | 40.5 / 52.2 |
| **v278fix** (이전 기준선) | 31.9 / 40.0 | 31.9 / 41.9 | 30.0 / 43.4 |
| v278 (버그 수정 전) | 학습 안 함 — 프레임 버그 3개를 고친 v278fix 부터 2x2 를 돌렸다 | | |
| C+방v2, Qwen3-VL-235B 로 생성 | 30.6 / 50.8 | 48.3 / 63.7 | 38.8 / 51.0 |

### 2-4. 실패 유형 (label→GT, %)
지나침 = 목표 3 m 안에 들어갔다가 다른 곳에서 멈춤 / 못 감 = 한 번도 못 들어감.

| label | 성공 | 지나침 | 못 감 |
|---|---|---|---|
| GT | 42.5 | 약 10 | 약 47.5 |
| b1 | 37.4 | 12.7 | 49.6 |
| b0 | 32.1 | 15.8 | 49.5 |
| v278fix | 31.9 | 8.1 | 60.0 |

### 2-5. 핵심
1. **b1 은 GT 의 88 %** (37.4 / 42.5), v278fix 대비 **+5.5 %p**.
2. **문장 정보는 이미 충분하다.** b1 문장을 GT 학습 모델에 주면 44.0 으로 사람 문장(42.5) 이상. v278fix 는 30.0 — 정보 손실이 컸다.
   b0 에서 정보가 늘어난 이유는 생성 방식(1단계/2단계)이 아니라 **입력 이미지와 카메라 방향** 이었다 (2단계로 되돌려도 33.8).
3. **남은 차이(5 %p)는 문체 차이다.** b1 은 자기 문장 52.9 / 사람 문장 37.4. 같은 경로의 문장 3개가 거의 같다
   (형제 BLEU-4 0.72 vs GT 0.14, 첫 3단어 같음 76 % vs 4 %).
4. **방 이동 정보가 가장 효과적이었다** (+3.2): 목표를 지나치는 비율을 17.7 → 12.7 % 로 줄였다. 정지 단서는 지나침은 줄이지만 못 감을 늘려 순이득이 작다.
5. **생성 모델은 gemma-4-31B 유지.** Qwen3-VL 32B −2.1, 235B −5.3 (235B 는 지나침 19 %).
6. **문장 지표(CIDEr 등)는 SR 순위와 맞지 않는다.** 판정은 정보량 검사 → 학습 2x2 순으로 한다.

### 2-6. 진행 중 — 형제 문장 다양화 (b1 기반, 2026-10-21 KST)
| 실험 | 바꾼 것 | 첫 3단어 같음 (GT 3.9 %) | 정보량 GT→label (b1 44.0, 기준 42) | 학습 |
|---|---|---|---|---|
| 1 | 형제마다 서술 방식(step / landmark / brief) + 그 문체의 fewshot | 61.6 % | 43.6 통과 | node4, 결과 10-21 15:40 |
| 2 | 출발 지점 좌·우·뒤 이미지 | 76.0 % | 43.6 통과 | node2, 결과 10-22 07:50 |
| 3 | 1 + 2 + landmark·brief 는 시작 회전 생략 | 20.7 % | **39.7 미달** | node3, 결과 10-21 15:40 |
| pathrule | 1 + 2 + 경로 사실로 첫머리 배정 (형제마다 다른 첫머리) | 19.8 % (dev) | **40.2 미달** | 대기 |

- 실험 3 손실 원인: 크게 돌아서 출발하는 경로에서 "Head forward" 처럼 **틀린 방향**으로 시작 (해당 문장 SR 26 %). 사람이 회전을 생략할 때는 "Leave the bedroom" 처럼 방으로 시작한다 (GT 생략 문장 41 %).
- pathrule 손실 원인: "목적지 먼저" 첫머리 (전체의 33 %) 가 −8.9 %p. 회전·직진·물체 첫머리는 손실 없음 → 다음 버전에서 목적지 첫머리를 뺀다.

## 3. 재현 위치
| 항목 | 위치 |
|---|---|
| full 재현 config | `scripts/train_eval/qwenvl_train/baseline/`, `.../baseline_ablation/` |
| full 재현 결과 | `/home/irteam/data-vol2/checkpoints/{baseline,baseline_ablation}/<run>/logs/*/result_h200.json` |
| label 생성 | `scripts/dataset_converters/instruction_labeling/` (README), 이전 label 은 `vln-s2-pipeline/annotate.py` |
| relabel 셀 | `scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.<L>_eval.<E>.py` |
| relabel 결과 | `/home/irteam/data-vol2/checkpoints/relabel_half/<run>/logs/<cell>/result_h200.json` |
| 이전 상세 보고 | `scripts/dataset_converters/relabel_vlnce/report_relabel_2x2_20261013.md`, `.claude/memory/261015_instruction_labeling_sr_first_result.md` |
