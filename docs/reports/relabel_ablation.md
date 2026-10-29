# Relabel ablation 보고서 — DualVLN-mini · R2R mini (2026-09-29 ~ 10-29)

학습 instruction(문장)만 바꿔 DualVLN-mini 를 학습하고, 사람 문장(GT)으로 평가한 실험 전부 (평가 172 개, node1~4).
원자료: `relabel_ablation_results.json` (수집: `collect_relabel_results.py`, 결과 위치 `data-vol2/checkpoints/relabel{,_half}/<run>/logs/<cell>/result_h200.json`).

## 한눈에
1. **최고 생성 라벨 b1 = SR 37.4, 사람 문장(GT) 42.5 의 88 %.** 이전 기준 v278fix(31.9)보다 +5.5.
2. **남은 5 %p 는 정보 부족이 아니라 문체 차이다.** GT 로 학습한 모델에 b1 문장을 주면 44.0 (> GT 42.5) — 경로 정보는 충분하다. b1 형제 문장끼리 너무 비슷하다 (BLEU-4 0.72 vs GT 0.14).
3. **가장 효과가 큰 추가 정보 = 지나는 방 순서 (+3.2).** 정지 단서·모델 교체·GEPA 프롬프트 최적화는 오차(±2) 안이거나 손해.
4. **문장 품질 지표(CIDEr)로는 고를 수 없다.** CIDEr 순서와 SR 순서가 반대인 경우가 있다 -> 결정 순서 = 정보량 검사 -> 학습.
5. **nav3d 경로: 지름길(sg) 경로 + 그 경로로 만든 b1 라벨 조합에서만 -6 %p.** sg 경로 + 사람 문장(41.8)·wp 경로 + sg 라벨(35.6)은 괜찮다.

## 용어 (숫자를 읽기 전에)
| 용어 | 뜻 |
|---|---|
| SR | 성공률. R2R val_unseen 1,839 ep 에서 목표 3 m 안 STOP 비율 (%) |
| **label→GT** | 그 라벨로 학습, 사람 문장으로 평가. **주 지표** (라벨이 학습 데이터로 쓸만한가) |
| label→label | 그 라벨로 학습, 같은 라벨 문장으로 평가. 문체가 단조로우면 부풀려진다 (b0 56.3) |
| **정보량** (GT→label) | GT 로 학습한 ckpt 를 그 라벨 문장으로 평가 (학습 없음). 문장에 경로 정보가 얼마나 있나. 10-20 부터 학습 전 관문 (기준 ≈ 42) |
| 전 / 후 | 전 = `b1_facts-memo-v1_knobs-v2_delivery-joint3-v1` (형제 3 문장 한 번에), 후 = 전 + GEPA 최적화 프롬프트 (`_writer-gepa-v3-joint-gemma`) |
| b1 | b0 + 작은 회전 생략 + GT 예시 3 개 + 모든 경유점 key frame + 방 순서 (`b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2`, 10-20 지정) |
| C | `b0_slight-omit_fewshot-3` (b0 축 조합 중간 단계) |
| v218 · v278 | Kemal 생성기 저장소의 스크립트 번호. v218 = 규칙/템플릿 (LLM 없음), v278 = gemma 2 단계 생성기 |
| sg · wp 경로 | nav3d 가 navmesh 로 다시 계획한 경로. sg = 시작·끝만 (지름길), wp = 사람 경로 경유점을 따라 |

## 실험 설정
- 모델·데이터: DualVLN (Qwen2.5-VL-7B S2 + nextdit S1), InternData-N1-v0.5-mini R2R (61 scene, 10,684 ep). 학습 문장만 바꾼다 (`<mini>/vln_ce_<label>`).
- **stage1 (S2 만)**: Qwen2.5-VL-7B 에서 출발, 전체 미세조정, lr 2e-5. 평가 = S2 + ShortestPathFollower.
- **stage2 (S1)**: VLM 고정, S1 3 epoch, 같은 라벨 stage1 에서 출발. 평가 = dual system.
- **relabel/** = stage1 2 epoch (13,810 step, ~33 h). **relabel_half/** = stage1 1 epoch (6,905 step, ~16.5 h). mini 에서는 1 epoch 가 더 좋다 (GT 39.4 -> 42.5).
- 10-13 부터 stage2 생략 (순위는 stage1 과 같고 차이만 23~26 로 압축, 셀당 ~10 h).
- 평가: val_unseen 1,839 ep, look-down 30° (`vln_r2r_mini_ld30.yaml`).
- **오차**: 학습 1 회 셀. 같은 설정 재학습 차이 0.3~0.9 SR (10-28), 평가 표본 오차 ±2 %p (95 %). **2 %p 미만 차이로는 결론 내지 않는다.** ep 단위로는 재학습마다 21~23 % 가 뒤집힌다.

## 결과 (relabel_half stage1, SR %: label→GT / label→label / 정보량)

### 1. 기준선
| 학습 | label→GT | 비고 |
|---|---|---|
| gt (relabel_half, 1 epoch) | **42.5** | 모든 비교의 기준. stage2 33.7 |
| gt (relabel, 2 epoch) | 39.4 | 2 epoch 는 mini 에서 과적합. stage2 28.5 |

### 2. Kemal 생성기 재현과 v278fix 프롬프트 ablation
| 학습 | 바꾼 것 | label→GT / →label / 정보량 |
|---|---|---|
| v6 (relabel) | 초기 gate3-Gemma 라벨 (템플릿, 고유 문장 17 %) | 8.7 / 11.6 / 24.7 (loss 0.589, 실패) |
| v218 (relabel) | 규칙/템플릿, train 에 랜드마크 없음 | 21.4 / 34.4 / 31.6 |
| v218_reproduce (relabel) | v218 재생성, train 에 랜드마크 | 28.5 / 42.4 / 32.0 |
| v218fix | v218 + 렌더 수정 (HFOV 79, 검은 프레임) | 27.3 / 40.5 / 30.9 |
| **v278fix** (이전 기준) | v278 + 프레임 버그 3 개 수정 + seed | **31.9** / 31.9 / 30.0 |
| v278fix_stop-gtdist | 끝맺음 동사를 GT 분포로 | 32.4 / 32.6 / 30.1 |
| v278fix_fewshot-3 | GT 예시 3 개 | 33.6 / 32.7 / 29.3 |
| v278fix_fewshot-5 | GT 예시 5 개 | 31.9 / 31.2 / 29.9 |
| v278fix_div-on | 형제 문장을 보여주고 다르게 | 29.7 / 30.9 / 27.8 |
| v278fix_len-gtdist | 길이를 GT 분포로 | 29.7 / 27.9 / 28.6 |
| v278fix_len-free | 길이 제한 없음 | 33.7 / 33.4 / 28.4 |
| v278fix_g3temp-070 | 장면 설명 temperature 0.4 -> 0.7 | 30.1 / 31.3 / 29.2 |
| v278fix_g4temp-100 | 문장 temperature 0.35 -> 1.0 | 32.5 / 30.0 / 28.9 |
- v278 계열은 정보량이 28~30 (GT 42.5 보다 12 %p 낮음) — **프롬프트를 아무리 바꿔도 30 근처** -> 생성기 구조 문제 (v278 프레임이 좌우 반전, 경로 정보 손실). fewshot-3 의 +1.7 은 fewshot-5 에서 재현 안 됨.

### 3. 새 생성기 b0 와 축 조합 -> b1
| 학습 | 바꾼 것 | label→GT / →label / 정보량 |
|---|---|---|
| b0 | InternNav 이미지 + 경로 단어로 VLM 1 회 (gemma-4-31B), 카메라 방향 수정 | 32.1 / **56.3** / 40.5 |
| b0_delivery-2stage-v1 | 장면을 v278 식 2 단계 텍스트로 | 33.8 / 54.7 / 38.1 |
| b0_slight-omit | 60° 미만 회전은 말하지 않음 | 33.8 / 50.5 / 39.4 |
| C (= + fewshot-3) | + GT 예시 3 개 | 32.7 / 49.4 / 39.5 |
| C + 출발 주변 시점 (SS) | 출발점 좌·우·뒤 사진 추가 | 33.7 / 50.8 / 38.9 |
| C + 모든 경유점 (WP) | 경유점마다 key frame | 33.7 / 52.5 / 43.6 |
| C + 방 순서 (방v2) | "leave the bathroom, enter the hallway…" 한 줄 | 35.9 / 51.9 / 41.4 |
| C + SS + 방v2 | | 35.6 / 51.2 / 42.2 |
| **C + WP + 방v2 = b1** | | **37.4** / 52.9 / **44.0** |
| b1 - fewshot | GT 예시 없이 | 35.3 / 53.2 / 44.2 |
- b0 는 정보량을 되찾았지만(40.5) label→label 56.3 vs label→GT 32.1 — **문체가 단조로워 사람 문장으로 일반화가 안 됨** (지나침 15.8 %).
- 방 순서 +3.2 (지나침 17.7 -> 12.7 %) 가 단일 축 최대. 경유점 key frame 은 정보량 +4.1.

### 4. 생성 모델 교체 (C + 방v2 기준 35.9)
| 모델 | label→GT / →label / 정보량 |
|---|---|
| gemma-4-31B (기준) | 35.9 / 51.9 / 41.4 |
| Qwen3-VL-32B | 33.8 / 47.8 / 37.4 (-2.1) |
| Qwen3-VL-235B-A22B fp8 | 30.6 / 48.3 / 38.8 (-5.3, 지나침 19 %) |
- 큰 모델이 더 좋지 않다 -> gemma 유지.

### 5. 정지 단서 (도착 지점 물체를 문장에)
| 학습 | label→GT / →label / 정보량 |
|---|---|
| b0_stop_cue-A-grounded-v1 (2 m 안 보이는 물체) | 34.6 / 51.9 / 41.4 |
| b0_stop_select-rule-v1 | 33.0 / 48.6 / 39.5 |
| b0_stop_select-vlm-v1 | 35.0 / 51.7 / 40.3 |
| b0_stop_select-rotate-v1 | 34.7 / 51.5 / 40.1 |
| b1 + stop_cue-A-grounded-v1 | 35.9 / 49.4 / 42.0 (b1 37.4 보다 -1.5) |
- 지나침은 줄지만 못 미침이 늘어 순효과 없음.

### 6. 형제 문장 문체 (b1 기준 37.4)
| 학습 | label→GT / →label / 정보량 |
|---|---|
| b1 + 형제마다 다른 문체 + 같은 문체 GT 예시 (실험 1) | 37.7 / 52.3 / 43.6 |
| b1 + 출발 주변 시점 (실험 2) | 36.3 / 52.6 / 43.6 |
| b1 + SS + 문체 v2 (실험 3) | **38.4** / 44.0 / 39.7 |
| 경로 규칙 문체 (pathrule-v1) | 학습 안 함 (정보량 40.2, 관문 미달) |
- 실험 3 이 R2R 최고 label→GT 지만 정보량 39.7 로 관문 미달 (큰 회전 경로에서 "Head forward" 로 잘못 시작). +1.0 은 오차 안.

### 7. GEPA 사다리 (사실 메모 -> 손잡이 -> 형제 함께 -> GEPA 프롬프트)
| 학습 | 바꾼 것 | label→GT / →label / 정보량 |
|---|---|---|
| b1_facts-memo-v1 | 번호 명령 목록 -> 서술형 사실 메모 | 35.1 / 52.5 / 42.7 |
| + knobs-v1 | 형제마다 끝맺음·설명 방식·길이를 GT 비율로 | 37.7 / 53.1 / 42.4 |
| + knobs-v1 + GEPA v4 (gemma 채점) | GEPA 최적화 프롬프트 | 35.0 / 52.4 / 40.8 (문장 이상: 사실 오류 24 %) |
| **전** (knobs-v2 + 형제 3 문장 한 번에) | | 35.1 / 42.9 / 38.1 (10-26 재학습; 10-24 run 은 발산, 무효) |
| **후** (전 + GEPA v3) | | 37.3 / 48.1 / 39.9 (전 대비 +2.2, 오차 경계) |
| knobs-v1 + GEPA v4 (Sonnet 채점) | | 학습 안 함 (정보량 41.8) |
- 형제 다양성은 사람 수준이 됐지만 (첫 3 단어 같은 비율 3 %) SR 은 b1 과 같거나 낮다. GEPA 효과는 오차 안.

### 8. nav3d 경로 (경로를 navmesh 로 다시 계획 + 그 경로로 라벨)
| 학습 (경로 + 문장) | label→GT / →label / 정보량 |
|---|---|
| R2R 경로 + b1 (참고) | 37.4 / 52.9 / 44.0 |
| wp + b1 (2 회) | 36.9 · 36.6 / 51.8 / 39.0 |
| wp + 후 | 34.0 / 44.1 / 35.2 |
| sg + b1 (2 회) | 31.1 · 30.2 / 49.0 / 40.7 |
| sg + 후 | 29.1 / 42.3 / 37.1 |
| **sg + 사람 문장** (교차) | **41.8** |
| **wp + sg-b1 라벨** (교차) | **35.6** |
- wp 경로는 R2R 경로와 거의 같다 (-0.5). sg 경로 + 자기 b1 라벨만 -6 (두 쌍·두 회 모두). 후 < b1 (sg -2.0, wp -2.9).
- sg + 사람 문장은 sg + b1 과 똑같이 짧게 걷는데(걸은 거리/사람 경로 0.88, 53 step) 성공 -> "짧은 경로가 조기 정지를 가르친다" 가설 **반증**.
- 실패 양상: sg + b1 은 출발 직후 이탈과 경로 위 조기 정지가 많고, 긴 에피소드(최단 12 m+)에서 20.5 (sg + 사람 35.4).
- 문장 회전 표현 vs 경로 큰 회전 일치도는 조합 간 차이 없음 (68 % 안팎) — 원인 아님.

### 9. stage2 (S1 학습) — 10-13 이후 생략
| 학습 | label→GT / →label / 정보량 |
|---|---|
| gt | 33.7 |
| v218fix | 19.6 / 31.6 / 24.5 |
| v278fix 와 변형 7 개 | 23.0 ~ 25.8 (차이 압축, stage1 순위와 같음) |
| (relabel 2 epoch) gt · v218 · v218_reproduce · v6 | 28.5 · 16.4 · 20.0 · 8.0 |

## 분석
1. **두 개의 축: 정보량과 문체.** 정보량(GT→label)은 v278 30 -> b0 40.5 -> b1 44.0 으로 해결됐다. 그런데 label→GT 는 37.4 에서 멈춘다. 같은 라벨로 평가하면(label→label) 50 이상 — 모델은 생성 문체에는 잘 맞추지만 사람 문체로 옮겨가지 못한다. 앞으로의 개선은 정보가 아니라 **문체 다양성**에서 나와야 한다.
2. **문체 다양성 시도는 아직 오차 안.** 형제 문체(실험 1·3), knobs, GEPA 모두 +0.3~+1.0 이거나 손해. label→label 이 낮아지는 쪽(실험 3 44.0, 전 42.9)이 label→GT 를 조금 올리는 경향은 있지만 정보량을 잃는다.
3. **정보 추가는 방 순서만 확실히 효과.** 정지 단서·출발 주변 시점·큰 생성 모델은 0 이거나 손해.
4. **지표: CIDEr/BLEU 로 고르면 틀린다.** C(CIDEr 최고)가 SR 은 낮았다. 정보량 관문(학습 없이 평가 1 회)이 학습 결과와 더 맞는다.
5. **nav3d**: 경로를 새로 계획해도(wp) 손해가 거의 없으므로 노은역처럼 사람 경로가 없는 곳에 적용 가능. 단 sg 경로는 자기 라벨과 짝지으면 -6 — 원인은 아직 추정 (생성 라벨이 자기 경로와 너무 정확히 맞아 사람 문장 일반화가 줄어듦).

## 무효 · 주의 run
- **전 첫 run (10-24 node3, `…_20261024_135951`)**: 발산 (step 864·1074 grad 스파이크, 마지막 loss 0.537) — 무효. 유효 = `…_20261026_021626`.
- v6 (relabel): loss 0.589, 라벨 품질 실패.
- 10-01~02 relabel 평가: habitat_sim 미설치 · scene 경로 · `parse_pixel_goal` IndexError 로 실패 후 수정·재실행.
- fewshot-5 · g3temp-070 · len-free · g4temp-100: 평가 yaml 누락으로 10-12/13 재실행.
- GEPA v4 gemma 라벨: 문장 이상 (사실 오류 24 %, 23 % 가 "version" 으로 시작).
- b1 계열 일부 폴더는 옛 이름 (`b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2` = b1).
- 학습 안 하고 정보량만 본 라벨: v278 원본, pathrule-v1, 화살표 overlay, top-down, GEPA v4 Sonnet.

## 다음 실험 (제안)
1. **sg 경로 + wp-b1 라벨** — nav3d 하락이 "자기 경로 라벨" 조합 때문인지 확인.
2. 단일 run 셀 재학습 (sg + 사람, wp + sg-b1) — ep 단위 비교용.
3. 문체: 사람 문장 일부(예: 10~30 %)를 섞어 학습 — 문체 차이가 원인이면 label→GT 가 가장 크게 오를 실험.

## 재현
- 결과 수집: `python3 docs/reports/collect_relabel_results.py docs/reports/relabel_ablation_results.json`
- 셀: `scripts/train_eval/qwenvl_train/relabel_half/` (`make_cells.py`), 라벨 생성: `scripts/dataset_converters/instruction_labeling/`, 데이터: `scripts/dataset_converters/relabel_vlnce/`.
- 자세한 기록: `.claude/memory/` 의 `261014_instruction_labeling_b0_axes_result.md`, `261021_reproduce_relabel_report_result.md`, `261024_gepa_prompt_opt_result.md`, `261027_nav3d_relabel_2x2_result.md`, `261029_nav3d_sg_cause_result.md`.
