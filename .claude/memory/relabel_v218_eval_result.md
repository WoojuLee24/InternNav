# relabel v6 / v218 진단 + evaluator crash 수정 (2026-10-02)

## v6 결과가 무너진 원인 = labeling 품질 (변환 아님)
- 변환 검증: label↔공식본 scene_id/reference_path 10819/10819 일치, 빌더 출력 독립 재매핑 10682 일치 / 0 불일치 / 2 ambiguous.
- v6 train: 고유 문장 17.2%, 같은 문장 최대 510회, "Pass the landmark" placeholder 485건 → loss 0.588 (gt 0.038), SR 8.7.
- train split 이 있는 label 19개 전부 템플릿 생성 (내용어 어휘 9~102 vs GT 2277). md5 동일: v24=v25=v4=v5, v218=auto_v218.
- v218: 고유 96.4% 이지만 내용어 36개(랜드마크 없음). loss @epoch0.5: gt 0.485 / v218 0.524 / v6 0.721.
- GT 와 비슷한 label 은 `val_unseen_gemma` (GT 의역, 내용어 겹침 0.45) 뿐이고 val_unseen 에만 있다.

## evaluator crash 2종 → `parse_pixel_goal` (internnav/habitat_extensions/vln/utils.py)
- 숫자 1개 출력 → `coord[1]` IndexError (v6 학습 모델 eval 3건).
- y=480 출력 → `depth[480]` IndexError (gt 학습 모델 × v218 eval). mini 학습 데이터 전체(216만 goal) 범위 x<=606, y<=455.
- crash 4건 모두 relabel mini 레시피(Qwen 원본에서 VLM 전체 finetune, mini R2R 만, 13810 step)로 System2 를 학습한 모델. 공개/full System2 모델 eval 에선 0건 (rank0 출력 ~10만 개 중 비정상 0). v6 학습 s2 는 GT 문장 eval 에서도 crash. gt×gt(mini) 결과가 없어 모델 vs OOD 문장 원인 분리는 아직 못 함.
- 수정 (방안 B, 사용자 결정): 숫자<2 또는 이미지 밖 좌표 → 무효 출력. clip 하지 않는다 (무효 출력에 점수를 주지 않기 위해).
  무효면 기존 해석불가 출력 경로(action 파싱 -> 없으면 STOP). 에피소드별 횟수를 progress.json `invalid_outputs` 에 기록.
- 숫자 3개 이상("166 2 218")은 기존대로 앞 2개를 쓴다 (무효 처리 안 함).
- 검증: 기존 로그의 모델 출력 137,547개에서 정상 pixel goal 51,283개 전부 기존 파싱과 동일 (diff 0).

## 공개 ckpt
- InternVLA-N1-DualVLN 의 Qwen 가중치 729개 == InternVLA-N1-System2 (torch.equal). stage2 는 VLM freeze.
  → DualVLN 의 stage1(system2) 평가 = 공개 System2 평가와 같다.
- 공개 System2 GT: SR 61.7 (dualvln_stage1_full_gemm config). 공개 DualVLN 은 full GT 결과가 없었다 (6 ep 디버그뿐).

## 평가 config
- `relabel/eval_v218.<parent>.py`: 부모 baseline_ablation config 에서 eval yaml 만 `relabel/vln_r2r_mini_v218.yaml` 로.
  mini/full 평가 데이터 byte 동일, 둘 다 ld30 → 부모 GT 결과와 문장만 다른 비교.

## v6 vs v218 vs GT 분석 (2026-10-05)
- **v218/v6 는 split 마다 생성 방식이 다르다**: train/val_seen 은 템플릿(내용어 어휘 26~51, 랜드마크 없음),
  val_unseen 만 랜드마크가 풍부(v218 509 / v6 377). 문장 내용어 중 GT 에도 있는 비율: v218 train 0.25, val_unseen 0.14.
  → "v218 학습 x v218 평가" 도 학습/평가 분포가 다르다.
- 방향 정합성 (경로 회전 vs 문장 left/right, GT 로 부호 보정; 마지막 회전 일치율):
  GT train 0.80 / val_unseen 0.84, v218 0.88 / 0.94, v6 0.69 / **0.47 (무작위 수준)**.
- 정지 위치: OS-SR 격차 공개 System2 GT 6.5 / v218 11.6, mini stage1 v218xv218 18.3 (52.7 vs 34.4).
- GT 학습 모델을 v218 val_unseen 으로 평가하면 전부 35~39 로 수렴 (공개 System2 61.7 -> 39.4).
- 제안: 좋은 label 의 빠른 판정 = 공개 System2 를 새 label val_unseen 으로 평가 (학습 없이 1.7h). GT 61.7 에 가까울수록 정보 보존.
