# stage1 학습 속도 개선 + resume 결과 (2026-08-28) — follow-up 가이드

**한 줄 요약**: dataloader의 dead-load 낭비를 제거해 stage1이 48→**20.0 s/step**(2.4×)이 됐고,
checkpoint-15000에서 warm resume하여 **~8일 후(≈9/5) 완료 예정**이다.

## 무엇이 느렸고 무엇을 고쳤나

| | 내용 |
|---|---|
| 증상 | stage1이 ~48s/step (ETA 26일), GPU util 100%인데 전력 190–340W (= 대기) |
| 원인 | `NavPixelGoalDataset.__getitem__`이 에피소드 0~현재 프레임을 **전부** 로드(프레임당 파일 3개 open + depth 리사이즈)하고 history 8장/현재/미래 traj 외에는 버림 — 후반 샘플은 수백 파일 dead load. Lustre 콜드 리드로 넘어간 시점(step ~400)부터 병목화 |
| 수정 | `internnav/dataset/internvla_n1_lerobot_dataset.py` `__getitem__`에 skip guard 2줄: history도 traj도 아닌 프레임은 열지 않음 (commit `099d9f3`) |
| 검증 | 수정 전/후 같은 seed로 12개 샘플(에피소드 후반 start_frame_id=300 극단 포함) 출력 tensor **bit 동일** + 샘플당 3.4× 고속화. 학습 수학 무변경 |
| 결과 | **20.0 s/step, 665W** (연산 포화 = 단일노드 물리 한계 도달. 더 줄이려면 멀티노드) |

## resume 내역
- 본선 run: `/home/irteam/data-vol2/checkpoints/baseline/dualvln_stage1_internvla-n1-system2_20260820_031307`
- checkpoint-15000(optimizer 포함)에서 warm resume — loss 0.40 연속, grad_norm ~1.5 정상.
  fast-forward는 sampler 스킵으로 29초 만에 끝남 (dataloader 재생 아님)
- 중복/사망 run 정리(이름 변경, 데이터 보존): `duplicate-run_...052742`, `dead-run_...{010835,154630,160947}`
  → stage2의 자동 glob은 이제 `_031307`만 잡음
- **run_queue는 현재 꺼져 있음** (중복 실행 사고의 원인이어서 정리함). wandb는 resume 구간이 새 run으로 기록됨

## Follow-up 명령어 (이 순서대로)

```bash
# 1) 진행 확인 (지금 ~15k/48,974, 20s/step)
tail -n 3 /home/irteam/data-vol2/checkpoints/baseline/dualvln_stage1_internvla-n1-system2_20260820_031307/logs/train.log

# 2) 학습이 죽어 있으면 — 이 한 줄로 최신 checkpoint에서 자동 재개 (몇 번이고 안전)
nohup bash scripts/train/qwenvl_train/dual_full/resume_dualvln_stage1_20260820_031307.sh >> /home/irteam/data-vol2/checkpoints/baseline/dualvln_stage1_internvla-n1-system2_20260820_031307/logs/train.log 2>&1 &

# 3) stage1 완료 후(top-level에 config.json 생기면) 평가: S2 + ShortestPathFollower, R2R val_unseen
#    목표: README "InternVLA-N1 (S2)+SPF" = NE 4.25 / OS 68.3 / SR 60.9 / SPL 55.2
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py --machine h200 --no-train --model-path /home/irteam/data-vol2/checkpoints/baseline/dualvln_stage1_internvla-n1-system2_20260820_031307

# 4) stage2 학습 + 자동 dual 평가 (stage1 dir 자동 탐색; 목표: README DualVLN SR 64.3)
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/baseline/dualvln_stage2_from_stage1.py --machine h200
```

주의:
- 3)을 학습 도중에 실행하면 top-level 모델이 없어 최신 checkpoint-\<N\>이 평가됨 (중간 점검 용도로는 OK)
- stage1을 **runner로 새로 실행하면 안 됨** (새 dir 생성 → resume 안 됨). 재개는 반드시 2)의 스크립트로
- 상세 이력: `dualvln_stage1_stage2_repro_result.md`
