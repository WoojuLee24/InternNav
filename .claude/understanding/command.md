# runner.py 커맨드

`scripts/train_eval/qwenvl_train/runner.py` — config 하나로 train(torchrun)+eval(eval.py) 구동.
`R=scripts/train_eval/qwenvl_train`로 줄여 표기. (guideline상 `python`=`/workspace/isaaclab/_isaac_sim/python.sh`)

## 4개 축
- **머신**: `--machine h200`(기본,8GPU) | `5090`(단일) | `h1`(Isaac,수동) → train+eval 인프라 프리셋 동시 적용
- **단계**: 기본 train+eval | `--no-eval`(train만) | `--no-train --model-path <ckpt>`(eval만)
- **BEV**: CLI 아님 → **config 선택** (`$R/batch_size2/*`=non-BEV, `$R/bev/*`=BEV)
- **디버깅**: `--in-process`(torchrun 우회) | `--debugpy`(attach 대기, 포트 5678)

기타: `--nproc N`(GPU 수 오버라이드) · `--checkpoints-root <path>` · `--data-root <path>` · `--print-train-argv`(dry run)

### `--machine` 프리셋 내용
| | h200 | 5090 |
|---|---|---|
| data_root | `/home/irteam/git/InternNav/data/.../vln_ce` | `/ws/src/InternNav/data/.../vln_ce` |
| system2_ckpt | `/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2` | `/ws/src/InternNav/checkpoints/InternVLA-N1-System2` |
| nproc_per_node | 8 | 1 |
| checkpoints_root | `/home/irteam/data-vol2/checkpoints` | `/ws/src/InternNav/checkpoints` |

## config
| | non-BEV | BEV |
|---|---|---|
| | `batch_size2/b4_eff128_base.py`(기본) `b2_eff128.py` `b8_eff128.py` | `bev/bev_s1_bev.py`(S1=bev) `bev/bev_s1_fpv_base.py`(baseline) |

## 레시피 (한 줄, repo 루트에서)
```
# H200 풀파이프 (train+eval)
python $R/runner.py --config $R/batch_size2/b4_eff128_base.py
# 5090 풀파이프
python $R/runner.py --config $R/batch_size2/b4_eff128_base.py --machine 5090
# train만 / eval만
python $R/runner.py --config $R/batch_size2/b4_eff128_base.py --no-eval
python $R/runner.py --config $R/batch_size2/b4_eff128_base.py --no-train --model-path <ckpt>
# BEV (5090)
python $R/runner.py --config $R/bev/bev_s1_bev.py --machine 5090
# 학습 인자 확인
python $R/runner.py --config $R/batch_size2/b2_eff128.py --print-train-argv
# 검증
python $R/batch_size2/verify_params.py
python $R/bev/verify_params.py
```

## 디버깅 (5090, 단일 GPU)
```
# train 디버그 — attach 포트 5678
python $R/runner.py --config $R/bev/bev_s1_bev.py --machine 5090 --debugpy --no-eval
```
- **eval 디버그는 attach-by-PID 권장** (OOM-safe): `python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_mini_5090_bev_cfg.py --bev_s1_mode bev --bev_s2_mode fpv` → 로드 후 VSCode "Attach to PID".
- ⚠️ `--debugpy`는 **로드 전** 대기 → eval 모델 디버깅 시 로딩 중 OOM 위험. train/eval은 **한 번에 하나씩**.

## 체크포인트 저장
`{--checkpoints-root}/{EXP_NAME}_{타임스탬프}/checkpoint-<step>/` + `train.log`(eval 시 `test.log`).
기본 루트 `/home/irteam/data-vol2/checkpoints`(H200) → 다른 머신은 `--checkpoints-root`로 변경.
