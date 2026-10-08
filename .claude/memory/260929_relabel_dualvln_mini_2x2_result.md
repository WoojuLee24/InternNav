# relabel 2x2 — DualVLN stage1->stage2 main recipe on mini(R2R), ld30 (2026-09-29)

## 결론

v6 에 대해 mini R2R 로 stage1 -> stage2 를 학습하고, stage1·stage2 각각을 GT/v6 yaml 로 평가하는
config 8개를 만들었다. 셀끼리는 **학습 data_root(`vln_ce` vs `vln_ce_v6`)와 평가 yaml 한 줄만** 다르다.

## 이전과 달라진 점

| | 이전 | 지금 |
|---|---|---|
| h200 relabel 평가 yaml | `vln_r2r_mini.yaml` 복사 (tilt 30 = **ld60**) | `vln_r2r_mini_ld30.yaml` 복사 (tilt 15 = **ld30**), 파일명 유지 |
| h200 GT 셀 평가 yaml | `None` -> 기본 yaml (ld60) | `relabel_base.eval_yaml("gt")` -> `vln_r2r_mini_ld30.yaml` |
| 5090 | ld60 | **변경 없음** (5090 용 ld30 GT yaml 이 없음. GT/relabel 모두 ld60 으로 서로 일치) |
| dualvln-mini | 없음 | `relabel/dualvln_mini_base.py` + 셀 8개 |

영향: 기존 default_config 기반 relabel config(`gt.py`, `train.*_eval.gt.py`)도 h200 에서는 GT 평가가 ld30 이 된다.

## recipe (부모 대비)

부모: `baseline_ablation/dualvln_stage1_full_cl.py`, `baseline_ablation/dualvln_stage2_full_cl.py`

| | 부모 (full) | dualvln-mini |
|---|---|---|
| data_root | `InternData-N1/vln_ce` | `InternData-N1-v0.5-mini/vln_ce[_<label>]` |
| stage1 datasets | r2r x4 + rxr x4 | **r2r x4** (125cm_0_30, 125cm_0_45, 60cm_15_15, 60cm_30_30) |
| stage2 datasets | r2r/rxr/scalevln 각 x2 %30 | **r2r_125cm_0_30%30, r2r_60cm_15_15%30** |
| eval yaml | `vln_r2r_full_ld30.yaml` | `vln_r2r_mini_ld30.yaml` / `relabel/vln_r2r_mini_<label>.yaml` |
| stage2 초기값 | released / baseline stage1 | **같은 train_label 의 mini stage1** (STAGE1_CKPT env > 최신 glob) |
| 나머지 (lr, epoch, batch, cl, eval GPU 배치) | — | 전부 상속 |

## 셀 (`scripts/train_eval/qwenvl_train/relabel/`)

| 파일 | 역할 |
|---|---|
| `dualvln_mini_s1.train.{gt,v6}_eval.gt.py` | stage1 학습 + GT 자동 평가 |
| `dualvln_mini_s1.train.{gt,v6}_eval.v6.py` | stage1 ckpt 를 v6 yaml 로 평가만 |
| `dualvln_mini_s2.train.{gt,v6}_eval.gt.py` | stage2 학습 + GT 자동 평가 |
| `dualvln_mini_s2.train.{gt,v6}_eval.v6.py` | stage2 ckpt 를 v6 yaml 로 평가만 |

출력: `/home/irteam/data-vol2/checkpoints/relabel/dualvln_mini_s{1,2}.train.<L>_eval.gt[_internvla-n1-system2]_<ts>`
평가 로그: `<ckpt>/logs/relabel_dualvln_mini_s*.train.<L>_eval.<E>*/` (셀별 분리)

## 큐 (chain 하나 = 노드 하나, 순서 필수)

```
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/dualvln_mini_s1.train.v6_eval.gt.py --machine h200
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/dualvln_mini_s1.train.v6_eval.v6.py --machine h200 --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel/dualvln_mini_s1.train.v6_eval.gt_internvla-n1-system2_*/ | head -1)
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/dualvln_mini_s2.train.v6_eval.gt.py --machine h200
python3 scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/dualvln_mini_s2.train.v6_eval.v6.py --machine h200 --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel/dualvln_mini_s2.train.v6_eval.gt_*/ | head -1)
```
GT chain 은 `v6` -> `gt` 로 바꾼 같은 4줄 (`train.gt_eval.{gt,v6}`).

## 검증

| 검증 | 결과 |
|---|---|
| GT ld30 yaml vs relabel yaml (v6/auto_v218/auto_v207) | 주석 제외 `data_path` **1줄만** 다름 |
| 5090 relabel yaml, `.py` eval config 재생성 | 변경 0 |
| stage1 train argv: 부모 vs mini-gt | vln_dataset_use, data_root, run_name 만 다름 |
| stage1 train argv: mini-gt vs mini-v6 | data_root(`vln_ce` -> `vln_ce_v6`), run_name 만 다름 |
| stage2 train argv: 부모 vs mini-gt | system2_ckpt(stage1), vln_dataset_use, data_root, run_name 만 다름 |
| stage2 train argv: mini-gt vs mini-v6 | data_root, run_name 만 다름 |
| eval_cfg 8셀 전 필드 비교 | 셀간 `env.env_settings.config_path` 만 다름. 부모 대비도 full->mini yaml 만 다름 |
| stage1 없을 때 stage2 import | FileNotFoundError (조용히 released 로 떨어지지 않음) |
| mini R2R parquet pose 컬럼 | 125cm_0/30/45deg, 60cm_15/30deg 전부 있음 |
| builder `--self_check` | 통과 |

못 한 것: 실제 학습/평가 스모크 (node3 GPU 는 full stage1 학습 중).
