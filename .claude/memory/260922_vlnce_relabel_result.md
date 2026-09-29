# VLN-CE relabel — 직접 labeling 데이터로 학습/평가 (2026-09-22)

## 결론

`data/vln/mp3d/r2r/v1/` 의 재작성 instruction 15종을 학습·평가에 쓸 수 있게 했다.
**GT(`vln_ce`)는 한 바이트도 안 바뀌었고, 기존 실험 config 55개는 출력이 이전과 완전히 동일하다.**

## 이전과 달라진 점

| | 이전 | 지금 |
|---|---|---|
| 학습 데이터 | `vln_ce` 하나 | `vln_ce` + `vln_ce_<label>` 15개 |
| 평가 데이터 | `vln_r2r_mini_5090.yaml` 하나 | + `relabel/vln_r2r_mini_5090_<label>.yaml` 15개 |
| 실험 선택 | — | `Params.train_label`(학습) + `Params.eval_config_path`(평가) |

기존 파일 수정은 2개뿐 (`default_config.py` +16줄, `runner.py` +7줄). `internnav/` 는 무수정.

**⚠ 브랜치 주의.** 이 작업은 `feature/input_v0.1` 위에서 만들어졌다. 처음 `feature/embaug_v0.1`
위에 짰을 때는 (a) 평가 yaml 교체용 `eval_label` 과 (b) 2x2 평가 로그 분리를 직접 넣었는데,
**input 브랜치에 이미 둘 다 있었다** — `Params.eval_config_path` 와
`run_eval` 의 `logs/<exp_slug>/`. 중복을 걷어내고 그쪽 메커니즘을 쓰도록 재작업했다.
남은 건 `train_label` 하나뿐이다(학습 `data_root` 를 `TRAIN_MACHINE` 이 덮어쓰는 문제는
input 브랜치에도 해법이 없다).

## 커맨드

```bash
# 게이트 검사만 (생성 안 함)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --self_check

# label 15개 데이터셋 + 평가 yaml 생성 (15초, 총 ~88 MB)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels all --emit data,yaml

# 실험 config 생성 (대조군 + label 별 2x2 셀 3개)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels gt,v6,auto_v218,auto_v207 --emit config

# 대조군 / 실험군 / 교차
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/gt.py --machine 5090
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/train.v6_eval.v6.py --machine 5090
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/train.v6_eval.gt.py --machine 5090 --no-train --model-path <ckpt>
```

| argument | 의미 |
|---|---|
| `--labels` | 쉼표 구분 label. `all` = 3개 split 모두 있는 것 전부. `gt` = 대조군 config 만 |
| `--emit` | `data` 데이터셋 / `yaml` 평가 config / `config` 실험 config |
| `--cells` | 실험 config 셀 (`all` 또는 `both,train_only,eval_only`) |
| `--label_dir` | 직접 labeling json.gz 위치 (기본 `data/vln/mp3d/r2r/v1`) |
| `--gt_root` | GT 루트, 읽기 전용 (기본 `data/InternData-N1-v0.5-mini/vln_ce`) |
| `--rebuild_map` | episode 매핑 캐시 무시하고 재생성 |
| `--self_check` | 생성 없이 게이트만 |
| `--machine 5090` | runner 의 학습/평가 인프라 프리셋 (경로·batch_size·GPU 수) |
| `--no-train --model-path` | 학습 건너뛰고 기존 ckpt 를 평가만 (교차 셀용) |

## 구조

```
data/InternData-N1-v0.5-mini/
├── vln_ce/                   ← GT, 읽기 전용
└── vln_ce_<label>/           ← 15개. GT 와 구조 1:1 대응, label 당 ~6 MB
    ├── raw_data/r2r/{train,val_seen,val_unseen}/{split}.json.gz
    └── traj_data/r2r/<scene>/   61 scene
        ├── data / videos        -> GT symlink (같은 inode)
        └── meta/
            ├── info.json, episodes_stats.jsonl -> GT symlink
            └── episodes.jsonl, tasks.jsonl     ★ 새 instruction
```

사용 가능한 label 15개:
`auto_v207 auto_v208 auto_v209 auto_v210 auto_v211 auto_v212 auto_v213 auto_v217 auto_v218 v218 v24 v25 v4 v5 v6`
제외 3개 (`auto_v206f`, `gate3style_v5`, `v219`) — `val_unseen_<label>.json.gz` 없음. 빌더가 hard fail 한다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| 기존 실험 config 55개 `--print-train-argv` 변경 전/후 | **diff 0** (git stash 로 대조) |
| `default_config.py` vs `relabel/gt.py` 의 EvalCfg | **완전일치** |
| `relabel/gt.py` vs `train.v6_eval.gt.py` 학습 argv | `data_root`, `run_name` 만 다름 |
| GT 트리 수정 | **0건** (mtime 검사) |
| 깨진 symlink | **0개** |
| episode 매핑 (10,684건) | unmatched **0**, ambiguous 2 (결정적 배정) |
| 매핑을 원본 `episode_id` 로 되짚기 | 10,684건 **불일치 0** |
| 학습 로더 (`get_annotations_from_lerobot_data`) 5개 label | id/action/pixel_goal/length **전부 동일**, 문장만 10,684건 다름 |
| 이미지·parquet 경로 (symlink 통과) | 540/540 열림, GT 와 **같은 inode** |
| 평가 json.gz 15개 label | episode_id 집합·`reference_path`·`goals` GT 와 동일, 문장만 1,839건 다름 |
| 문장 끝 | 전부 `". "` (GT 와 같은 꼴) |
| 2x2 로그 분리 | 5개 셀이 `logs/relabel_<EXP_NAME>/` 로 전부 갈림 (input 의 `exp_slug`) |

## 구현 중 발견한 것 3가지

**1. lerobot `episode_index` 는 R2R `episode_id` 와 순서가 다르다.**
파일 순서로 매칭하면 75개 중 2개만 맞는다. scene 안에서 GT 문장 완전일치로 매칭해야 한다
(10,684건 unmatched 0). 기준은 **공식** `vln_ce/raw_data/r2r/train/train.json.gz` —
`data/vln/.../train/train.json.gz` 는 공식본과 **601개 문장이 달라서** 기준으로 못 쓴다.

**2. evaluator 가 문장 마지막 글자를 잘라낸다.**
`habitat_vln_evaluator.py:614` 의 `instruction_text[:-1]`.
GT 는 `". "`(마침표+공백)라 마침표가 남지만 variant 원본은 `"."` 라 **마침표가 사라진다**.
그대로 두면 labeling 이 아니라 구두점 차이가 섞인 실험이 된다.
→ `strip()` 후 구두점 없으면 `.` 붙이고 뒤에 공백. 이 규칙은 GT 에 identity 다
(val_unseen 0/1839, val_seen 0/778; train 은 8/10819 — 구두점 없이 끝나는 원본 예외).

**3. variant 파일 45개 중 7개는 `instruction_vocab` 이 빈 dict 다.**
`{'v24','v25','v4','v5','v6'}×train`, `v6×{val_seen,val_unseen}`.
habitat `VLNDatasetV1.from_json` 이 `deserialized["instruction_vocab"]["word_list"]` 를
무조건 읽으므로 KeyError 로 죽는다. **키 존재 여부가 아니라 `word_list` 유효성**으로
판단해 공식본에서 주입해야 한다 — 처음엔 `"instruction_vocab" not in data` 로 짰다가
`v6` 평가 검증에서 잡혔다.

## 이 환경에서 못 돌린 것 (코드 문제 아님)

- **학습 스모크**: `checkpoints/` 가 비어 있어 베이스 모델(`InternVLA-N1-System2`)이 없다.
- **habitat 평가 스모크**: 두 파이썬 모두 `habitat_sim` 미설치.
  대신 `VLNDatasetV1.from_json` 이 하는 일(vocab·에피소드 키·문장)을 그대로 재현해 검증했다.

체크포인트와 habitat 이 있는 머신에서 남은 확인:

```bash
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/train.v6_eval.v6.py --machine 5090 --max-steps 2 --no-eval
grep -m5 "episode start" <ckpt>/logs_v6/test_5090.log   # variant 문장이 나와야 함
wc -l <ckpt>/logs/progress.json <ckpt>/logs_v6/progress.json   # 각각 1839줄
```

## 관련 파일

- 빌더/문서: `scripts/dataset_converters/relabel_vlnce/{build_relabel_dataset.py,README.md}`
- 매핑·리포트: `scripts/dataset_converters/relabel_vlnce/logs/{train_episode_map.json,<label>.md}`
- 실험 config: `scripts/train_eval/qwenvl_train/relabel/` (`relabel_base.eval_yaml()` 헬퍼 포함)
- 평가 config: `scripts/eval/configs/relabel/`
- 플랜: `/root/.claude/plans/vln-ce-training-test-eager-mccarthy.md`
