# relabel_vlnce — 직접 labeling 한 instruction 으로 VLN-CE 학습/평가

`data/vln/mp3d/r2r/v1/{split}/{split}_<label>.json.gz` 의 재작성 instruction 을
학습·평가에 쓸 수 있게 만든다. **GT(`vln_ce`)는 읽기만 하고 한 바이트도 안 바꾼다.**

## 한 줄 요약

label 하나당 `data/InternData-N1-v0.5-mini/vln_ce_<label>/` 을 만든다.
이미지·parquet 은 GT symlink 라 label 당 ~6 MB 다. 바뀌는 건 문장뿐.

## 커맨드

```bash
# 검증만 (데이터 생성 안 함)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --self_check

# 사용 가능한 label 전부 생성 (15개, ~15초, 총 ~90 MB)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels all --emit data,yaml

# 비교 실험용 config (대조군 gt + label 별 2x2 셀)
/usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels gt,v6,auto_v218 --emit config
```

| 인자 | 의미 |
|---|---|
| `--labels` | 쉼표 구분 label. `all` = 3개 split 에 모두 있는 것 전부. `gt` = 대조군 config 만 |
| `--emit` | `data` 데이터셋 / `yaml` 평가 config / `config` 실험 config |
| `--cells` | 실험 config 셀: `all` 또는 `both,train_only,eval_only` 조합 |
| `--label_dir` | 직접 labeling json.gz 위치 (기본 `data/vln/mp3d/r2r/v1`) |
| `--gt_root` | GT 루트, 읽기 전용 (기본 `data/InternData-N1-v0.5-mini/vln_ce`) |
| `--rebuild_map` | episode 매핑 캐시 무시하고 재생성 |
| `--self_check` | 생성 없이 게이트만 |

사용 가능한 label 15개:
`auto_v207 auto_v208 auto_v209 auto_v210 auto_v211 auto_v212 auto_v213 auto_v217 auto_v218 v218 v24 v25 v4 v5 v6`

제외된 3개 (`auto_v206f`, `gate3style_v5`, `v219`)는 `val_unseen_<label>.json.gz` 가 없다.
빌더는 조용히 GT 로 떨어지지 않고 **hard fail** 한다.

## 실험 실행

```bash
# 대조군 — 학습·평가 모두 기존 GT labeling (기존 동작과 동일)
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/gt.py --machine 5090

# 학습·평가 모두 새 labeling
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/train.v6_eval.v6.py --machine 5090

# 교차 — 새 labeling 으로 학습한 ckpt 를 GT 로 평가
python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/train.v6_eval.gt.py --machine 5090 --no-train --model-path <ckpt>
```

runner 없이 `scripts/eval/eval.py` 만 쓸 때:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/relabel/habitat_dual_system_mini_5090_cfg_v6.py
```

## 만들어지는 것

```
data/InternData-N1-v0.5-mini/
├── vln_ce/                      ← GT (읽기 전용)
└── vln_ce_<label>/              ← GT 와 구조가 1:1 대응
    ├── raw_data/r2r/{train,val_seen,val_unseen}/{split}.json.gz   (평가용)
    └── traj_data/r2r/<scene>/   61 scene
        ├── data   -> GT symlink
        ├── videos -> GT symlink
        └── meta/
            ├── info.json            -> GT symlink
            ├── episodes_stats.jsonl -> GT symlink
            ├── episodes.jsonl       ★ 새 instruction
            └── tasks.jsonl          ★ 새 instruction
```

```
scripts/eval/configs/relabel/
├── vln_r2r_mini_5090_<label>.yaml              habitat 평가 config (원본과 data_path 한 줄만 다름)
├── vln_r2r_mini_<label>.yaml
└── habitat_dual_system_mini_{5090,h200}_cfg_<label>.py   eval.py 직접 실행용

scripts/train_eval/qwenvl_train/relabel/
├── relabel_base.py                  eval_yaml() + 기존 config 를 경로로 읽는 load_exp()
├── gt.py                            대조군
├── train.<L>_eval.<L>.py            학습·평가 모두 새 labeling
├── train.<L>_eval.gt.py             새 labeling 학습 → GT 평가
└── train.gt_eval.<L>.py             GT 학습 → 새 labeling 평가

scripts/dataset_converters/relabel_vlnce/logs/
├── train_episode_map.json           episode 매핑 캐시
└── <label>.md                       매핑 통계 + GT vs 새 문장 샘플
```

## 노브 2개

| | 노브 | 값 |
|---|---|---|
| 학습 | `Params.train_label` | `"gt"`(기본) 또는 label. `data_root` 를 `<root>_<label>` 로 바꾼다 |
| 평가 | `Params.eval_config_path` | `None`(기본, 기존 GT yaml) 또는 `relabel_base.eval_yaml("<label>")` |

둘이 따로라서 2x2(학습 GT/new x 평가 GT/new)가 전부 표현된다.
평가 로그는 `runner.run_eval` 이 `logs/<EXP_NAME slug>/` 로 나누므로 셀끼리 섞이지 않는다
— 이건 relabel 이 추가한 게 아니라 이미 있던 동작이다.

## 알아야 할 것 3가지

**1. 학습이 instruction 을 읽는 곳은 `meta/episodes.jsonl` 하나다.**
`internnav/dataset/internvla_n1_lerobot_dataset.py:805` 의 `ep["tasks"][0]`.
`tasks.jsonl` 도 parquet 의 `task_index` 도 안 읽는다. 그래도 `tasks.jsonl` 을 같이 다시 쓰는 건,
GT `tasks.jsonl` 옆에 새 `episodes.jsonl` 이 놓이면 다음 사람이 오해하기 때문이다.

**2. episode 매핑은 파일 순서가 아니다.**
lerobot `episode_index` 와 R2R `episode_id` 는 순서가 다르다. scene 안에서 GT 문장
완전일치로 매칭한다 — 10,684건 중 unmatched 0, ambiguous 2 (결정적 배정, 리포트에 기록).
기준 파일은 **공식** `vln_ce/raw_data/r2r/train/train.json.gz` 다.
`data/vln/.../train/train.json.gz` 는 공식본과 601개 문장이 달라서 기준으로 못 쓴다.

**3. 문장 끝을 GT 와 같은 `". "` 로 맞춘다.**
evaluator 가 `instruction_text[:-1]` 로 마지막 글자를 잘라낸다
(`internnav/habitat_extensions/vln/habitat_vln_evaluator.py:614`).
GT 는 `". "` 로 끝나 마침표가 남지만 variant 원본은 `"."` 로 끝나 마침표가 사라진다.
이 정규화는 GT 에 대해 identity 다 (`--self_check` 가 매번 확인).

**보너스.** variant 파일 45개 중 7개는 `instruction_vocab` 이 **빈 dict** 다.
habitat `VLNDatasetV1.from_json` 이 `instruction_vocab["word_list"]` 를 무조건 읽으므로
빌더가 공식본에서 주입한다. 키 존재 여부가 아니라 `word_list` 유효성으로 판단해야 한다.
