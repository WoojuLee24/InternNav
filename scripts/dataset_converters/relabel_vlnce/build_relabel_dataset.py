"""직접 labeling 한 R2R instruction 으로 VLN-CE 학습/평가 데이터셋을 만든다.

무엇을 만드나
-------------
GT 는 한 바이트도 건드리지 않고, 옆에 같은 구조의 형제 루트를 만든다::

    data/InternData-N1-v0.5-mini/
    ├── vln_ce/                  ← GT (읽기 전용)
    └── vln_ce_<label>/          ← 이 스크립트가 만드는 것
        ├── raw_data/r2r/{train,val_seen,val_unseen}/{split}.json.gz   (평가용)
        └── traj_data/r2r/<scene>/
            ├── data   -> symlink to GT
            ├── videos -> symlink to GT
            └── meta/
                ├── info.json            -> symlink to GT
                ├── episodes_stats.jsonl -> symlink to GT
                ├── episodes.jsonl       ★ 새 instruction (학습이 읽는 유일한 파일)
                └── tasks.jsonl          ★ 새 instruction

왜 symlink 인가
---------------
`vln_ce/traj_data` 는 346 GB / 800만 파일이다. 새 labeling 은 문장만 바꾸므로
이미지·parquet 은 GT 를 그대로 가리키면 된다. 학습 로더
(`internnav/dataset/internvla_n1_lerobot_dataset.py:805`)가 instruction 을 읽는 곳은
`<scene>/meta/episodes.jsonl` 의 `tasks[0]` 단 하나다 — `tasks.jsonl` 도, parquet 의
`task_index` 도 읽지 않는다. 그래도 `tasks.jsonl` 을 같이 다시 쓰는 이유는,
GT `tasks.jsonl` 옆에 새 `episodes.jsonl` 이 놓이면 다음 사람이 오해하기 때문이다.

episode 매핑
------------
lerobot `episode_index` 는 R2R `episode_id` 와 **파일 순서가 다르다**.
scene 안에서 GT 문장 완전일치로 매칭한다 (실측: 10,684개 중 unmatched 0, ambiguous 2).
기준 파일은 **공식** `vln_ce/raw_data/r2r/train/train.json.gz` 다 —
`data/vln/.../train/train.json.gz` 는 공식본과 601개 문장이 달라서 기준으로 못 쓴다.

문장 정규화
-----------
evaluator 가 `instruction_text[:-1]` 로 마지막 글자를 잘라낸다
(`internnav/habitat_extensions/vln/habitat_vln_evaluator.py:614`).
GT 는 ``". "``(마침표+공백)로 끝나 마침표가 남지만 variant 는 ``"."`` 로 끝나
마침표가 사라진다. 그래서 variant 문장을 GT 와 같은 ``". "`` 꼴로 맞춘다.
이 규칙은 GT 에 대해 identity 다 (`--self_check` 가 매번 확인한다).

커맨드
------
    /usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels all --emit data,yaml
    /usr/bin/python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels v6 --self_check
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOG_DIR = os.path.join(HERE, "logs")

SPLITS = ("train", "val_seen", "val_unseen")
TRAIN_SPLIT = "train"
GT_LABEL = "gt"  # default_config.Params 의 기본값과 같은 문자열

# 평가 yaml 원본 -> 복사본 stem. 복사본 이름 규칙은 relabel_base.BASE_YAML 과 맞춰야 한다.
# h200 은 ld30(tilt_angle 15 x 2 = 30도, main parity) 이 정본이라 vln_r2r_mini_ld30.yaml 을 원본으로
# 쓰되 복사본 이름은 기존 그대로(vln_r2r_mini_<label>.yaml) 둔다. 짝이 되는 GT 평가 yaml 은
# relabel_base.GT_YAML["h200"] (= vln_r2r_mini_ld30.yaml) 이라 두 셀이 data_path 한 줄만 다르다.
EVAL_YAMLS = (
    ("scripts/eval/configs/vln_r2r_mini_5090.yaml", "vln_r2r_mini_5090"),
    ("scripts/eval/configs/vln_r2r_mini_ld30.yaml", "vln_r2r_mini"),
)
EVAL_PY_CONFIGS = (
    "scripts/eval/configs/habitat_dual_system_mini_5090_cfg.py",
    "scripts/eval/configs/habitat_dual_system_mini_h200_cfg.py",
)
RELABEL_YAML_DIR = "scripts/eval/configs/relabel"
EXP_CONFIG_DIR = "scripts/train_eval/qwenvl_train/relabel"


# --------------------------------------------------------------------------- #
# 문장 정규화
# --------------------------------------------------------------------------- #
def normalize(text):
    """GT 와 같은 꼴(`"... 문장. "`)로 맞춘다. GT 에 대해서는 identity."""
    t = text.strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t + " "


# --------------------------------------------------------------------------- #
# 입출력 헬퍼
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def dump_gz(path, obj):
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def scan_scenes(gt_root):
    traj = os.path.join(gt_root, "traj_data", "r2r")
    return sorted(d for d in os.listdir(traj) if os.path.isdir(os.path.join(traj, d)))


def scan_labels(label_dir):
    """3개 split 에 모두 존재하는 label 만 유효하다."""
    per_split = {}
    for split in SPLITS:
        found = set()
        for f in glob.glob(os.path.join(label_dir, split, f"{split}_*.json.gz")):
            found.add(os.path.basename(f)[len(split) + 1: -len(".json.gz")])
        per_split[split] = found
    valid = set.intersection(*per_split.values())
    return sorted(valid), per_split


def scan_id(scene_id):
    """'mp3d/17DRP5sb8fy/17DRP5sb8fy.glb' -> '17DRP5sb8fy'"""
    return scene_id.split("/")[1]


def relative_symlink(src, dst):
    """dst 를 src 로 가리키는 상대경로 symlink 로 만든다 (이미 있으면 교체)."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst, ignore_errors=True)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


# --------------------------------------------------------------------------- #
# 1. episode 매핑
# --------------------------------------------------------------------------- #
MAP_PATH = os.path.join(LOG_DIR, "train_episode_map.json")


def build_episode_map(gt_root, force=False):
    """(scene, episode_index) -> R2R episode_id.

    scene 안에서 GT `meta/episodes.jsonl` 의 문장과 공식 train.json.gz 의
    `instruction_text` 를 **원문 그대로** 완전일치시킨다. 같은 scene 에 같은 문장이
    여러 개면(ambiguous) episode_index 오름차순으로 미사용 후보를 결정적으로 배정한다.
    """
    if os.path.isfile(MAP_PATH) and not force:
        with open(MAP_PATH) as f:
            cached = json.load(f)
        return cached["map"], cached["stats"]

    raw = load_gz(os.path.join(gt_root, "raw_data", "r2r", TRAIN_SPLIT, f"{TRAIN_SPLIT}.json.gz"))
    by_scene = {}
    for ep in raw["episodes"]:
        by_scene.setdefault(scan_id(ep["scene_id"]), []).append(ep)

    mapping = {}
    stats = {"total": 0, "unmatched": 0, "ambiguous": 0, "ambiguous_cases": []}
    for scene in scan_scenes(gt_root):
        candidates = {}
        for ep in by_scene.get(scene, []):
            candidates.setdefault(ep["instruction"]["instruction_text"], []).append(ep["episode_id"])
        used = set()
        scene_map = {}
        for row in read_jsonl(os.path.join(gt_root, "traj_data", "r2r", scene, "meta", "episodes.jsonl")):
            stats["total"] += 1
            idx = row["episode_index"]
            cand = candidates.get(row["tasks"][0])
            if not cand:
                stats["unmatched"] += 1
                continue
            if len(cand) > 1:
                stats["ambiguous"] += 1
                stats["ambiguous_cases"].append({"scene": scene, "episode_index": idx, "candidates": cand})
            pick = next((c for c in cand if c not in used), cand[0])
            used.add(pick)
            scene_map[str(idx)] = pick
        mapping[scene] = scene_map

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(MAP_PATH, "w") as f:
        json.dump({"map": mapping, "stats": stats}, f, indent=1)
    return mapping, stats


# --------------------------------------------------------------------------- #
# 2. 데이터셋 생성
# --------------------------------------------------------------------------- #
def out_root_for(gt_root, label):
    """`.../vln_ce` + 'v6' -> `.../vln_ce_v6`.  default_config.labeled_data_root() 와 같은 규칙."""
    return gt_root.rstrip("/") + "_" + label


def emit_data(gt_root, label_dir, label, mapping):
    out_root = out_root_for(gt_root, label)
    gt_traj = os.path.join(gt_root, "traj_data", "r2r")
    out_traj = os.path.join(out_root, "traj_data", "r2r")

    # --- raw_data: variant json.gz 를 정규화 + vocab 보정해 복사 (평가용) ---
    vocab_injected = []
    for split in SPLITS:
        src = os.path.join(label_dir, split, f"{split}_{label}.json.gz")
        dst_dir = os.path.join(out_root, "raw_data", "r2r", split)
        os.makedirs(dst_dir, exist_ok=True)
        data = load_gz(src)
        for ep in data["episodes"]:
            ep["instruction"]["instruction_text"] = normalize(ep["instruction"]["instruction_text"])
        # habitat VLNDatasetV1.from_json 은 `deserialized["instruction_vocab"]["word_list"]` 를
        # 무조건 읽는다. 키가 없는 파일뿐 아니라 `instruction_vocab: {}` 인 파일도 있어서
        # (실측: 45개 중 7개) 존재 여부가 아니라 word_list 가 쓸 수 있는지로 판단한다.
        if not (data.get("instruction_vocab") or {}).get("word_list"):
            official = load_gz(os.path.join(gt_root, "raw_data", "r2r", split, f"{split}.json.gz"))
            if not (official.get("instruction_vocab") or {}).get("word_list"):
                official = load_gz(os.path.join(gt_root, "raw_data", "r2r", TRAIN_SPLIT, f"{TRAIN_SPLIT}.json.gz"))
            data["instruction_vocab"] = official["instruction_vocab"]
            vocab_injected.append(split)
        dump_gz(os.path.join(dst_dir, f"{split}.json.gz"), data)

    # --- traj_data: symlink farm + 새 meta (학습용, train split) ---
    variant = load_gz(os.path.join(label_dir, TRAIN_SPLIT, f"{TRAIN_SPLIT}_{label}.json.gz"))
    text_by_id = {ep["episode_id"]: ep["instruction"]["instruction_text"] for ep in variant["episodes"]}

    n_scene = n_ep = n_changed = n_fallback = 0
    for scene in scan_scenes(gt_root):
        gt_scene, out_scene = os.path.join(gt_traj, scene), os.path.join(out_traj, scene)
        os.makedirs(os.path.join(out_scene, "meta"), exist_ok=True)
        for name in ("data", "videos"):
            relative_symlink(os.path.join(gt_scene, name), os.path.join(out_scene, name))
        for name in ("info.json", "episodes_stats.jsonl"):
            relative_symlink(os.path.join(gt_scene, "meta", name), os.path.join(out_scene, "meta", name))

        episodes, tasks = [], []
        for row in read_jsonl(os.path.join(gt_scene, "meta", "episodes.jsonl")):
            idx = row["episode_index"]
            ep_id = mapping[scene].get(str(idx))
            new = text_by_id.get(ep_id) if ep_id is not None else None
            if new is None:  # 매핑 실패 -> GT 문장 유지 (조용히 빈 문장을 넣지 않는다)
                text, n_fallback = row["tasks"][0], n_fallback + 1
            else:
                text = normalize(new)
            n_ep += 1
            n_changed += text != row["tasks"][0]
            episodes.append({"episode_index": idx, "tasks": [text], "length": row["length"]})
            tasks.append({"task_index": idx, "task": text})
        write_jsonl(os.path.join(out_scene, "meta", "episodes.jsonl"), episodes)
        write_jsonl(os.path.join(out_scene, "meta", "tasks.jsonl"), tasks)
        n_scene += 1

    return {
        "out_root": out_root, "scenes": n_scene, "episodes": n_ep,
        "instruction_changed": n_changed, "map_fallback_to_gt": n_fallback,
        "vocab_injected": vocab_injected,
    }


# --------------------------------------------------------------------------- #
# 3. 평가 yaml / eval.py용 config 생성
# --------------------------------------------------------------------------- #
def emit_yaml(gt_root, label):
    """원본 yaml 을 통째로 복사하고 habitat.dataset.data_path 한 줄만 바꾼다.

    habitat Hydra 는 yaml 자신의 디렉토리를 primary search dir 로 쓰므로
    (`habitat/config/default.py:131-134`) 하위 폴더에서 기존 yaml 을
    `defaults:` 로 include 할 수 없다 -> 자기완결 복사본이어야 한다.
    """
    gt_rel = os.path.relpath(gt_root, REPO_ROOT)
    out_rel = os.path.relpath(out_root_for(gt_root, label), REPO_ROOT)
    written = []
    os.makedirs(os.path.join(REPO_ROOT, RELABEL_YAML_DIR), exist_ok=True)
    for src_rel, stem in EVAL_YAMLS:
        src = os.path.join(REPO_ROOT, src_rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(REPO_ROOT, RELABEL_YAML_DIR, f"{stem}_{label}.yaml")
        lines, hit = [], 0
        for line in open(src).read().splitlines(True):
            if line.lstrip().startswith("data_path:") and gt_rel in line:
                line, hit = line.replace(gt_rel, out_rel), hit + 1
            lines.append(line)
        if hit != 1:
            raise RuntimeError(f"{src_rel}: data_path 치환 {hit}건 (1건이어야 한다)")
        with open(dst, "w") as f:
            f.write(f"# 생성물 — build_relabel_dataset.py --labels {label} --emit yaml\n"
                    f"# 원본: {src_rel} (habitat.dataset.data_path 한 줄만 다름)\n")
            f.writelines(lines)
        written.append(os.path.relpath(dst, REPO_ROOT))

    # runner 를 안 쓰고 scripts/eval/eval.py 만 쓸 때용 config
    for src_rel in EVAL_PY_CONFIGS:
        src = os.path.join(REPO_ROOT, src_rel)
        if not os.path.isfile(src):
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(REPO_ROOT, RELABEL_YAML_DIR, f"{stem}_{label}.py")
        text, hit = "", 0
        for line in open(src).read().splitlines(True):
            if "'config_path'" in line or '"config_path"' in line:
                yaml_stem = os.path.splitext(os.path.basename(line.split(":")[1].strip().strip("',\" ")))[0]
                line, hit = line.replace(
                    f"{yaml_stem}.yaml", f"relabel/{yaml_stem}_{label}.yaml"), hit + 1
            text += line
        if hit != 1:
            raise RuntimeError(f"{src_rel}: config_path 치환 {hit}건 (1건이어야 한다)")
        with open(dst, "w") as f:
            f.write(f'"""생성물 — build_relabel_dataset.py --labels {label} --emit yaml.\n\n'
                    f'원본: {src_rel} (config_path 한 줄만 다름).\n"""\n\n')
            f.write(text)
        written.append(os.path.relpath(dst, REPO_ROOT))
    return written


# --------------------------------------------------------------------------- #
# 4. 실험 config 생성 (2x2 셀)
# --------------------------------------------------------------------------- #
EXP_TEMPLATE = '''"""{title}

생성물 — build_relabel_dataset.py --labels {label} --emit config

    python scripts/train_eval/qwenvl_train/runner.py --config {cfg_rel} --machine 5090
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # relabel_base

import default_config as base  # noqa: E402
import relabel_base  # noqa: E402  (같은 폴더 — eval_yaml 헬퍼)

EXP_NAME = "relabel/{name}"
# 학습 쪽은 train_label 이 data_root 를 `<root>_<label>` 로 바꾼다.
# 평가 쪽은 eval_config_path 에 relabel yaml 을 직접 지정한다 (None = 기존 GT yaml).
# 평가 로그는 runner 가 logs/<EXP_NAME slug>/ 로 이미 나누므로 셀끼리 안 섞인다.
PARAMS = replace(base.PARAMS, train_label="{train_label}", eval_config_path={eval_config_path})
eval_cfg = base.make_eval_cfg(PARAMS)
'''


def emit_configs(label, cells):
    out_dir = os.path.join(REPO_ROOT, EXP_CONFIG_DIR)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    wanted = []
    if label == GT_LABEL:
        wanted = [(GT_LABEL, GT_LABEL, "gt", "대조군 — 학습·평가 모두 기존 GT labeling (기존 동작과 동일)")]
    else:
        pairs = {
            "both": ((label, label), f"train.{label}_eval.{label}", f"학습·평가 모두 새 labeling '{label}'"),
            "train_only": ((label, GT_LABEL), f"train.{label}_eval.gt", f"새 labeling '{label}' 로 학습, 기존 GT 로 평가"),
            "eval_only": ((GT_LABEL, label), f"train.gt_eval.{label}", f"기존 GT 로 학습, 새 labeling '{label}' 로 평가"),
        }
        for cell in (pairs if cells == "all" else cells.split(",")):
            (tl, el), name, title = pairs[cell]
            wanted.append((tl, el, name, title))

    for train_label, eval_label, name, title in wanted:
        path = os.path.join(out_dir, f"{name}.py")
        cfg_rel = os.path.join(EXP_CONFIG_DIR, f"{name}.py")
        # eval_config_path 는 머신별 yaml 을 골라야 하므로 리터럴이 아니라 헬퍼 호출로 적는다.
        # GT 셀도 헬퍼를 탄다: h200 은 relabel 복사본과 짝인 ld30 GT yaml, 5090 은 None(기존 GT yaml).
        eval_expr = f'relabel_base.eval_yaml("{eval_label}")'
        with open(path, "w") as f:
            f.write(EXP_TEMPLATE.format(title=title, label=label, cfg_rel=cfg_rel,
                                        name=name, train_label=train_label,
                                        eval_config_path=eval_expr))
        written.append(cfg_rel)
    return written


# --------------------------------------------------------------------------- #
# 5. 리포트 / self-check
# --------------------------------------------------------------------------- #
def gt_normalize_identity(gt_root):
    """정규화 규칙이 GT 에 identity 인지 — 이게 깨지면 GT 분포를 바꾸게 된다."""
    out = {}
    for split in SPLITS:
        p = os.path.join(gt_root, "raw_data", "r2r", split, f"{split}.json.gz")
        if not os.path.isfile(p):
            continue
        eps = load_gz(p)["episodes"]
        t = [e["instruction"]["instruction_text"] for e in eps]
        out[f"raw_data/{split}"] = (sum(normalize(x) != x for x in t), len(t))
    n = c = 0
    for scene in scan_scenes(gt_root):
        for row in read_jsonl(os.path.join(gt_root, "traj_data", "r2r", scene, "meta", "episodes.jsonl")):
            n += 1
            c += normalize(row["tasks"][0]) != row["tasks"][0]
    out["traj_data/meta/episodes.jsonl"] = (c, n)
    return out


def write_report(gt_root, label_dir, label, info, map_stats):
    os.makedirs(LOG_DIR, exist_ok=True)
    gt_traj = os.path.join(gt_root, "traj_data", "r2r")
    out_traj = os.path.join(out_root_for(gt_root, label), "traj_data", "r2r")
    rows = []
    for scene in scan_scenes(gt_root)[:4]:
        g = read_jsonl(os.path.join(gt_traj, scene, "meta", "episodes.jsonl"))
        r = read_jsonl(os.path.join(out_traj, scene, "meta", "episodes.jsonl"))
        for a, b in list(zip(g, r))[:5]:
            rows.append((scene, a["episode_index"], a["tasks"][0], b["tasks"][0]))

    path = os.path.join(LOG_DIR, f"{label}.md")
    with open(path, "w") as f:
        f.write(f"# relabel `{label}`\n\n")
        f.write(f"- 출력 루트: `{os.path.relpath(info['out_root'], REPO_ROOT)}`\n")
        f.write(f"- scene {info['scenes']} / episode {info['episodes']}\n")
        f.write(f"- 문장이 바뀐 episode: {info['instruction_changed']}\n")
        f.write(f"- 매핑 실패로 GT 문장 유지: {info['map_fallback_to_gt']}\n")
        f.write(f"- instruction_vocab 주입한 split: {info['vocab_injected'] or '없음'}\n")
        f.write(f"- 매핑 통계: {map_stats['total']}개 중 unmatched {map_stats['unmatched']}, "
                f"ambiguous {map_stats['ambiguous']}\n\n")
        if map_stats["ambiguous_cases"]:
            f.write("## ambiguous (같은 scene 에 동일 GT 문장)\n\n")
            f.write("episode_index 오름차순으로 미사용 후보를 결정적 배정한다.\n\n")
            for c in map_stats["ambiguous_cases"]:
                f.write(f"- `{c['scene']}` episode_index={c['episode_index']} 후보 {c['candidates']}\n")
            f.write("\n")
        f.write("## GT vs 새 labeling 문장 샘플\n\n")
        f.write("| scene | ep | GT | " + label + " |\n|---|---|---|---|\n")
        for scene, idx, a, b in rows:
            f.write(f"| {scene} | {idx} | {a.strip()} | {b.strip()} |\n")
    return os.path.relpath(path, REPO_ROOT)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="all",
                    help="쉼표 구분 label 목록. 'all' 이면 3개 split 에 모두 있는 label 전부")
    ap.add_argument("--emit", default="data,yaml",
                    help="data(데이터셋) / yaml(평가 config) / config(실험 config) 쉼표 구분")
    ap.add_argument("--cells", default="all",
                    help="실험 config 셀: all 또는 both,train_only,eval_only 조합")
    ap.add_argument("--label_dir", default="data/vln/mp3d/r2r/v1", help="직접 labeling 한 json.gz 들이 있는 곳")
    ap.add_argument("--gt_root", default="data/InternData-N1-v0.5-mini/vln_ce", help="GT 데이터셋 루트 (읽기 전용)")
    ap.add_argument("--rebuild_map", action="store_true", help="episode 매핑 캐시를 무시하고 다시 만든다")
    ap.add_argument("--self_check", action="store_true", help="데이터 생성 없이 검증만 한다")
    args = ap.parse_args()

    gt_root = os.path.join(REPO_ROOT, args.gt_root) if not os.path.isabs(args.gt_root) else args.gt_root
    label_dir = os.path.join(REPO_ROOT, args.label_dir) if not os.path.isabs(args.label_dir) else args.label_dir
    emit = set(args.emit.split(","))

    valid, per_split = scan_labels(label_dir)
    labels = valid if args.labels == "all" else [x for x in args.labels.split(",") if x]
    bad = [x for x in labels if x != GT_LABEL and x not in valid]
    if bad:
        missing = {x: [s for s in SPLITS if x not in per_split[s]] for x in bad}
        sys.exit(f"[FAIL] 3개 split 이 모두 있지 않은 label: {missing}\n"
                 f"       사용 가능한 label {len(valid)}개: {' '.join(valid)}")

    scenes = scan_scenes(gt_root)
    mapping, map_stats = build_episode_map(gt_root, force=args.rebuild_map)
    print(f"[map] {map_stats['total']}개 매핑 / unmatched {map_stats['unmatched']} "
          f"/ ambiguous {map_stats['ambiguous']}  ({os.path.relpath(MAP_PATH, REPO_ROOT)})")
    if map_stats["unmatched"]:
        sys.exit(f"[FAIL] 매핑 실패 {map_stats['unmatched']}건 — 중단한다")

    if args.self_check:
        print(f"[check] scene {len(scenes)}개")
        print(f"[check] 사용 가능한 label {len(valid)}개: {' '.join(valid)}")
        # traj_data/meta 는 train split 에서 온 것이므로 raw_data/train 과 같은 예외를 허용한다.
        # val_seen / val_unseen 은 평가 입력이므로 0 이 아니면 GT 분포를 바꾸는 것 -> FAIL.
        TRAIN_DERIVED = ("raw_data/train", "traj_data/meta/episodes.jsonl")
        ok = True
        for k, (changed, total) in gt_normalize_identity(gt_root).items():
            if changed == 0:
                flag = "OK"
            elif k in TRAIN_DERIVED:
                flag = "OK (구두점 없이 끝나는 원본 예외)"
            else:
                flag, ok = "FAIL", False
            print(f"[check] 정규화가 GT 에 identity? {k}: {changed}/{total} 변경 [{flag}]")

        # habitat VLNDatasetV1.from_json 은 instruction_vocab["word_list"] 를 무조건 읽는다.
        # 키가 없는 파일과 `instruction_vocab: {}` 인 파일이 둘 다 있으므로 빌더가 주입해야 한다.
        need = [(x, s) for x in labels if x != GT_LABEL for s in SPLITS
                if not (load_gz(os.path.join(label_dir, s, f"{s}_{x}.json.gz"))
                        .get("instruction_vocab") or {}).get("word_list")]
        print(f"[check] word_list 주입이 필요한 variant 파일: {len(need)}건 "
              f"{need if need else ''} (빌더가 공식본에서 주입한다)")
        sys.exit(0 if ok else 1)

    for label in labels:
        if label == GT_LABEL:
            if "config" in emit:
                print(f"[gt] 실험 config: {emit_configs(GT_LABEL, args.cells)}")
            continue
        print(f"\n=== {label} ===")
        if "data" in emit:
            info = emit_data(gt_root, label_dir, label, mapping)
            print(f"[data] {os.path.relpath(info['out_root'], REPO_ROOT)}  "
                  f"scene {info['scenes']} / episode {info['episodes']} / "
                  f"문장변경 {info['instruction_changed']} / GT유지 {info['map_fallback_to_gt']}")
            print(f"[report] {write_report(gt_root, label_dir, label, info, map_stats)}")
        if "yaml" in emit:
            print(f"[yaml] {emit_yaml(gt_root, label)}")
        if "config" in emit:
            print(f"[config] {emit_configs(label, args.cells)}")


if __name__ == "__main__":
    main()
