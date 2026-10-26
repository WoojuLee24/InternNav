"""nav3d 경로(nav3d_sg / nav3d_wp) 렌더 데이터 + 새 라벨 -> relabel_half 2x2 용 학습·평가 데이터.

build_relabel_dataset.py 와 같은 결과 구조 (`<gt_root>_<label>`) 를 만든다. 다른 점은 둘뿐이다:
  - 이미지·행동은 릴리스가 아니라 nav3d 렌더(`--traj_root`, scene 이 한 단계 아래 `nav3d_{sg,wp}/<scene>`)에서 가져온다.
  - 에피소드 대응은 문장 일치가 아니라 라벨 meta 의 (scene, episode_index) 로 바로 한다 (nav3d 라벨 = nav3d/common/label_paths.py 출력).

  train : <gt_root>_<label>/traj_data/r2r/<scene>/  data · videos · meta/info.json · meta/episodes_stats.jsonl 는 nav3d scene 으로 상대 링크,
          meta/episodes.jsonl · tasks.jsonl 은 문장만 라벨로 (episode_index · length 는 nav3d 그대로)
  eval  : <gt_root>_<label>/raw_data/r2r/val_unseen/val_unseen.json.gz = 원본 R2R val_unseen 1,839 에피소드 (출발·도착·경로 그대로) 에
          nav3d val 라벨 문장만 episode_id 로 넣은 것. 평가 yaml 은 build_relabel_dataset.emit_yaml (ld30 복사, data_path 한 줄)
  -> 학습 셀은 default_config.labeled_data_root(base, label) = `vln_ce_<label>` 이라 relabel_half/make_cells.py 셀을 그대로 쓴다.

    /usr/bin/python scripts/dataset_converters/relabel_vlnce/build_nav3d_relabel.py --label n3sg-b1 \
        --train_labels data/InternData-N1-v0.5-mini/instruction_labels_nav3d/<name>/nav3d_sg \
        --val_labels data/InternData-N1-v0.5-mini/instruction_labels_nav3d/<name>/nav3d_sg_val_unseen \
        --traj_root data/InternData-N1-v0.5-mini/vln_ce_nav3d_rig5/traj_data/r2r/nav3d_sg --emit data,yaml
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_relabel_dataset as B  # noqa: E402


def label_texts(label_dir):
    """라벨 폴더 meta/ep_*.json -> {(scene, episode_index): 문장}, 그리고 {episode_id: 문장}. 라벨 이름·gen_hash 가 하나인지 확인."""
    by_idx, by_id, hashes = {}, {}, set()
    for f in sorted(glob.glob(os.path.join(label_dir, "meta", "ep_*.json"))):
        m = json.load(open(f))
        t = m["generated"]["text"]
        assert t.strip(), f"빈 문장: {f}"
        by_idx[(m["scene"], int(m["episode_index"]))] = t
        by_id[int(m["episode_id"])] = t
        hashes.add((m["name"], m["gen_hash"]))
    assert len(hashes) == 1, f"{label_dir}: 라벨 이름·gen_hash 가 섞였다 {hashes}"
    return by_idx, by_id, hashes.pop()


def emit_train(out_root, traj_root, by_idx, index_dir, max_fill):
    """라벨이 없는 에피소드는 같은 경로(nav3d _index 의 trajectory_id)의 다른 형제 라벨로 채운다 — 최대 max_fill 개, 넘으면 멈춘다.
    알려진 경우: jh4fc5c5qoQ 릴리스 ep 82·85 는 사람 문장이 글자까지 같아 label_paths 의 문장 일치(match_raw)가 같은 R2R episode_id 로 보내
    라벨 파일이 하나로 겹친다. nav3d 51 도 같은 이유로 둘을 trajectory 82 의 같은 경로로 렌더했으므로, 형제 라벨이 그 경로를 설명한다."""
    out_traj = os.path.join(out_root, "traj_data", "r2r")
    n_scene = n_ep = 0
    missing, filled = [], []
    tid_of = {}
    for f in glob.glob(os.path.join(index_dir, "*.jsonl")):
        for r in B.read_jsonl(f):
            tid_of[(r["scene"], r["episode_index"])] = r["trajectory_id"]
    sib_text = {}
    for (sc, idx), t in sorted(by_idx.items()):
        if (sc, idx) in tid_of:
            sib_text.setdefault((sc, tid_of[(sc, idx)]), t)
    for scene in sorted(d for d in os.listdir(traj_root) if os.path.isdir(os.path.join(traj_root, d, "meta"))):
        src, dst = os.path.join(traj_root, scene), os.path.join(out_traj, scene)
        os.makedirs(os.path.join(dst, "meta"), exist_ok=True)
        for name in ("data", "videos"):
            B.relative_symlink(os.path.join(src, name), os.path.join(dst, name))
        for name in ("info.json", "episodes_stats.jsonl"):
            B.relative_symlink(os.path.join(src, "meta", name), os.path.join(dst, "meta", name))
        episodes, tasks = [], []
        for row in B.read_jsonl(os.path.join(src, "meta", "episodes.jsonl")):
            idx = row["episode_index"]
            t = by_idx.get((scene, idx))
            if t is None:  # 같은 경로 형제 라벨로 채우거나, 없으면 멈춘다 (사람 문장을 조용히 남기지 않는다)
                t = sib_text.get((scene, tid_of.get((scene, idx))))
                if t is None:
                    missing.append((scene, idx))
                    continue
                filled.append((scene, idx, tid_of[(scene, idx)]))
            text = B.normalize(t)
            episodes.append({"episode_index": idx, "tasks": [text], "length": row["length"]})
            tasks.append({"task_index": idx, "task": text})
            n_ep += 1
        B.write_jsonl(os.path.join(dst, "meta", "episodes.jsonl"), episodes)
        B.write_jsonl(os.path.join(dst, "meta", "tasks.jsonl"), tasks)
        n_scene += 1
    assert not missing, f"라벨 없는 에피소드 {len(missing)}개, 예: {missing[:5]}"
    assert len(filled) <= max_fill, f"형제 라벨로 채운 에피소드 {len(filled)}개 > --max_fill {max_fill}: {filled[:5]}"
    return {"scenes": n_scene, "episodes": n_ep, "filled_from_sibling": filled}


def emit_eval(out_root, gt_root, by_id):
    src = os.path.join(gt_root, "raw_data", "r2r", "val_unseen", "val_unseen.json.gz")
    d = json.load(gzip.open(src))
    eps = d["episodes"]
    missing = [e["episode_id"] for e in eps if e["episode_id"] not in by_id]
    assert not missing, f"val 라벨 없는 에피소드 {len(missing)}개, 예: {missing[:5]}"
    for e in eps:
        e["instruction"] = {"instruction_text": B.normalize(by_id[e["episode_id"]]), "instruction_tokens": []}
    dst = os.path.join(out_root, "raw_data", "r2r", "val_unseen", "val_unseen.json.gz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(dst, "wt") as f:
        json.dump(d, f)
    return {"val_episodes": len(eps), "path": os.path.relpath(dst, B.REPO_ROOT)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="짧은 라벨 이름 ('.' 없이). 결과 = <gt_root>_<label>")
    ap.add_argument("--train_labels", required=True, help="nav3d 라벨 폴더 (meta/ep_*.json)")
    ap.add_argument("--val_labels", required=True, help="nav3d val_unseen 라벨 폴더 (meta/ep_*.json)")
    ap.add_argument("--traj_root", required=True, help="nav3d 렌더 scene 들의 부모 (…/traj_data/r2r/nav3d_sg)")
    ap.add_argument("--gt_root", default="data/InternData-N1-v0.5-mini/vln_ce", help="결과 이름과 평가 원본 json 의 기준")
    ap.add_argument("--emit", default="data,yaml")
    ap.add_argument("--max_fill", type=int, default=1, help="라벨이 없어 같은 경로 형제 라벨로 채울 수 있는 train 에피소드 수 상한 (알려진 경우 1개)")
    a = ap.parse_args()
    assert "." not in a.label, "라벨 이름에 '.' 은 쓸 수 없다 (셀 파일명 규칙)"
    gt_root = os.path.join(B.REPO_ROOT, a.gt_root) if not os.path.isabs(a.gt_root) else a.gt_root
    out_root = B.out_root_for(gt_root, a.label)
    emit = set(a.emit.split(","))
    report = {"label": a.label, "out_root": os.path.relpath(out_root, B.REPO_ROOT)}
    if "data" in emit:
        tr_idx, _, tr_name = label_texts(a.train_labels)
        _, va_id, va_name = label_texts(a.val_labels)
        assert tr_name[0] == va_name[0], f"train·val 라벨 방법이 다르다: {tr_name} vs {va_name}"
        report["source_label"] = {"train": tr_name, "val_unseen": va_name}
        index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.traj_root))), "_index", os.path.basename(os.path.normpath(a.traj_root)))
        report["train"] = emit_train(out_root, a.traj_root, tr_idx, index_dir, a.max_fill)
        report["eval"] = emit_eval(out_root, gt_root, va_id)
        with open(os.path.join(out_root, "nav3d_relabel_source.json"), "w") as f:
            json.dump(dict(report, train_labels=a.train_labels, val_labels=a.val_labels, traj_root=a.traj_root), f, ensure_ascii=False, indent=1)
    if "yaml" in emit:
        report["yaml"] = B.emit_yaml(gt_root, a.label)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
