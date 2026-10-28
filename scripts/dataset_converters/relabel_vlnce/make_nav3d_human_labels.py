"""사람(R2R GT) 문장을 nav3d 라벨 폴더 형식으로 — build_nav3d_relabel.py 가 nav3d 경로 렌더에 사람 문장을 붙이게 한다.

용도: "경로 모양만 바꾸면 얼마나 떨어지나" 실험 (n3sg-gt = nav3d_sg 경로 + 사람 문장). 라벨 문장 대신 사람 문장이 들어갈 뿐,
build_nav3d_relabel.py 는 그대로 쓴다 (meta/ep_*.json 의 scene · episode_index · episode_id · name · gen_hash · generated.text 만 읽는다).
  train: 릴리스 vln_ce/traj_data/r2r/<scene>/meta/episodes.jsonl 의 tasks[0] -> (scene, 릴리스 episode_index)
  val  : 원본 R2R val_unseen.json.gz 의 instruction_text -> episode_id
nav3d 라벨(label_paths.py)도 같은 (scene, 릴리스 episode_index) 를 쓰므로 nav3d_sg · nav3d_wp 렌더 어느 쪽에도 붙는다.

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/relabel_vlnce/make_nav3d_human_labels.py
출력: <out>/train/meta/ep_*.json, <out>/val_unseen/meta/ep_*.json
"""

import argparse
import gzip
import json
import os

D = "data/InternData-N1-v0.5-mini"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=f"{D}/vln_ce/traj_data/r2r")
    ap.add_argument("--val_json", default=f"{D}/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz")
    ap.add_argument("--out", default=f"{D}/instruction_labels_nav3d/gt-human")
    return ap.parse_args()


def write(md, eid, scene, idx, text):
    m = {"episode_id": eid, "scene": scene, "episode_index": idx, "gen_hash": "human", "name": "gt-human", "generated": {"text": text}}
    with open(os.path.join(md, f"ep_{eid:06d}.json"), "w") as f:
        json.dump(m, f, ensure_ascii=False)


def main():
    a = parse()
    md = os.path.join(a.out, "train", "meta")
    os.makedirs(md, exist_ok=True)
    n = 0
    for scene in sorted(os.listdir(a.release)):
        p = os.path.join(a.release, scene, "meta", "episodes.jsonl")
        if not os.path.isfile(p):
            continue
        for line in open(p):
            r = json.loads(line)
            n += 1
            write(md, n, scene, int(r["episode_index"]), r["tasks"][0])   # train 은 build_nav3d_relabel 이 (scene, episode_index) 로만 찾는다
    md_v = os.path.join(a.out, "val_unseen", "meta")
    os.makedirs(md_v, exist_ok=True)
    eps = json.load(gzip.open(a.val_json))["episodes"]
    for e in eps:
        write(md_v, int(e["episode_id"]), e["scene_id"].split("/")[-2], -1, e["instruction"]["instruction_text"])
    print(f"train {n} · val_unseen {len(eps)} -> {a.out}")


if __name__ == "__main__":
    main()
