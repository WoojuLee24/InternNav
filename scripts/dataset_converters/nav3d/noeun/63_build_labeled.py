"""noeun 63 · 데이터셋 — 62_label.py 라벨을 노은역 vln_ce · vln_pe 데이터셋에 붙여 새 데이터셋을 만든다 (원본은 그대로).

결과: <data_root>/<원본 이름>_<label>/  (label = b1 | post)
  traj_data/<r2r|r2r_val_seen>/noeunstation/{data,videos} -> 원본 링크, meta/ 는 복사 후 tasks 만 바꿈
  raw_data/r2r/<split>/<split>.json.gz -> instruction 을 바꾼 R2R 형식
문장 넣는 규칙 (경로 하나에 형제 n 개):
  vln_ce: tasks 1 칸 = 형제 0 / vln_pe: tasks 3 칸 = 형제 0, 1, 2 (tasks.jsonl 의 task_index = 경로 x 3 + 형제)
  raw json: vln_ce · vln_pe 둘 다 instruction = 형제 0

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/63_build_labeled.py
"""

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

DS = ['v6', 'hunav_v1']
SPLITS = {'train': 'r2r', 'val_seen': 'r2r_val_seen'}
LABELS = {  # 짧은 이름 -> 62_label.py 라벨 이름
    'b1': 'b1_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v1',
    'post': 'b1_facts-memo-v1_knobs-v2_delivery-joint3-v1_writer-gepa-v3-joint-gemma_model-gemma31b-bf16_rig-60cm_15deg_rooms-noeun-v1',
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default='/home/irteam/data-vol1/nav3d/noeun')
    ap.add_argument('--labels_root', default='/home/irteam/data-vol1/nav3d/noeun/instruction_labels')
    ap.add_argument('--labels', default='b1,post')
    return ap.parse_args()


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def write_jsonl(p, rows):
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def texts(labels_root: Path, name: str, ds: str, split: str) -> dict:
    """{raw episode 번호: [형제 0, 1, 2 문장]}"""
    eps = json.load(gzip.open(labels_root / name / f'{ds}_{split}' / f'{ds}_{split}.json.gz'))['episodes']
    out = {}
    for e in eps:
        out.setdefault(e['raw_episode_id'], {})[e['sib_idx']] = e['instruction']['instruction_text']
    return {i: [s[k] for k in sorted(s)] for i, s in out.items()}


def build(a, short: str, kind: str, ds: str):
    src = Path(a.data_root) / f'{kind}_noeun_nav3d_rbq_{ds}'
    dst = Path(a.data_root) / f'{kind}_noeun_nav3d_rbq_{ds}_{short}'
    for split, tdir in SPLITS.items():
        T = texts(Path(a.labels_root), LABELS[short], ds, split)
        s_sc, d_sc = src / 'traj_data' / tdir / 'noeunstation', dst / 'traj_data' / tdir / 'noeunstation'
        (d_sc / 'meta').mkdir(parents=True, exist_ok=True)
        for sub in ('data', 'videos'):
            link = d_sc / sub
            if not link.is_symlink():
                os.symlink(os.path.relpath(s_sc / sub, d_sc), link)
        for f in (s_sc / 'meta').iterdir():
            if f.name not in ('episodes.jsonl', 'tasks.jsonl'):
                shutil.copy2(f, d_sc / 'meta' / f.name)
        eps, tasks = read_jsonl(s_sc / 'meta' / 'episodes.jsonl'), read_jsonl(s_sc / 'meta' / 'tasks.jsonl')
        assert len(eps) == len(T), f'{src}/{tdir}: 에피소드 {len(eps)} != 라벨 경로 {len(T)}'
        n_task = len(eps[0]['tasks'])
        assert len(tasks) == len(eps) * n_task and all(len(e['tasks']) == n_task for e in eps), f'{src}/{tdir}: tasks 칸 수가 일정하지 않다'
        assert n_task <= len(T[0]), f'tasks {n_task} 칸 > 형제 {len(T[0])} 개'
        for e in eps:
            e['tasks'] = T[e['episode_index']][:n_task]
        for t in tasks:
            i, k = divmod(t['task_index'], n_task)
            t['task'] = T[i][k]
        write_jsonl(d_sc / 'meta' / 'episodes.jsonl', eps)
        write_jsonl(d_sc / 'meta' / 'tasks.jsonl', tasks)
        raw_s, raw_d = src / f'raw_data/r2r/{split}/{split}.json.gz', dst / f'raw_data/r2r/{split}/{split}.json.gz'
        raw_d.parent.mkdir(parents=True, exist_ok=True)
        d = json.load(gzip.open(raw_s))
        for e in d['episodes']:
            e['instruction'] = dict(e['instruction'], instruction_text=T[int(e['episode_id'])][0])
        d['_label'] = {'name': LABELS[short], 'short': short, 'rule': 'instruction = 형제 0'}
        with gzip.open(raw_d, 'wt') as f:
            json.dump(d, f)
        print(f'{dst.name}/{tdir}: ep {len(eps)} tasks/ep {n_task} | {eps[0]["tasks"][0][:90]}')


def main():
    a = parse()
    for short in a.labels.split(','):
        for kind in ('vln_ce', 'vln_pe'):
            for ds in DS:
                build(a, short, kind, ds)


if __name__ == '__main__':
    main()
