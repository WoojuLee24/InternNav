"""noeun 64 · 학습 데이터 — 62_label.py 라벨로 S2 학습용 vln_ce 데이터셋을 만든다 (relabel_half_noeun-v01).

runner 의 train_label 규칙(`<mini>/vln_ce` -> `<mini>/vln_ce_<label>`)과 로더 설정 `r2r_60cm_15_15`(traj_data/r2r, 60cm_15deg)를 그대로 쓴다.
  원본   : <mini>/vln_ce_noeun_src/<ds>/traj_data/<r2r|r2r_val_seen>/noeunstation/  (nav3d/noeun 의 vln_ce 를 복사; 노드로 rsync 할 실물)
  학습   : <mini>/vln_ce_noeun-<paths>-<label>/traj_data/r2r/<ds>_s<k>/
  val    : <mini>/vln_ce_noeun-<paths>-<label>-val/traj_data/r2r/<ds>_s<k>/
scene 폴더 = 경로 종류(ds) x 형제(k = 0, 1, 2): data · videos 는 원본으로 상대 링크, meta/episodes.jsonl 의 tasks = 형제 k 문장.
같은 경로가 형제 3 개 문장으로 3 번 들어간다 (vln_pe 의 tasks 3 개와 같은 정보량).
paths: v6 | hunav (= hunav_v1) | both (= v6 + hunav_v1),  label: b1 | post (62_label.py 라벨)

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/64_build_train.py
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('build_labeled', HERE / '63_build_labeled.py')
BL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BL)            # LABELS · texts · read_jsonl · write_jsonl 재사용

PATHS = {'v6': ['v6'], 'hunav': ['hunav_v1'], 'both': ['v6', 'hunav_v1']}
SCENE = {'v6': 'v6', 'hunav_v1': 'hunav'}
SPLITS = {'train': ('r2r', ''), 'val_seen': ('r2r_val_seen', '-val')}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--noeun_root', default='/home/irteam/data-vol1/nav3d/noeun', help='vln_ce_noeun_nav3d_rbq_<ds> 가 있는 곳')
    ap.add_argument('--labels_root', default='/home/irteam/data-vol1/nav3d/noeun/instruction_labels')
    ap.add_argument('--mini', default='/home/irteam/data-vol1/InternData-N1-v0.5-mini')
    ap.add_argument('--paths', default='v6,hunav,both')
    ap.add_argument('--labels', default='b1,post')
    ap.add_argument('--n_sib', type=int, default=3)
    return ap.parse_args()


def copy_src(a):
    """nav3d/noeun 의 vln_ce (train + val_seen) -> <mini>/vln_ce_noeun_src/<ds>/ (rsync, 이미 같으면 그대로)."""
    for ds in ('v6', 'hunav_v1'):
        src = Path(a.noeun_root) / f'vln_ce_noeun_nav3d_rbq_{ds}' / 'traj_data'
        dst = Path(a.mini) / 'vln_ce_noeun_src' / ds / 'traj_data'
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', f'{src}/', f'{dst}/'], check=True)
        print(f'src {ds}: {dst}')


def build(a, paths: str, label: str, split: str):
    tdir, suf = SPLITS[split]
    out = Path(a.mini) / f'vln_ce_noeun-{paths}-{label}{suf}' / 'traj_data' / 'r2r'
    n_ep = 0
    for ds in PATHS[paths]:
        T = BL.texts(Path(a.labels_root), BL.LABELS[label], ds, split)
        src = Path(a.mini) / 'vln_ce_noeun_src' / ds / 'traj_data' / tdir / 'noeunstation'
        eps = BL.read_jsonl(src / 'meta' / 'episodes.jsonl')
        assert len(eps) == len(T) and all(len(v) >= a.n_sib for v in T.values()), f'{ds}/{split}: 에피소드 {len(eps)} · 라벨 경로 {len(T)}'
        for k in range(a.n_sib):
            sc = out / f'{SCENE[ds]}_s{k}'
            (sc / 'meta').mkdir(parents=True, exist_ok=True)
            for sub in ('data', 'videos'):
                if not (sc / sub).is_symlink():
                    os.symlink(os.path.relpath(src / sub, sc), sc / sub)
            for f in (src / 'meta').iterdir():
                if f.name not in ('episodes.jsonl', 'tasks.jsonl'):
                    shutil.copy2(f, sc / 'meta' / f.name)
            rows = [{'episode_index': e['episode_index'], 'tasks': [T[e['episode_index']][k].strip()], 'length': e['length']} for e in eps]
            BL.write_jsonl(sc / 'meta' / 'episodes.jsonl', rows)
            BL.write_jsonl(sc / 'meta' / 'tasks.jsonl', [{'task_index': r['episode_index'], 'task': r['tasks'][0]} for r in rows])
            n_ep += len(rows)
    src_info = {'paths': paths, 'label': label, 'label_name': BL.LABELS[label], 'split': split, 'n_sib': a.n_sib,
                'datasets': PATHS[paths], 'episodes': n_ep, 'loader': 'r2r_60cm_15_15'}
    (out.parent.parent / 'noeun_train_source.json').write_text(json.dumps(src_info, ensure_ascii=False, indent=1))
    print(f'{out.parent.parent.name}: scenes {len(PATHS[paths]) * a.n_sib} episodes {n_ep}')


def main():
    a = parse()
    copy_src(a)
    for paths in a.paths.split(','):
        for label in a.labels.split(','):
            for split in SPLITS:
                build(a, paths, label, split)


if __name__ == '__main__':
    main()
