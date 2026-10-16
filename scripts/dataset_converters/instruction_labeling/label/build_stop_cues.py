"""정지 단서 후보(A·P·R) 를 split 전체에 대해 미리 계산해 캐시에 둔다 (축 2 의 모든 선택 방식이 같은 후보를 쓰도록).

trajectory 마다 1회 (형제 episode 는 경로·마지막 프레임이 같다). GT 문장은 읽지 않는다 — 통계 단계에서만 GT 동의율을 잰다.
출력: <out_root>/_stop_cues/<A>+<P>+<R>/<split>.jsonl  ({trajectory_id, scene, episode_ids, A: [...], P: [...], R: [...]})
      <split>_summary.json  (단서 종류별 후보 있음 비율, GT 동의율)

    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/build_stop_cues.py --split train --workers 32
"""

import argparse
import collections
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval'))
from sources import REPO, episode_track, gt_instructions, house_path, items  # noqa: E402
from stop_cues import object_candidates, place_candidates  # noqa: E402

RIGS = ['125cm_0deg', '125cm_30deg']
CFG = {}


def work(item):
    tr = episode_track(item, RIGS[0])
    T = len(tr['action'])
    full = {r: tr['pose_of'](r) for r in RIGS}
    poses = {r: full[r][T - 1] for r in RIGS}
    depths = {r: cv2.imread(str(tr['depth'](T - 1, r)), -1).astype(np.float32) / 1000.0 for r in RIGS}
    hp = house_path(item['ep'])
    a = object_candidates(item['ep'], hp, poses, depths, CFG['A'])
    p = place_candidates(item['ep'], hp, full[RIGS[0]], CFG['P'])
    r = place_candidates(item['ep'], hp, full[RIGS[0]], CFG['R'])
    return {'A': [{k: c[k] for k in ('name', 'dist_m', 'side', 'visible_in')} for c in a],
            'P': [{k: c[k] for k in ('type', 'phrase')} for c in p],
            'R': [{k: c[k] for k in ('type', 'phrase')} for c in r]}


def init(cfg):
    CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', required=True)
    ap.add_argument('--A', default='A-grounded-v1')
    ap.add_argument('--P', default='P-v2')
    ap.add_argument('--R', default='R-v1')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/instruction_labels')
    a = ap.parse_args()
    sc = yaml.safe_load(open(HERE.parent / 'configs' / 'route.yaml'))['stop_cue']
    cfg = {'A': sc[a.A], 'P': sc[a.P], 'R': sc[a.R]}
    out = REPO / a.out_root / '_stop_cues' / f'{a.A}+{a.P}+{a.R}'
    out.mkdir(parents=True, exist_ok=True)
    its = items(a.split)
    by_t = collections.defaultdict(list)
    for it in its:
        by_t[it['ep']['trajectory_id']].append(it)
    reps = [g[0] for _, g in sorted(by_t.items())]
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(cfg,)) as ex:
        res = list(ex.map(work, reps, chunksize=8))
    rows = []
    for rep, r in zip(reps, res):
        tid = rep['ep']['trajectory_id']
        rows.append({'trajectory_id': tid, 'scene': rep['scene'],
                     'episode_ids': sorted(i['ep']['episode_id'] for i in by_t[tid]), **r})
    with open(out / f'{a.split}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 통계 (GT 는 여기서만)
    from check_stop_cues import PLACE_RE, last_sentence, words_of
    gt = gt_instructions(a.split)
    stat = collections.defaultdict(lambda: [0, 0])  # type -> [후보 있는 traj, GT 동의 traj]
    any_cue = 0
    for r in rows:
        lasts = [last_sentence(gt[e]) for e in r['episode_ids']]
        any_cue += bool(r['A'] or r['P'] or r['R'])
        if r['A']:
            c = r['A'][0]
            stat['A'][0] += 1
            stat['A'][1] += any(re.search(rf'\b{w}\b', ls) for ls in lasts for w in words_of(c['name']))
        for kind in ('P', 'R'):
            if r[kind]:
                c = r[kind][0]
                stat[f'{kind}:{c["type"]}'][0] += 1
                stat[f'{kind}:{c["type"]}'][1] += any(
                    re.search(PLACE_RE[c['type']]({'name': c['phrase']}), ls) for ls in lasts)
    n = len(rows)
    summ = {'split': a.split, 'trajectories': n, 'episodes': len(its), 'any_cue': round(any_cue / n, 3),
            'by_type': {k: {'traj': v[0], 'coverage': round(v[0] / n, 3), 'gt_agree': round(v[1] / max(1, v[0]), 3)}
                        for k, v in sorted(stat.items())}}
    (out / f'{a.split}_summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
