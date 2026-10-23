"""03ab — 계단을 따로 다듬는가(split) vs 구별 없이 전부 바닥처럼 다듬는가(none) 비교.

두 방식은 `03_refine_paths.py --stairs_mode {split,none}` 로 **같은 코드 · 같은 경로 · 동시에** 돌린 결과다
(`logs/03_ab_stairs/{split,none}/`). 평가 자는 둘 다 같다(계단 라벨 = 기본 NavScene 라벨, 점 아래 삼각형).

| 지표 | 뜻 |
|---|---|
| 사용 가능 | 관통(navmesh 밖 또는 mesh < 0.02 m) 0 · r_b 미달 0 |
| GT 같은 루트 | Fréchet < 0.8 m (vln_ce) |
| 계단 정면 각도 | 계단 위 점에서 진행 방향과 계단 경사 방향 사이 각. 0° = 계단을 정면으로 오름, 90° = 계단을 옆으로 가로지름 |
| 회전 °/m | 경로 전체 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/03ab_stairs_compare.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))

MODES = ('split', 'none')


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ab_dir', default='scripts/dataset_converters/nav3d/logs/03_ab_stairs')
    return ap.parse_args()


def load(d):
    rows = {}
    for f in sorted((d / 'scenes').glob('*.json')):
        j = json.load(open(f))
        for r in j['rows']:
            if not r.get('found') or r.get('mode') == 'gt':
                continue
            grp = 'noeun' if j['meta']['kind'] == 'noeun' else f"vln_ce_{r['mode']}_{'multifloor' if r['multifloor'] else 'flat'}"
            key = (j['meta']['scene'], r.get('id', r.get('ep')), r.get('mode'))
            rows[key] = dict(r, grp=grp, r_b=j['meta']['r_b'], scene=j['meta']['scene'])
    return rows


def usable(r):
    return r['ref_on_frac'] == 1.0 and r['ref_mesh_clear_min_m'] >= r['r_b']


def main():
    args = parse()
    ab = Path(args.ab_dir)
    R = {m: load(ab / m) for m in MODES}
    keys = sorted(set(R['split']) & set(R['none']))
    groups = sorted({R['split'][k]['grp'] for k in keys})
    summ = {'n paired paths': len(keys)}
    for g in groups:
        K = [k for k in keys if R['split'][k]['grp'] == g]
        d = {'n': len(K)}
        for m in MODES:
            G = [R[m][k] for k in K]
            d[f'{m} usable'] = sum(usable(r) for r in G)
            if 'ref_gt_same_route' in G[0]:
                d[f'{m} GT same route'] = sum(r['ref_gt_same_route'] for r in G)
            d[f'{m} turn deg/m (median)'] = float(np.median([r['ref_turn_per_m_deg'] for r in G]))
            d[f'{m} margin p5 (median)'] = float(np.median([r['ref_margin_p5_m'] for r in G]))
            h = [r.get('ref_stairs_headon_med_deg') for r in G]
            d[f'{m} stairs head-on median deg (median over paths)'] = float(np.median([x for x in h if x is not None])) if any(x is not None for x in h) else None
            h9 = [r.get('ref_stairs_headon_p90_deg') for r in G]
            d[f'{m} stairs head-on p90 deg (median over paths)'] = float(np.median([x for x in h9 if x is not None])) if any(x is not None for x in h9) else None
        P = [(R['split'][k].get('ref_stairs_headon_med_deg'), R['none'][k].get('ref_stairs_headon_med_deg')) for k in K]
        P = [(a, b) for a, b in P if a is not None and b is not None]
        if P:
            diff = np.array([b - a for a, b in P])
            d['paths with stairs (paired)'] = len(P)
            d['none - split head-on (median diff deg)'] = float(np.median(diff))
            d['none worse > 10 deg'] = int((diff > 10).sum())
            d['none better > 10 deg'] = int((diff < -10).sum())
        summ[g] = d
    out = ab / 'compare.json'
    json.dump(summ, open(out, 'w'), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # 그림: 경로별 계단 정면 각도 split(가로) vs none(세로) + 차이가 가장 큰 경로 목록
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for g in groups:
        K = [k for k in keys if R['split'][k]['grp'] == g]
        P = [(R['split'][k].get('ref_stairs_headon_med_deg'), R['none'][k].get('ref_stairs_headon_med_deg')) for k in K]
        P = np.array([(a, b) for a, b in P if a is not None and b is not None])
        if len(P):
            axs[0].scatter(P[:, 0], P[:, 1], s=6, alpha=0.5, label=g)
        T = np.array([(R['split'][k]['ref_turn_per_m_deg'], R['none'][k]['ref_turn_per_m_deg']) for k in K])
        axs[1].scatter(T[:, 0], T[:, 1], s=6, alpha=0.5, label=g)
    for ax, t in zip(axs, ('stairs head-on angle, median per path [deg] (0 = straight up the stairs)', 'turn [deg/m]')):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], 'k:', lw=0.8); ax.set_xlabel('split (stairs refined separately)'); ax.set_ylabel('none (all as floor)')
        ax.set_title(t, fontsize=9); ax.legend(fontsize=7)
    fig.savefig(ab / 'compare_scatter.png', dpi=80, bbox_inches='tight'); plt.close(fig)
    worst = sorted([k for k in keys if R['split'][k].get('ref_stairs_headon_med_deg') is not None
                    and R['none'][k].get('ref_stairs_headon_med_deg') is not None],
                   key=lambda k: -(R['none'][k]['ref_stairs_headon_med_deg'] - R['split'][k]['ref_stairs_headon_med_deg']))[:12]
    json.dump([{'scene': k[0], 'id': k[1], 'mode': k[2],
                'split_headon': R['split'][k]['ref_stairs_headon_med_deg'], 'none_headon': R['none'][k]['ref_stairs_headon_med_deg']}
               for k in worst], open(ab / 'largest_diff.json', 'w'), indent=1)
    print(f'  -> {out}, compare_scatter.png, largest_diff.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
