"""noeun 85 · 84 목표 근처 데이터 점검 그림 — raw 에피소드마다 (1) 위에서 본 목표 근처 (2) 가지 프레임 + 라벨.

(1) top_<ds>_<rid>.png: 정답 경로(회색) · prefix(검정) · 가지 A+ A- B (색) · 목표(빨강 별) · 반경 r 원(점선).
(2) frames_<ds>_<eid>.jpg: 가지 프레임 고른 것 — 위 글자 = 목표까지 m · 전문가 라벨(pixel 초록 원 / STOP / 턴) · 반경 r 이면 'STOP(r)'.
마지막에 가지별 프레임 수 · 목표 거리 분포 · 반경 r STOP 프레임 수를 출력한다.

    /usr/bin/python3 scripts/dataset_converters/nav3d/noeun/85_goal_region_viz.py --root <84 out_root> --raw_json <84 에 준 raw_json> --out <dir> [--r 1.0]
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ACT = {0: 'STOP', 1: 'F', 2: 'L', 3: 'R', -1: '-'}
COL = {'A+': 'tab:blue', 'A-': 'tab:green', 'B': 'tab:orange'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--raw_json', required=True)
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--r', type=float, default=1.0, help='반경 STOP (점검 표시용)')
    ap.add_argument('--n_frames', type=int, default=12)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = {e['episode_id']: e for e in json.load(gzip.open(a.raw_json))['episodes']}
    stat = defaultdict(lambda: defaultdict(int))
    for sc in sorted((Path(a.root) / 'traj_data' / 'r2r').glob('abc_*')):
        by_rid = defaultdict(list)
        for m in (json.loads(l) for l in open(sc / 'meta' / 'episodes.jsonl')):
            by_rid[m['raw_episode_id']].append(m)
        for rid, ms in sorted(by_rid.items()):
            ref_h = raw[rid]['reference_path']
            ref_h = json.loads(ref_h) if isinstance(ref_h, str) else ref_h
            R = np.array([[p[0], -p[2], p[1]] for p in ref_h])
            g = R[-1]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(R[:, 0], R[:, 1], '-', color='0.7', lw=5, label='reference')
            for m in ms:
                eid = m['episode_index']
                t = pq.read_table(sc / 'data' / 'chunk-000' / f'episode_{eid:06d}.parquet').to_pydict()
                W = np.array(t['dagger.world_pose'])
                smp = np.array(t['abc.sample'])
                if m is ms[0]:
                    ax.plot(W[~smp, 0], W[~smp, 1], 'k.', ms=3, label='prefix')
                ax.plot(W[smp, 0], W[smp, 1], '.-', color=COL[m['branch']], ms=4, lw=1, label=m['branch'])
                d = np.array(t['abc.goal_dist_m'])
                dz = np.abs(np.array(t['abc.goal_dz_m']))
                st = stat[m['branch']]
                st['frames'] += int(smp.sum())
                st[f'STOP(r={a.r})'] += int((smp & (d < a.r) & (dz < 1.0)).sum())
                st['band r~r+1'] += int((smp & (d >= a.r) & (d < a.r + 1) & (dz < 1.0)).sum())
                st['expert STOP 0.35'] += int(sum(s for s, k in zip(t['dagger.expert_stop'], smp) if k))
                # 가지 프레임 그림
                idx = np.where(smp)[0]
                pick = idx[np.unique(np.linspace(0, len(idx) - 1, min(a.n_frames, len(idx))).astype(int))]
                tiles = []
                for f in pick:
                    im = Image.open(sc / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{eid:06d}_{f}.jpg').convert('RGB')
                    dr = ImageDraw.Draw(im)
                    L, (u, v) = t[f'relative_goal_frame_id.{a.rig}'][f], t[f'goal.{a.rig}'][f]
                    if d[f] < a.r and dz[f] < 1.0:
                        lab = f'STOP(r)'
                    elif t['dagger.expert_stop'][f]:
                        lab = 'STOP'
                    elif L >= 0:
                        dr.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=4)
                        lab = f'pixel len {L}'
                    else:
                        lab = 'turn ' + ''.join(ACT[x] for x in t['dagger.expert_actions'][f][:6])
                    if a.r <= d[f] < a.r + 1:
                        lab += ' [band]'
                    dr.rectangle([0, 0, 640, 26], fill=(0, 0, 0))
                    dr.text((6, 6), f"{m['branch']} f{f} goal {d[f]:.2f}m view {t['abc.yaw_to_goal_deg'][f]:+.0f}° · {lab} · exec {ACT[t['dagger.exec_action'][f]]}", fill=(255, 255, 0))
                    tiles.append(im.resize((320, 240)))
                cols = 4
                grid = Image.new('RGB', (320 * cols, 240 * ((len(tiles) + cols - 1) // cols)))
                for k, im in enumerate(tiles):
                    grid.paste(im, (320 * (k % cols), 240 * (k // cols)))
                grid.save(out / f'frames_{sc.name}_{eid}.jpg')
            ax.plot(g[0], g[1], 'r*', ms=16)
            ax.add_patch(plt.Circle(g[:2], a.r, fill=False, ls='--', color='r'))
            ax.add_patch(plt.Circle(g[:2], 3.0, fill=False, ls=':', color='0.4'))
            ax.set_xlim(g[0] - 7, g[0] + 7)
            ax.set_ylim(g[1] - 7, g[1] + 7)
            ax.set_aspect('equal')
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f'{sc.name} raw {rid}: ' + ' · '.join(f"{m['branch']} {m['n_branch']}f end {m['end_goal_dist_m']:.2f}m" for m in ms), fontsize=8)
            fig.savefig(out / f'top_{sc.name}_{rid}.png', dpi=90)
            plt.close(fig)
    for br, st in sorted(stat.items()):
        print(br, dict(st))


if __name__ == '__main__':
    main()
