"""층 이동 경로(03mf) 눈으로 확인 — 에피소드마다 PNG 1장.

    층별 top-down (그 층 통행 가능 지도 위)        옆에서 본 z 프로파일
    회색 = 통행 가능(plan_r_b), 검정 = 벽/불가       노랑 = GT 바닥 z, 초록/빨강 = 우리(평지/계단)
    노랑 점선 = GT, 초록 = 우리 평지(A*), 빨강 = 계단(GT 그대로), 파랑 □ = 물린 입구/출구

사용:
    python checks/03mf_plot_multifloor.py --dataset vln_ce --rig 125cm_30deg --paths_mf <out>/paths_mf/<S>_vlnce.json --esdf_dir <esdf> --episodes 0,5 --out <dir>
    --episodes 생략 시 앞 4개. --failures 를 주면 실패 에피소드(층·GT 만)도 그린다.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GS))
import dataset_utils  # noqa: E402
import multifloor_utils as mfu  # noqa: E402
from esdf_utils import truncate_navigable, compute_scan_coverage_mask  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location('m03', GS / '03_sample_gt_paths.py')
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)


def floor_map(occ, origin, plan_z, h_b, a):
    esdf, _ = mfu.floor_nav_coarse(occ, origin, plan_z, h_b, a)
    return truncate_navigable(esdf, a.plan_r_b) & compute_scan_coverage_mask(occ)


def plot_episode(j, ep_id, occ, origin, a, args, out: Path, entry=None, status=None):
    gt_xyz, h_b, pitch, ep_floor = m03.load_gt_episode(args.data_root, j['scene_id'], ep_id, args.dataset, args.rig)
    gxy = gt_xyz[:, :2]
    gfz = mfu.point_floor_z(gt_xyz, ep_floor)
    if entry is not None:
        floors = entry['multifloor']['floors']
    else:
        floors = mfu.segment_levels(gfz, gxy, a.mf_level_tol_m, a.mf_plateau_min_m)
        for f in floors:
            f['plan_z'] = f['level'] if abs(f['level'] - j['params']['floor_z']) > a.floor_tol_m else j['params']['floor_z']
    n = len(floors)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 5.2))
    lo = gxy.min(0) - 2.0; hi = gxy.max(0) + 2.0
    traj = np.asarray(entry['trajectory']) if entry else None
    tz = np.asarray(entry['trajectory_z']) if entry else None
    for k, f in enumerate(floors):
        ax = axes[k]
        nav = floor_map(occ, origin, f['plan_z'], h_b, a)
        ext = [origin[0], origin[0] + nav.shape[1] * a.cell_m, origin[1], origin[1] + nav.shape[0] * a.cell_m]
        ax.imshow(nav, origin='lower', extent=ext, cmap='gray', vmin=-0.6, vmax=1.3, interpolation='nearest')
        ax.plot(gxy[:, 0], gxy[:, 1], '--', color='gold', lw=1.2, label='GT')
        ax.plot(gxy[f['i0']:f['i1'] + 1, 0], gxy[f['i0']:f['i1'] + 1, 1], '.', color='gold', ms=4)
        if entry:
            for sgm in entry['segments']:
                p = traj[sgm['i0']:sgm['i1'] + 1]
                if sgm['kind'] == 'floor' and abs(sgm['floor_z'] - f['plan_z']) < 1e-9:
                    ax.plot(p[:, 0], p[:, 1], '-', color='lime' if sgm['planned'] else 'cyan', lw=2,
                            label='ours floor (A*)' if sgm['planned'] else 'floor (GT, short)')
                elif sgm['kind'] == 'stairs':
                    ax.plot(p[:, 0], p[:, 1], '-', color='red', lw=2, alpha=0.8, label='stairs (plan)' if sgm.get('source') == 'plan' else 'stairs (GT)')
            for key in ('a', 'b'):          # gt 모드: 물린 입구/출구 (GT 점)
                if key in f:
                    q = gxy[f[key]]
                    ax.plot(q[0], q[1], 's', mfc='none', mec='deepskyblue', ms=9, mew=2)
            for sgm in entry['segments']:   # plan 모드: 새로 고른 계단 아래(E)·위(X)
                if sgm.get('source') == 'plan':
                    for q in (traj[sgm['i0']], traj[sgm['i1']]):
                        ax.plot(q[0], q[1], 's', mfc='none', mec='deepskyblue', ms=9, mew=2)
        ax.plot(*gxy[0], 'o', color='magenta', ms=7); ax.plot(*gxy[-1], '*', color='magenta', ms=12)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_title(f'floor {k}: level {f["level"]:+.2f} / plan {f["plan_z"]:+.2f} m', fontsize=9)
        if k == 0:
            h, l = ax.get_legend_handles_labels(); d = dict(zip(l, h))
            ax.legend(d.values(), d.keys(), fontsize=7, loc='upper right')
    ax = axes[-1]
    sg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gxy, axis=0), axis=1))])
    ax.plot(sg, gfz, '.-', color='goldenrod', label='GT floor z')
    if entry:
        so = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
        for sgm in entry['segments']:
            i0, i1 = sgm['i0'], sgm['i1'] + 1
            ax.plot(so[i0:i1], tz[i0:i1], '-', color='red' if sgm['kind'] == 'stairs' else 'green', lw=2)
    ax.set_xlabel('xy arclength [m]'); ax.set_ylabel('floor z [m]'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    head = (f'{j["scene_id"]} ep{ep_id} — ' + (f'ok | floor-part min clearance {entry["check_trajectory"]["points_min_m"]:.3f} m | '
            f'stairs {entry["multifloor"]["n_stairs"]} | snap {entry["multifloor"]["snap_shift_m"]}' if entry else f'FAIL {status}'))
    fig.suptitle(head, fontsize=10)
    fig.tight_layout()
    path = out / f'{j["scene_id"]}_ep{ep_id:04d}{"" if entry else "_" + str(status)}.png'
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['vln_ce', 'vln_pe'])
    ap.add_argument('--rig', default=None)
    ap.add_argument('--paths_mf', required=True)
    ap.add_argument('--esdf_dir', required=True)
    ap.add_argument('--episodes', default=None)
    ap.add_argument('--failures', action='store_true')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    args.data_root = args.data_root or dataset_utils.default_data_root(args.dataset)
    j = json.load(open(args.paths_mf))
    pr = j['params']
    data = np.load(Path(args.esdf_dir) / f'{j["scene_id"]}{dataset_utils.dataset_tag(args.dataset)}.npz')
    occ, origin = data['occupancy'], data['origin']
    a = SimpleNamespace(cell_m=pr['cell_m'], h_nav=pr['h_nav'], h_nav_ratio=pr['h_nav_ratio'], plan_r_b=pr['plan_r_b'],
                        downsample_factor=int(round(pr['astar_cell_m'] / pr['cell_m'])),
                        downsample_mode=pr['downsample_mode'], floor_tol_m=pr['floor_tol_m'],
                        **{f'mf_{k}': v for k, v in pr['multifloor'].items()})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    by_id = {e['episode_id']: e for e in j['episodes']}
    if args.episodes:
        ids = [int(x) for x in args.episodes.split(',')]
    else:
        ids = sorted(by_id)[:4]
    if args.failures:
        ids += [int(k) for k in j['stats']['failures']]
    for ep in ids:
        st = j['stats']['failures'].get(str(ep))
        print(plot_episode(j, ep, occ, origin, a, args, out, entry=by_id.get(ep), status=st))


if __name__ == '__main__':
    main()
