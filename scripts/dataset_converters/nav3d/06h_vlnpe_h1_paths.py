"""06h — vln_pe 에서 **H1 로봇이 실제로 걸은 궤적**을 기준으로 GT 경로와 nav3d 생성 경로를 비교한다 (CPU, habitat 컨테이너).

## 용어
- **H1 궤적**: vln_pe parquet 의 `observation.robot_position` (Isaac 에서 H1 이 GT action 을 실행하며 실제로 지나간 몸통 위치, 프레임마다).
- **GT 경로**: R2R `reference_path` (vln_ce · vln_pe 공통, 꺾인 점 몇 개). H1 은 이 경로를 따라가도록 action 을 받았다.
- **nav3d sg**: nav3d 03 이 시작·끝 점만 받고 navmesh 위에 계획해 다듬은 경로. **노은역에 쓴 방식** (노은역엔 GT 경유점이 없다).
- **nav3d wp**: nav3d 03 이 GT 경유점을 차례로 지나도록 계획한 경로 (참고).
- **같은 루트**: 두 경로의 xy 이산 Fréchet 거리 < 0.8 m (문 폭, 02·03 과 같은 정의). **평균 거리**: 한 경로의 점마다 다른 경로까지 최단 거리의 평균.
- **길이비**: 경로 길이 / H1 궤적 길이.

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 |
|---|---|---|
| H0 (참고) | H1 이 GT 경로를 얼마나 따라갔나 | 판정 없음 — 이후 비교의 바닥선 |
| H1 sg↔H1 | nav3d sg 가 H1 궤적과 같은 루트인 비율이, 같은 ep 에서 sg↔GT 같은 루트 비율보다 5 %p 넘게 낮지 않다 | ≥ (sg↔GT) − 5 %p |
| H2 sg 거리 | 같은 루트인 sg 경로의 H1 까지 평균 거리 중앙 | ≤ 0.30 m |
| H3 높이 | 계단 ep 에서 sg 경로 z 범위가 H1 궤적 (몸통 − 0.95 m = 발 높이) z 범위와 같은가 | 범위 끝 차 ≤ 0.30 m |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06h_vlnpe_h1_paths.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GS = HERE.parent / 'gs_vlnpe'
for p in (str(HERE), str(GS), str(HERE.parent / 'instruction_labeling' / 'render')):
    if p not in sys.path:
        sys.path.insert(0, p)
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from esdf_utils import discrete_frechet_distance  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'
DOOR = 0.8
TORSO = 0.95


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_root', default=f'{D}/vln_pe/traj_data/r2r')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/03_refine_paths/scenes')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/06h_vlnpe_h1')
    return ap.parse_args()


def mean_dist(A, B):
    """A 의 점마다 꺾은선 B 까지 최단 거리 (xy) 평균."""
    a, b = B[:-1, :2], B[1:, :2]
    ab = b - a
    t = np.clip(np.einsum('qkd,kd->qk', A[:, None, :2] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(A[:, None, :2] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)
    return float(d.mean())


def L(X):
    return float(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1).sum())


def cmp(S, T):
    """이산 Fréchet 은 주어진 점끼리만 짝짓는다 -> 꺾인 점 몇 개뿐인 GT 경로를 그대로 넣으면 크게 나온다. 둘 다 0.1 m 로 촘촘히 (02 와 같게)."""
    S, T = nu.densify(S, 0.1), nu.densify(T, 0.1)
    fr = discrete_frechet_distance(S[:, :2], T[:, :2])
    return {'frechet': float(fr), 'same': bool(fr < DOOR), 'mean_d': mean_dist(nu.densify(S, 0.1), T), 'len_ratio': L(S) / max(L(T), 1e-6)}


def main():
    import pyarrow.parquet as pq
    from release_io import load_cfg, match_raw, release_episodes
    args = parse()
    rc = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sd in sorted(Path(args.gt_root).iterdir()):
        s = sd.name
        zf = Path(args.nav03_dir) / f'{s}_traj.npz'
        if not zf.exists():
            continue
        Z = np.load(zf)
        ce = {}
        for e in release_episodes(rc, s):
            r = match_raw(rc, s, e['task'])
            if r:
                ce[r['episode_id']] = e['episode_index']
        for line in open(sd / 'meta' / 'episodes.jsonl'):
            p = json.loads(line)
            r = match_raw(rc, s, p['tasks'][0])
            ci = ce.get(r['episode_id']) if r else None
            if ci is None or f'{ci}_sg_ref' not in Z.files:
                continue
            rob = np.asarray(pq.read_table(sd / 'data' / 'chunk-000' / f"episode_{p['episode_index']:06d}.parquet",
                                           columns=['observation.robot_position']).column(0).to_pylist(), dtype=np.float64)
            if len(rob) < 3:
                continue
            H = rob.copy()
            H[:, 2] -= TORSO
            G = nu.hab_to_ours(np.asarray(r['reference_path']))
            row = {'scene': s, 'pe_ep': p['episode_index'], 'ce_ep': ci, 'h1_len': L(H), 'stairs': bool(np.ptp(H[:, 2]) > 0.5),
                   'gt_h1': cmp(G, H)}
            for m in ('sg', 'wp'):
                if f'{ci}_{m}_ref' in Z.files:
                    S = Z[f'{ci}_{m}_ref'].astype(np.float64)
                    row[f'{m}_h1'] = cmp(S, H)
                    row[f'{m}_gt'] = cmp(S, G)
                    row[f'{m}_zerr'] = float(max(abs(S[:, 2].max() - H[:, 2].max()), abs(S[:, 2].min() - H[:, 2].min())))
            rows.append(row)
        print(f'[06h] {s}: 누적 {len(rows)} ep', flush=True)

    def pct(key, sub, sel=lambda r: True):
        v = [r[key][sub] for r in rows if key in r and sel(r)]
        return (100 * float(np.mean(v)) if v else float('nan')), len(v)

    def med(key, sub, sel=lambda r: True):
        v = [r[key][sub] for r in rows if key in r and sel(r)]
        return float(np.median(v)) if v else float('nan')

    gt_same, n = pct('gt_h1', 'same')
    sg_h1, n_sg = pct('sg_h1', 'same')
    sg_gt, _ = pct('sg_gt', 'same')
    wp_h1, n_wp = pct('wp_h1', 'same')
    wp_gt, _ = pct('wp_gt', 'same')
    sg_d = med('sg_h1', 'mean_d', lambda r: r['sg_h1']['same'])
    st = [r for r in rows if r['stairs'] and 'sg_zerr' in r]
    h1 = sg_h1 >= sg_gt - 5
    h2 = sg_d <= 0.30
    h3 = bool(st) and all(r['sg_zerr'] <= 0.30 for r in st)
    gates = [
        {'검증': 'H0 H1 ↔ GT (참고)', '결과': R.pill('참고'), '값': f'같은 루트 {gt_same:.1f} % ({n} ep) · 평균 거리 중앙 {med("gt_h1", "mean_d"):.3f} m · 길이비 중앙 {med("gt_h1", "len_ratio"):.3f}', '기준': '-'},
        {'검증': 'H1 sg ↔ H1', '결과': R.pill(h1), '값': f'같은 루트 {sg_h1:.1f} % vs sg ↔ GT {sg_gt:.1f} % ({n_sg} ep)', '기준': '≥ sg↔GT − 5 %p'},
        {'검증': 'H2 sg 거리', '결과': R.pill(h2), '값': f'같은 루트 ep 의 H1 까지 평균 거리 중앙 {sg_d:.3f} m · 길이비 중앙 {med("sg_h1", "len_ratio"):.3f}', '기준': '≤ 0.30 m'},
        {'검증': 'H3 계단 높이', '결과': R.pill(h3) if st else R.pill('계단 없음'), '값': f'계단 ep {len(st)} · z 범위 끝 차 최대 {max([r["sg_zerr"] for r in st] or [np.nan]):.3f} m', '기준': '≤ 0.30 m'},
        {'검증': '(참고) wp ↔ H1', '결과': R.pill('참고'), '값': f'같은 루트 {wp_h1:.1f} % vs wp ↔ GT {wp_gt:.1f} % ({n_wp} ep)', '기준': '-'},
    ]
    fr = {k: [r[k]['frechet'] for r in rows if k in r] for k in ('gt_h1', 'sg_h1', 'sg_gt', 'wp_h1')}
    figs = make_figs(out, rows, fr, args)
    tab = [{'scene': r['scene'], 'pe ep': r['pe_ep'], '계단': '예' if r['stairs'] else '', 'H1 길이': f"{r['h1_len']:.1f}",
            'GT↔H1 Fréchet': f"{r['gt_h1']['frechet']:.2f}", 'sg↔H1': f"{r['sg_h1']['frechet']:.2f}", 'sg↔GT': f"{r['sg_gt']['frechet']:.2f}",
            'wp↔H1': f"{r['wp_h1']['frechet']:.2f}" if 'wp_h1' in r else '-'} for r in rows]
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': [h1, h2, h3], 'rows': rows}, open(out / 'summary.json', 'w'), indent=1, default=float)
    lead = (f'vln_pe GT 에피소드 중 nav3d 03 sg 경로가 있는 {len(rows)} ep ({len(set(r["scene"] for r in rows))} 씬, 계단 {sum(r["stairs"] for r in rows)}) 에서 '
            f'H1 이 실제로 걸은 궤적을 기준으로 GT 경로와 nav3d 경로를 비교했다. <b>판정 {sum([h1, h2, h3])}/3 통과.</b>')
    body = R.header('nav3d 06h · vln_pe H1 궤적 vs GT · nav3d', 'H1 궤적 기준 경로 비교 — vln_pe', lead)
    body += R.section('용어', R.bullets(['H1 궤적 = vln_pe 로봇이 GT action 을 실행하며 실제로 지나간 위치 (몸통 − 0.95 m)',
                                         'GT 경로 = R2R reference_path (꺾인 점)', 'nav3d sg = 시작·끝만 주고 계획 (노은역 방식) · nav3d wp = GT 경유점을 지나게 계획',
                                         '같은 루트 = xy Fréchet < 0.8 m (문 폭)']))
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다.</p>' + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('그림', ''.join(R.figure(f, c) for f, c in figs))
    body += R.section('에피소드 별', R.table(tab, list(tab[0].keys())))
    R.write(out, 'H1 궤적 기준 경로 비교', body)
    print('gates:', [f"{g['검증']} {'PASS' if p else 'FAIL'}" for g, p in zip(gates[1:4], [h1, h2, h3])])
    for g in gates:
        print(' ', g['검증'], '|', g['값'])
    print(f'-> {out}/report.html')


def make_figs(out, rows, fr, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = []
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 4, 41)
    for k, c in (('gt_h1', 'k'), ('sg_h1', '#e67e22'), ('sg_gt', '#2b5fa8')):
        ax.hist(np.clip(fr[k], 0, 4), bins=bins, histtype='step', color=c, lw=1.5, label=k.replace('_', ' vs '))
    ax.axvline(DOOR, color='r', ls='--', lw=1)
    ax.set_xlabel('xy Frechet distance [m] (red = 0.8 m same-route line)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'frechet_hist.png', dpi=80)
    plt.close(fig)
    figs.append(('frechet_hist.png', 'Fréchet 분포: 검정 GT↔H1, 주황 nav3d sg↔H1, 파랑 nav3d sg↔GT. 빨간 선 왼쪽 = 같은 루트.'))
    import pyarrow.parquet as pq
    from release_io import load_cfg, match_raw  # noqa: F401
    pick = sorted(rows, key=lambda r: -r['sg_h1']['frechet'])[:3] + [r for r in rows if r['stairs']][:2] + rows[:1]
    fig, axs = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.8), squeeze=False)
    for ax, r in zip(axs[0], pick):
        rob = np.asarray(pq.read_table(Path(args.gt_root) / r['scene'] / 'data' / 'chunk-000' / f"episode_{r['pe_ep']:06d}.parquet",
                                       columns=['observation.robot_position']).column(0).to_pylist())
        Z = np.load(Path(args.nav03_dir) / f"{r['scene']}_traj.npz")
        ax.plot(rob[:, 0], rob[:, 1], 'k-', lw=1.5, label='H1')
        S = Z[f"{r['ce_ep']}_sg_ref"]
        ax.plot(S[:, 0], S[:, 1], '-', color='#e67e22', lw=1, label='nav3d sg')
        if f"{r['ce_ep']}_wp_ref" in Z.files:
            W = Z[f"{r['ce_ep']}_wp_ref"]
            ax.plot(W[:, 0], W[:, 1], '--', color='#2b5fa8', lw=0.8, label='nav3d wp')
        ax.set_aspect('equal')
        ax.set_title(f"{r['scene'][:6]} ep{r['pe_ep']} sg-H1 {r['sg_h1']['frechet']:.2f} m", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'examples.png', dpi=80)
    plt.close(fig)
    figs.append(('examples.png', '예: sg↔H1 Fréchet 이 큰 3 개 · 계단 2 개 · 첫 ep. 검정 H1 궤적, 주황 nav3d sg, 파랑 점선 nav3d wp.'))
    return figs


if __name__ == '__main__':
    main()
