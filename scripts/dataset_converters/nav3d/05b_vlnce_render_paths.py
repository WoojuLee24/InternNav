"""05b — nav3d 가 만든 경로(03 다듬은 경로)를 vln_ce 와 **같은 habitat 렌더러·같은 포맷**으로 렌더해 내보내고 검증한다.

    nav3d 03 경로 (wp · sg, 연속 3D)
      -> 0.5 m 간격 경유점 -> habitat GreedyGeodesicFollower (전진 0.25 m / 회전 15°, 릴리스와 같은 이산 action)
      -> instruction_labeling render_trajectory (RigSim: 640×480 · hfov 79° · 1.25 m · pitch 0/30°)
      -> export_scene (릴리스 LeRobot 구조, parquet 21 컬럼, pixel goal)          -> <out_root>/<split>/<scene>/
      -> 판정 (아래) + GT 렌더와 나란히 + report.html

05a 에서 이 렌더러가 GT 를 재현함을 확인했다. 여기서는 **경로만** 바뀐다(같은 에피소드의 시작·끝).

## 용어
- **wp / sg**: nav3d 03 의 두 경로 — wp = GT 경유점을 이어 계획, sg = 시작·끝만 주고 계획.
- **걸은 경로**: follower 가 이산 action 으로 실제로 걸은 위치(프레임마다). 렌더 카메라는 이 위치에 있다.
- **충실도**: 걸은 경로 ↔ nav3d 경로 거리. 걸은→nav3d 최대(벗어남), nav3d→걸은 최대(빠뜨림).
- **충돌**: habitat 이 그 step 에서 벽에 막혀 미끄러진(sliding) 표시(`collided`).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| P1 완주 | follower 가 끝까지 가나 | 전부: 잘림 0 · follower 오류 0 · 끝점 ≤ 0.5 m | 못 가면 그 경로는 버린다 |
| P2 충실도 | 걸은 경로가 nav3d 경로를 따르나 | 경로의 ≥ 95 %: 벗어남 ≤ 0.30 m · 빠뜨림 ≤ 0.40 m | 회전 15° · 전진 0.25 m 이산화 폭 |
| P3 충돌 | 벽에 막힌 step | 경로의 ≥ 95 % 가 0 회 | GT follower 도 벽 근처에서 드물게 미끄러진다 |
| P4 계단 높이 | 계단 경로에서 걸은 z 가 nav3d z 를 따르나 (걸은 위치를 nav3d 경로에 xy 투영한 자리의 z 와 비교) | 계단 경로 전부 최대 |Δz| ≤ 0.15 m | 챌판 0.17 m 보다 작아야 다른 단을 밟은 게 아니다 |
| P5 형식 | 내보낸 것이 릴리스와 같은 구조 | 스트림 프레임 수 == parquet 행, 21 컬럼, 이미지 640×480 | 학습 로더가 그대로 읽어야 한다 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/05b_vlnce_render_paths.py
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'render'))
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402

LOG = 'scripts/dataset_converters/nav3d/logs/05b_vlnce_render_paths'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/dataset_converters/instruction_labeling/configs/render.yaml')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/episodes.json')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/03_refine_paths/scenes')
    ap.add_argument('--nav02_dir', default='scripts/dataset_converters/nav3d/logs/02_vlnce_gt_compare/scenes')
    ap.add_argument('--modes', default='wp,sg')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/vln_ce_nav3d/traj_data/r2r', help='<split>/<scene>/ 로 쓴다')
    ap.add_argument('--waypoint_m', type=float, default=0.5, help='follower 경유점 간격 (nav3d 경로를 이 간격으로 솎음)')
    ap.add_argument('--goal_radius', type=float, default=0.25, help='경유점 도달 반경 (release config 와 같게)')
    ap.add_argument('--out_dir', default=f'{LOG}')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--no_render', action='store_true', help='캐시만으로 판정·리포트 다시 (채우기·내보내기는 다시 한다)')
    ap.add_argument('--depth_fill', default='none', choices=['none', 'min19'],
                    help='depth 구멍 채우기. none(기본) = habitat 원본 — 2026-10-15 사용자 결정: 채워도 릴리스와의 차이가 크게 줄지 않음(05a). '
                         'min19 = 3×3 최소 이웃 × 19 (depth_fill.py, 참고용)')
    return ap.parse_args()


def resample(P: np.ndarray, step: float) -> np.ndarray:
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.r_[np.arange(0, L[-1], step), L[-1]]
    return np.stack([np.interp(s, L, P[:, k]) for k in range(3)], 1)


def seg_dist(Q: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Q 의 각 점 -> 꺾은선 P 까지 최소 거리 (3D)."""
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(np.einsum('qkd,kd->qk', Q[:, None] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(Q[:, None] - proj, axis=2).min(1)


def stairs_dz_projected(W: np.ndarray, P: np.ndarray, st: np.ndarray) -> float:
    """걸은 위치(프레임)마다 xy 를 nav3d 경로 꺾은선에 투영해 그 자리의 nav3d z 와 비교 — 계단 구간에 투영된 프레임만, 최대 |Δz|."""
    a, b = P[:-1, :2], P[1:, :2]
    ab = b - a
    t = np.clip(np.einsum('qkd,kd->qk', W[:, None, :2] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(W[:, None, :2] - (a[None] + t[..., None] * ab[None]), axis=2)
    k = d.argmin(1)
    tt = t[np.arange(len(W)), k]
    zp = P[k, 2] + tt * (P[k + 1, 2] - P[k, 2])
    on = st[k] | st[k + 1]
    return float(np.max(np.abs(W[on, 2] - zp[on]))) if on.any() else 0.0


def apply_depth_fill(tdir: Path, rigs: list, mode: str) -> dict:
    """캐시 depth 를 채운다. 원본은 depth_raw.<rig>/ 에 한 번만 옮겨 두고, depth.<rig>/ 는 늘 원본에서 다시 만든다(멱등)."""
    import cv2
    from depth_fill import fill_depth_holes
    zr, zf = [], []
    for r in rigs:
        raw_dir, out_dir = tdir / f'depth_raw.{r}', tdir / f'depth.{r}'
        if not raw_dir.exists():
            out_dir.rename(raw_dir)
            out_dir.mkdir()
        for pth in sorted(raw_dir.glob('*.png')):
            d = cv2.imread(str(pth), -1)
            if mode == 'min19':
                f = fill_depth_holes(d)
            elif mode == 'none':
                f = d
            else:
                assert False, f'unreachable depth_fill={mode!r}'
            cv2.imwrite(str(out_dir / pth.name), f.astype(np.uint16))
            zr.append(float(np.mean(d == 0)))
            zf.append(float(np.mean(f == 0)))
    return {'depth_zero_raw': float(np.mean(zr)), 'depth_zero_filled': float(np.mean(zf)), 'depth_fill': mode}


def main():
    args = parse()
    from release_io import load_cfg, match_raw, release_episodes, scene_glb
    from habitat_rig_sim import RigSim, ACTION_NAMES
    import render_split as RS

    cfg = load_cfg(args.config)
    cfg = copy.deepcopy(cfg)
    cfg['paths']['out_root'] = Path(args.out_root)
    cfg['paths']['cache_root'] = Path(args.out_root).parent / '_traj_cache'
    cfg['action_gen'] = {'method': 'greedy_wp', 'goal_radius': args.goal_radius, 'max_steps': 1000}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.episodes_json).read_text())
    modes = [m for m in args.modes.split(',') if m]
    qt = RS.release_qtables(cfg)
    schema = RS.load_schema(cfg['image']['schema_ref'])
    rows = []
    for scene, eps in sel.items():
        Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        mf = {e['ep']: e for e in json.load(open(Path(args.nav02_dir) / f'{scene}.json'))['episodes']}
        rel = {e['episode_index']: e for e in release_episodes(cfg, scene)}
        rs = None if args.no_render else RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        for mode in modes:
            cache = cfg['paths']['cache_root'] / f'nav3d_{mode}' / scene
            cache.mkdir(parents=True, exist_ok=True)
            trajs = {}
            for ep in eps:
                key = f'{ep}_{mode}_ref'
                if key not in Z:
                    rows.append({'scene': scene, 'ep': ep, 'mode': mode, 'status': 'no_path'})
                    continue
                raw = match_raw(cfg, scene, rel[ep]['task'])
                P = Z[key].astype(np.float64)
                st_mask = Z[f'{ep}_{mode}_ref_st'].astype(bool) if f'{ep}_{mode}_ref_st' in Z else np.zeros(len(P), bool)
                W = resample(P, args.waypoint_m)
                Wh = nu.ours_to_hab(W)
                tid = int(ep)
                e = {'episode_id': int(ep), 'trajectory_id': tid, 'scene_id': f'{scene}.glb',
                     'start_position': Wh[0].tolist(), 'start_rotation': raw['start_rotation'],
                     'reference_path': Wh.tolist(), 'goals': [{'position': Wh[-1].tolist()}],
                     'instruction': {'instruction_text': raw['instruction']['instruction_text']}}
                trajs[tid] = [e]
                tdir = cache / str(tid)
                if rs is not None:
                    meta = RS.render_trajectory(rs, e, cfg, tdir, qt)
                    # 충돌 수: 같은 action 을 다시 재생하며 habitat 의 collided 를 센다 (렌더 없음)
                    rs.set_start(e['start_position'], e['start_rotation'])
                    col = 0
                    for a in meta['actions'][1:]:
                        o = rs.sim.step(ACTION_NAMES[int(a)])
                        col += int(bool(o.get('collided', False)))
                    meta['collisions'] = col
                    (tdir / 'meta.json').write_text(json.dumps(meta))
                meta = json.loads((tdir / 'meta.json').read_text())
                meta.update(apply_depth_fill(tdir, list(cfg['camera']['rigs']), args.depth_fill))
                (tdir / 'meta.json').write_text(json.dumps(meta))
                Wk = nu.hab_to_ours(np.array(meta['positions']))
                off = seg_dist(Wk, P)
                miss = seg_dist(P, Wk) if len(Wk) > 1 else np.full(len(P), np.inf)
                dz_st, dz_near = None, None
                if mf[ep]['multifloor'] or st_mask.any():
                    k = st_mask if st_mask.any() else np.ones(len(P), bool)
                    dz_st = stairs_dz_projected(Wk, P, k)
                    # (참고) 이전 측정: nav3d 계단 점마다 xy 로 가장 가까운 걸은 프레임의 z 차 — 프레임 간격 0.25 m 때문에
                    # 가파른 계단에서 최대 ~0.2 m 까지 부풀려진다(1LXt ep14: 이 값 0.22 vs 투영 0.02). 판정에는 쓰지 않는다.
                    d2 = np.linalg.norm(P[k, None, :2] - Wk[None, :, :2], axis=2)
                    dz_near = float(np.max(np.abs(P[k, 2] - Wk[d2.argmin(1), 2])))
                rows.append({'scene': scene, 'ep': ep, 'mode': mode, 'status': 'ok', 'multifloor': bool(mf[ep]['multifloor']),
                             'gt_dz_m': float(mf[ep]['gt_dz_m']), 'path_len_m': float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                             'frames': meta['frames'], 'truncated': meta['truncated'], 'follower_errors': meta['follower_errors'],
                             'end_m': float(np.linalg.norm(Wk[-1] - P[-1])), 'off_max_m': float(off.max()), 'off_med_m': float(np.median(off)),
                             'miss_max_m': float(miss.max()), 'collisions': int(meta.get('collisions', -1)), 'stairs_dz_max_m': dz_st, 'stairs_dz_nearest_frame_m': dz_near,
                             'n_forward': int(np.sum(np.array(meta['actions']) == 1)), 'n_turn': int(np.sum(np.isin(meta['actions'], [2, 3]))),
                             'depth_zero_raw': meta['depth_zero_raw'], 'depth_zero_filled': meta['depth_zero_filled']})
                r_ = rows[-1]
                print(f"  {scene} ep{ep} {mode}: {r_['frames']} f, end {r_['end_m']:.2f} off {r_['off_max_m']:.2f} miss {r_['miss_max_m']:.2f} "
                      f"col {r_['collisions']} dz {dz_st}", flush=True)
            if trajs:
                idx = cfg['paths']['out_root'] / f'nav3d_{mode}' / '_index'
                idx.mkdir(parents=True, exist_ok=True)
                with open(idx / f'{scene}.jsonl', 'w') as f:
                    RS.export_scene(cfg, f'nav3d_{mode}', scene, trajs, cache, schema, f)
        if rs is not None:
            rs.close()

    fmt = check_format(cfg, sel, modes)
    report(out, args, cfg, rows, fmt, sel, modes)


def check_format(cfg, sel, modes) -> list:
    import cv2
    import pyarrow.parquet as pq
    res = []
    for mode in modes:
        for scene in sel:
            base = cfg['paths']['out_root'] / f'nav3d_{mode}' / scene
            pqs = sorted((base / 'data' / 'chunk-000').glob('episode_*.parquet'))
            for p in pqs:
                t = pq.read_table(p)
                ei = int(p.stem.split('_')[1])
                ok, why = t.num_columns == 21, [] if t.num_columns == 21 else [f'{t.num_columns} cols']
                for d in sorted((base / 'videos' / 'chunk-000').iterdir()):
                    n = len(list(d.glob(f'episode_{ei:06d}_*')))
                    if 'topdown' in d.name:
                        continue
                    if n != t.num_rows and n != 0:
                        ok = False
                        why.append(f'{d.name}: {n} != {t.num_rows}')
                im = cv2.imread(str(base / 'videos' / 'chunk-000' / 'observation.images.rgb.125cm_30deg' / f'episode_{ei:06d}_0.jpg'))
                if im is None or im.shape != (480, 640, 3):
                    ok = False
                    why.append('rgb shape')
                res.append({'mode': mode, 'scene': scene, 'ep_index': ei, 'rows': t.num_rows, 'ok': ok, 'why': '; '.join(why)})
    return res


def report(out, args, cfg, rows, fmt, sel, modes):
    ok = [r for r in rows if r['status'] == 'ok']
    st = [r for r in ok if r['multifloor']]
    p1 = all((not r['truncated']) and r['follower_errors'] == 0 and r['end_m'] <= 0.5 for r in ok)
    p2v = np.mean([(r['off_max_m'] <= 0.30) and (r['miss_max_m'] <= 0.40) for r in ok]) if ok else 0
    p3v = np.mean([r['collisions'] == 0 for r in ok]) if ok else 0
    p4 = bool(st) and all(r['stairs_dz_max_m'] is not None and r['stairs_dz_max_m'] <= 0.15 for r in st)
    p5 = bool(fmt) and all(f['ok'] for f in fmt)
    gates = [
        {'검증': 'P1 완주', '결과': R.pill(p1), '값': f"{sum((not r['truncated']) and r['follower_errors'] == 0 and r['end_m'] <= 0.5 for r in ok)}/{len(ok)} 경로 (끝점 최대 {max([r['end_m'] for r in ok] or [0]):.2f} m)", '기준': '전부'},
        {'검증': 'P2 충실도', '결과': R.pill(p2v >= 0.95), '값': f"{100 * p2v:.1f} % (벗어남 최대 {max([r['off_max_m'] for r in ok] or [0]):.2f} · 빠뜨림 최대 {max([r['miss_max_m'] for r in ok] or [0]):.2f} m)", '기준': '≥ 95 % 가 ≤ 0.30 / ≤ 0.40 m'},
        {'검증': 'P3 충돌', '결과': R.pill(p3v >= 0.95), '값': f"{100 * p3v:.1f} % 가 0 회 (최대 {max([r['collisions'] for r in ok] or [0])} 회)", '기준': '≥ 95 %'},
        {'검증': 'P4 계단 높이', '결과': R.pill(p4), '값': f"계단 경로 {len(st)} · |Δz| 최대 {max([r['stairs_dz_max_m'] or 0 for r in st] or [float('nan')]):.3f} m", '기준': '전부 ≤ 0.15 m'},
        {'검증': 'P5 형식', '결과': R.pill(p5), '값': f"{sum(f['ok'] for f in fmt)}/{len(fmt)} 에피소드", '기준': '프레임 수 == 행 · 21 컬럼 · 640×480'},
    ]
    passed = [g['결과'].count('PASS') == 1 for g in gates]
    figs = make_figures(out, args, cfg, ok, sel, modes)
    tab = [{'scene': r['scene'], 'ep': r['ep'], 'mode': r['mode'], '종류': f"계단 {r['gt_dz_m']:+.1f} m" if r['multifloor'] else '평지',
            '길이 [m]': f"{r['path_len_m']:.1f}", '프레임': r['frames'], '전진/회전': f"{r['n_forward']}/{r['n_turn']}",
            '끝점 [m]': f"{r['end_m']:.2f}", '벗어남 최대 [m]': f"{r['off_max_m']:.2f}", '빠뜨림 최대 [m]': f"{r['miss_max_m']:.2f}",
            'depth 0 % (원본→채움)': f"{100 * r['depth_zero_raw']:.2f}→{100 * r['depth_zero_filled']:.2f}",
            '충돌': r['collisions'], '계단 |Δz| [m]': '-' if r['stairs_dz_max_m'] is None else f"{r['stairs_dz_max_m']:.3f}",
            '(참고) 가까운 프레임 |Δz|': '-' if r.get('stairs_dz_nearest_frame_m') is None else f"{r['stairs_dz_nearest_frame_m']:.3f}"} for r in ok]
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': passed, 'rows': rows, 'format': fmt},
              open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False, default=float)
    lead = (f'nav3d 03 경로(wp · sg)를 05a 에서 검증한 habitat 렌더러로 그리고 릴리스와 같은 구조로 내보냈다. '
            f'경로 {len(ok)} (계단 {len(st)}) · 출력 <span class="mono">{R.esc(str(cfg["paths"]["out_root"]))}/nav3d_{{wp,sg}}/</span>. '
            f'<b>판정 {sum(passed)}/{len(passed)} 통과.</b> depth 는 릴리스처럼 구멍을 채웠다(<span class="mono">--depth_fill {args.depth_fill}</span>: '
            f'0 인 픽셀 평균 {100 * np.mean([r["depth_zero_raw"] for r in ok]):.2f} → {100 * np.mean([r["depth_zero_filled"] for r in ok]):.2f} %). '
            f'지시문은 아직 GT 문장을 그대로 넣었다(경로가 다르므로 다시 만들어야 함).')
    body = R.header('nav3d 05b · vln_ce 생성 경로 렌더 (habitat)', 'vln_ce 생성 경로 렌더 — habitat, 릴리스 포맷', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다. P4 측정 방법만 고쳤다 — 처음엔 nav3d 계단 점마다 xy 로 가장 가까운 걸은 프레임의 z 와 비교했는데, '
                      '프레임 간격 0.25 m 때문에 가파른 계단에서 최대 0.22 m 까지 부풀었다. 걸은 위치를 nav3d 경로에 투영해 비교하도록 바꿨다(이전 값은 표의 참고 열). 걸은 경로 = follower 가 이산 action 으로 실제 걸은 위치(= 카메라 위치). '
                      '벗어남 = 걸은 점에서 nav3d 경로까지 최대 거리, 빠뜨림 = nav3d 경로 점에서 걸은 경로까지 최대 거리.</p>'
                      + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('경로 별', R.table(tab, list(tab[0].keys())) if tab else '<p>없음</p>')
    body += R.section('위에서 본 경로 · 높이', ''.join(R.figure(f, c) for f, c in figs['paths']))
    body += R.section('GT 렌더 vs 생성 경로 렌더 (같은 에피소드)', '<p class="muted">rig 125cm_30deg. 경로 진행률 0 · 20 · 40 · 60 · 80 · 100 % 지점. '
                      '위: 릴리스(GT 경로) · 가운데: nav3d wp · 아래: nav3d sg.</p>' + ''.join(R.figure(f, c) for f, c in figs['frames']))
    body += R.section('형식 검사', R.table([f for f in fmt if not f['ok']][:20] or [{'결과': '모두 통과'}],
                                          ['mode', 'scene', 'ep_index', 'rows', 'why'] if any(not f['ok'] for f in fmt) else ['결과']))
    R.write(out, 'vln_ce 생성 경로 렌더', body)
    print('gates:', ['%s %s' % (g['검증'].split()[0], 'PASS' if p else 'FAIL') for g, p in zip(gates, passed)])
    print(f'-> {out}/report.html')


def make_figures(out, args, cfg, ok, sel, modes):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from release_io import release_image_path
    figs = {'paths': [], 'frames': []}
    col = {'gt': 'k', 'wp': '#2b5fa8', 'sg': '#e67e22'}
    for scene, eps in sel.items():
        Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        Zg = np.load(Path(args.nav02_dir) / f'{scene}_traj.npz')
        n = len(eps)
        fig, axs = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)
        for c, ep in enumerate(eps):
            ax, az = axs[0, c], axs[1, c]
            if f'{ep}_gt' in Zg:
                G = Zg[f'{ep}_gt']
                ax.plot(G[:, 0], G[:, 1], color=col['gt'], lw=1.5, label='GT')
                az.plot(np.r_[0, np.cumsum(np.linalg.norm(np.diff(G[:, :2], axis=0), axis=1))], G[:, 2], color=col['gt'], lw=1.5)
            for mode in modes:
                if f'{ep}_{mode}_ref' not in Z:
                    continue
                P = Z[f'{ep}_{mode}_ref']
                ax.plot(P[:, 0], P[:, 1], color=col[mode], lw=1, ls='--', label=f'nav3d {mode}')
                tdir = cfg['paths']['cache_root'] / f'nav3d_{mode}' / scene / str(ep)
                if (tdir / 'meta.json').exists():
                    W = nu.hab_to_ours(np.array(json.loads((tdir / 'meta.json').read_text())['positions']))
                    ax.plot(W[:, 0], W[:, 1], '.', color=col[mode], ms=2)
                    az.plot(np.r_[0, np.cumsum(np.linalg.norm(np.diff(W[:, :2], axis=0), axis=1))], W[:, 2], color=col[mode], lw=1)
            ax.set_aspect('equal')
            ax.set_title(f'ep{ep}', fontsize=9)
            az.set_xlabel('distance along path [m]')
            if c == 0:
                ax.legend(fontsize=7)
                az.set_ylabel('z [m]')
        fig.suptitle(f'{scene}: top = paths (line = nav3d, dots = walked frames), bottom = height along path')
        fig.tight_layout()
        fn = f'paths_{scene}.png'
        fig.savefig(out / fn, dpi=70)
        plt.close(fig)
        figs['paths'].append((fn, f'{scene}: 검정 = GT, 파랑 = nav3d wp, 주황 = nav3d sg. 선 = nav3d 경로, 점 = follower 가 걸은 프레임 위치.'))
        # 계단 에피소드 하나 + 평지 하나: GT 렌더 vs wp vs sg 프레임
        show = [e for e in eps if any(r['scene'] == scene and r['ep'] == e and r['multifloor'] for r in ok)][:1] + \
               [e for e in eps if any(r['scene'] == scene and r['ep'] == e and not r['multifloor'] for r in ok)][:1]
        for ep in show:
            fig, axs = plt.subplots(1 + len(modes), 6, figsize=(18, 2.6 * (1 + len(modes))), squeeze=False)
            rel_dir = Path(cfg['paths']['release_root'])
            T = len(list((rel_dir / scene / 'videos' / 'chunk-000' / 'observation.images.rgb.125cm_30deg').glob(f'episode_{ep:06d}_*.jpg')))
            for c, q in enumerate(np.linspace(0, 1, 6)):
                f = int(round(q * (T - 1)))
                axs[0, c].imshow(cv2.cvtColor(cv2.imread(str(release_image_path(cfg, scene, 'rgb', '125cm_30deg', ep, f))), cv2.COLOR_BGR2RGB))
                axs[0, c].set_title(f'GT f{f}', fontsize=8)
                for r_, mode in enumerate(modes, 1):
                    tdir = cfg['paths']['cache_root'] / f'nav3d_{mode}' / scene / str(ep)
                    if (tdir / 'meta.json').exists():
                        Tm = json.loads((tdir / 'meta.json').read_text())['frames']
                        fm = int(round(q * (Tm - 1)))
                        axs[r_, c].imshow(cv2.cvtColor(cv2.imread(str(tdir / 'rgb.125cm_30deg' / f'{fm}.jpg')), cv2.COLOR_BGR2RGB))
                        axs[r_, c].set_title(f'nav3d {mode} f{fm}', fontsize=8)
            for a in axs.ravel():
                a.axis('off')
            fig.tight_layout()
            fn = f'frames_{scene}_{ep}.png'
            fig.savefig(out / fn, dpi=60)
            plt.close(fig)
            kind = '계단' if any(r['scene'] == scene and r['ep'] == ep and r['multifloor'] for r in ok) else '평지'
            figs['frames'].append((fn, f'{scene} ep{ep} ({kind})'))
    return figs


if __name__ == '__main__':
    main()
