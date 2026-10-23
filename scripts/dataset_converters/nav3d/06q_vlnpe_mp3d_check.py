"""06q — vln_pe 판 05b: MP3D 에서 nav3d **sg** 경로를 노은역과 같은 방식(04b mesh 바닥 → 06p)으로 vln_pe 포맷으로 렌더하고 검사한다.

## 용어
- **sg 경로**: nav3d 03 이 시작·끝 점만 받고 계획한 경로 (노은역 방식). 05 의 4 씬 28 ep.
- **카메라 값**: 06v 에서 vln_pe GT (H1) 로 잰 값 — mesh 바닥 위 1.656 m · 몸보다 앞 0.239 m · 아래로 18.1°. robot_position z = 바닥 + 0.95 m (H1 몸통).
- **mesh 바닥 위 카메라 높이**: parquet camera_position 에서 아래로 쏜 광선이 MP3D mesh 에 닿는 거리.

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 |
|---|---|---|
| Q1 완주 | 마지막 몸 위치가 경로 끝에 닿았나 | 전부 ≤ 0.5 m |
| Q2 충실도 | 몸 위치가 경로를 따르나 | 몸→경로 최대 ≤ 0.30 m 인 ep ≥ 95 % |
| Q3 카메라 높이 | 평지 프레임의 mesh 위 높이 | |중앙 − 1.656| ≤ 0.03 m · |차| p95 ≤ 0.08 m |
| Q4 형식 | 릴리스 스키마 · 학습 로더(LerobotAsLmdb) | 08 과 같은 검사 전부 통과 |
| Q5 시작 heading | GT start_rotation 으로 출발하면 첫 프레임 방향이 vln_pe GT 와 같은가 (2026-10-18 추가, 결과 전에 정함) | 첫 프레임 yaw 차 중앙 ≤ 5° |
| (참고) GT 와 나란히 | 같은 R2R ep 의 vln_pe GT 첫·중간·끝 프레임 | 그림만 |

사용 (저장소 루트):
    (habitat) /usr/bin/python scripts/dataset_converters/nav3d/06q_vlnpe_mp3d_check.py --stage prepare
    (isaac)   씬마다 06p_render_noeun_vlnpe.py --paths_json <prepare 출력> --mesh_root data/scene_data/mp3d_pe --scene <씬> ... (prepare 가 명령을 출력)
    (habitat) /usr/bin/python scripts/dataset_converters/nav3d/06q_vlnpe_mp3d_check.py --stage check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent / 'gs_vlnpe'), str(HERE.parent / 'instruction_labeling' / 'render')):
    if p not in sys.path:
        sys.path.insert(0, p)
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from floor_mesh import mesh_floor_z  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'
LOG = 'scripts/dataset_converters/nav3d/logs/06q_vlnpe_mp3d'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['prepare', 'check'])
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/episodes.json')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/03_refine_paths/scenes')
    ap.add_argument('--nav02_dir', default='scripts/dataset_converters/nav3d/logs/02_vlnce_gt_compare/scenes')
    ap.add_argument('--scene_root', default=f'{D}/scene_data/mp3d_ce/mp3d', help='glb (z-up) — 바닥 광선')
    ap.add_argument('--h06v', default='scripts/dataset_converters/nav3d/logs/06v_vlnpe/prepare.json', help='06v 카메라 값')
    ap.add_argument('--h06h', default='scripts/dataset_converters/nav3d/logs/06h_vlnpe_h1/summary.json', help='vln_ce ep ↔ vln_pe ep 짝')
    ap.add_argument('--out_root', default=f'{D}/vln_pe_nav3d_sg/traj_data/r2r')
    ap.add_argument('--gt_root', default=f'{D}/vln_pe/traj_data/r2r')
    ap.add_argument('--log_dir', default=LOG)
    return ap.parse_args()


def prepare(args):
    import trimesh
    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    import math
    from release_io import load_cfg, match_raw, release_episodes
    rc = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')
    rig = json.load(open(args.h06v))['rig']
    sel = json.loads(Path(args.episodes_json).read_text())
    cmds = []
    for scene, ids in sel.items():
        Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        mf = {e['ep']: e for e in json.load(open(Path(args.nav02_dir) / f'{scene}.json'))['episodes']}
        mesh = trimesh.load(f'{args.scene_root}/{scene}/{scene}.glb', force='mesh')
        rel = {e['episode_index']: e for e in release_episodes(rc, scene)}
        eps = []
        for ep in ids:
            if f'{ep}_sg_ref' not in Z.files:
                continue
            T = Z[f'{ep}_sg_ref'].astype(np.float64)
            st = Z[f'{ep}_sg_ref_st'].astype(bool) if f'{ep}_sg_ref_st' in Z.files else np.zeros(len(T), bool)
            gs = nu.hab_to_ours(np.asarray(match_raw(rc, scene, rel[ep]['task'])['start_position']))
            if np.linalg.norm(T[0, :2] - gs[:2]) > 1e-3:                       # 시작 위치 = GT 그대로 (03 이 벽에서 밀어낸 경우 GT 점을 앞에 붙인다)
                T = np.vstack([np.r_[gs[:2], T[0, 2]], T])
                st = np.r_[False, st]
            fm = mesh_floor_z(T, st, mesh)
            q = match_raw(rc, scene, rel[ep]['task'])['start_rotation']           # GT 시작 heading (R2R, 실험 셋업 그대로)
            compass = -math.degrees(2 * math.atan2(q[1], q[3]))
            eps.append({'episode_id': f'{scene}:{ep}', 'ce_ep': int(ep), 'kind': 'multifloor' if mf[ep]['multifloor'] else 'flat',
                        'start_yaw': math.radians(90.0 - compass),
                        'trajectory': np.c_[T[:, :2], fm['z']].tolist(), 'stairs_mask': st.tolist()})
        pj = log / f'paths_{scene}.json'
        pj.write_text(json.dumps({'scene': scene, 'episodes': eps}))
        cmds.append(f"/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06p_render_noeun_vlnpe.py --paths_json {pj} "
                    f"--mesh_root data/scene_data/mp3d_pe --scene {scene} --out_root {args.out_root} --out_scene {scene} "
                    f"--mount_height_m {rig['h_mesh']:.4f} --pitch_down_deg {rig['pitch']:.3f} --forward_m {rig['fwd']:.4f} --robot_base_h 0.95 "
                    f"--render_far_m 100 --log_dir {log}/render_{scene}")
        print(f'[06q] {scene}: {len(eps)} sg 경로 -> {pj}', flush=True)
    (log / 'render_cmds.sh').write_text('\n'.join(cmds) + '\n')
    print(f'[06q] 렌더 명령 -> {log}/render_cmds.sh')


def check(args):
    import pyarrow.parquet as pq
    import trimesh
    log = Path(args.log_dir)
    rig = json.load(open(args.h06v))['rig']
    H = rig['h_mesh']
    pairs = {}
    for r in json.load(open(args.h06h))['rows']:
        pairs[(r['scene'], r['ce_ep'])] = r['pe_ep']
    ref_schema = pq.read_schema(Path(args.gt_root) / '17DRP5sb8fy' / 'data' / 'chunk-000' / 'episode_000000.parquet')
    rows, figs = [], []
    for pj in sorted(log.glob('paths_*.json')):
        d = json.loads(pj.read_text())
        s = d['scene']
        mesh = trimesh.load(f'{args.scene_root}/{s}/{s}.glb', force='mesh')
        base = Path(args.out_root) / s
        for ei, e in enumerate(d['episodes']):
            pqf = base / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet'
            if not pqf.exists():
                rows.append({'scene': s, 'ep': e['ce_ep'], 'missing': True})
                continue
            t = pq.read_table(pqf).to_pydict()
            cam = np.asarray(t['observation.camera_position'])
            rob = np.asarray(t['observation.robot_position'])
            loc, idx, _ = mesh.ray.intersects_location(cam, np.tile([0, 0, -1.0], (len(cam), 1)), multiple_hits=False)
            h = np.full(len(cam), np.nan)
            h[idx] = cam[idx, 2] - loc[:, 2]
            T = np.asarray(e['trajectory'])
            st = np.asarray(e['stairs_mask'], bool)
            a, b = T[:-1, :2], T[1:, :2]
            ab = b - a
            tt = np.clip(np.einsum('qkd,kd->qk', rob[:, None, :2] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
            dd = np.linalg.norm(rob[:, None, :2] - (a[None] + tt[..., None] * ab[None]), axis=2)
            k = dd.argmin(1)
            on_floor = ~(st[k] | st[np.minimum(k + 1, len(st) - 1)])
            rows.append({'scene': s, 'ep': e['ce_ep'], 'kind': e['kind'], 'frames': len(cam), 'end_m': float(np.linalg.norm(rob[-1, :2] - T[-1, :2])),
                         'off_max_m': float(dd.min(1).max()), 'h': h[on_floor].tolist(),
                         'schema_ok': pq.read_schema(pqf).equals(ref_schema, check_metadata=True), 'pe_ep': pairs.get((s, e['ce_ep']))})
    ok = [r for r in rows if not r.get('missing')]
    # Q5 시작 heading: 첫 프레임 카메라 yaw 가 vln_pe GT 와 같은가 + 처음 제자리 회전 수
    for pj in sorted(log.glob('paths_*.json')):
        d = json.loads(pj.read_text())
        for ei, e in enumerate(d['episodes']):
            r = next((x for x in ok if x['scene'] == d['scene'] and x['ep'] == e['ce_ep']), None)
            if r is None or r['pe_ep'] is None:
                continue
            o = pq.read_table(Path(args.out_root) / d['scene'] / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet').to_pydict()
            g = pq.read_table(Path(args.gt_root) / d['scene'] / 'data' / 'chunk-000' / f"episode_{r['pe_ep']:06d}.parquet").to_pydict()
            r['yaw0_diff_deg'] = float(abs((np.degrees(o['observation.camera_yaw'][0] - g['observation.camera_yaw'][0]) + 180) % 360 - 180))
            lead = lambda a: next((i for i, x in enumerate(a) if x not in (2, 3)), len(a))  # noqa: E731
            r['init_turns'] = [lead(o['observation.action']), lead(g['observation.action'])]
    q1 = bool(ok) and all(r['end_m'] <= 0.5 for r in ok) and len(ok) == len(rows)
    q2v = float(np.mean([r['off_max_m'] <= 0.30 for r in ok])) if ok else 0.0
    hh = np.concatenate([np.asarray(r['h'], float) for r in ok]) if ok else np.array([np.nan])
    q3 = bool(abs(np.nanmedian(hh) - H) <= 0.03 and np.nanpercentile(np.abs(hh - H), 95) <= 0.08)
    s08 = json.load(open('scripts/dataset_converters/nav3d/logs/08_loader_check/summary_06q.json')) if Path('scripts/dataset_converters/nav3d/logs/08_loader_check/summary_06q.json').exists() else None
    q4 = all(r['schema_ok'] for r in ok) and bool(s08) and all(x['ok'] for x in s08['vln_pe'])
    yd = [r['yaw0_diff_deg'] for r in ok if 'yaw0_diff_deg' in r]
    q5 = bool(yd) and float(np.median(yd)) <= 5.0
    gates = [
        {'검증': 'Q1 완주', '결과': R.pill(q1), '값': f"{sum(r['end_m'] <= 0.5 for r in ok)}/{len(rows)} ep (끝 차 최대 {max([r['end_m'] for r in ok] or [np.nan]):.2f} m)", '기준': '전부 ≤ 0.5 m'},
        {'검증': 'Q2 충실도', '결과': R.pill(q2v >= 0.95), '값': f"{100 * q2v:.1f} % (최대 {max([r['off_max_m'] for r in ok] or [np.nan]):.2f} m)", '기준': '≥ 95 % 가 ≤ 0.30 m'},
        {'검증': 'Q3 카메라 높이', '결과': R.pill(q3), '값': f"평지 프레임 mesh 위 중앙 {np.nanmedian(hh):.3f} (목표 {H:.3f}) · |차| p95 {np.nanpercentile(np.abs(hh - H), 95):.3f} m", '기준': '|중앙 차| ≤ 0.03 · p95 ≤ 0.08'},
        {'검증': 'Q5 시작 heading', '결과': R.pill(q5), '값': f"GT 짝 {len(yd)} ep · 첫 프레임 yaw 차 중앙 {np.median(yd) if yd else float('nan'):.1f}° · 처음 제자리 회전 수 (우리, GT): {[r['init_turns'] for r in ok if 'init_turns' in r]}", '기준': '중앙 ≤ 5°'},
        {'검증': 'Q4 형식·로더', '결과': R.pill(q4), '값': f"스키마 {sum(r['schema_ok'] for r in ok)}/{len(ok)} · 로더 {'통과' if s08 and all(x['ok'] for x in s08['vln_pe']) else '미실행/실패'}", '기준': '전부'},
    ]
    figs = make_figs(log, ok, args)
    tab = [{'scene': r['scene'], 'ep': r['ep'], '종류': r['kind'], '프레임': r['frames'], '끝 차': f"{r['end_m']:.2f}", '벗어남 최대': f"{r['off_max_m']:.2f}",
            'mesh 위 높이 중앙': f"{np.nanmedian(r['h']):.3f}" if r['h'] else '-', 'vln_pe GT ep': r['pe_ep'] if r['pe_ep'] is not None else '-'} for r in ok]
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': [q1, q2v >= 0.95, q3, q4, q5], 'rows': rows}, open(log / 'summary.json', 'w'), indent=1, default=float)
    lead = (f'MP3D 4 씬 nav3d sg 경로 {len(rows)} ep 를 노은역과 같은 방식(mesh 바닥 · 06p)으로 vln_pe 포맷 렌더. 카메라 = vln_pe H1 값 (06v). '
            f'<b>판정 {sum([q1, q2v >= 0.95, q3, q4, q5])}/5 통과.</b> 시작 heading = GT start_rotation (R2R 셋업).')
    body = R.header('nav3d 06q · vln_pe 판 05b (MP3D sg)', 'vln_pe 생성 경로 렌더 — MP3D sg', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다.</p>' + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('그림', ''.join(R.figure(f, c) for f, c in figs))
    body += R.section('에피소드 별', R.table(tab, list(tab[0].keys())) if tab else '-')
    R.write(log, 'vln_pe 생성 경로 렌더', body)
    print('gates:', [f"{g['검증']} {'PASS' if p else 'FAIL'}" for g, p in zip(gates, [q1, q2v >= 0.95, q3, q5, q4])])
    for g in gates:
        print(' ', g['검증'], '|', g['값'])


def make_figs(log, ok, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = []
    pick = [r for r in ok if r['pe_ep'] is not None][:4]
    for r in pick:
        o = np.load(Path(args.out_root) / r['scene'] / 'videos' / 'chunk-000' / 'observation.images.rgb' / f"episode_{[e['ce_ep'] for e in json.load(open(log / ('paths_' + r['scene'] + '.json')))['episodes']].index(r['ep']):06d}.npy", mmap_mode='r')
        g = np.load(Path(args.gt_root) / r['scene'] / 'videos' / 'chunk-000' / 'observation.images.rgb' / f"episode_{r['pe_ep']:06d}.npy", mmap_mode='r')
        fig, ax = plt.subplots(2, 3, figsize=(7.5, 5.2))
        for c, q in enumerate((0, 0.5, 1)):
            ax[0, c].imshow(g[int(q * (len(g) - 1))])
            ax[0, c].set_title(f'vln_pe GT f{int(q * (len(g) - 1))}', fontsize=7)
            ax[1, c].imshow(o[int(q * (len(o) - 1))])
            ax[1, c].set_title(f'ours (sg) f{int(q * (len(o) - 1))}', fontsize=7)
        for a in ax.ravel():
            a.axis('off')
        fig.tight_layout()
        fn = f"cmp_{r['scene']}_{r['ep']}.jpg"
        fig.savefig(log / fn, dpi=75)
        plt.close(fig)
        figs.append((fn, f"{r['scene']} ep{r['ep']} — 위: vln_pe GT (H1), 아래: 우리 sg 경로 렌더. 시작·중간·끝 (경로가 달라 중간은 다를 수 있다)"))
    return figs


def main():
    args = parse()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'check':
        check(args)
    else:
        assert False, f'unreachable stage={args.stage!r}'


if __name__ == '__main__':
    main()
