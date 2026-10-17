"""06v — 노은역 렌더 방식(06)을 **vln_pe GT**(같은 MP3D 장면, Isaac 렌더, H1 로봇 카메라)로 검증한다.

06 이 새로 넣은 부분 = 경로(점마다 navmesh 바닥 z) -> 이산화 -> 프레임별 몸 바닥 z -> 카메라 자세(앞으로 띄움·높이·pitch) -> Isaac 렌더.
렌더러 자체는 2026-09 에 vln_pe GT 자세 재생으로 검증됐다(61 씬, SSIM 중앙 0.83). 여기서는 **06 의 자세 계산과 그 결과 이미지**를 본다.

    vln_pe GT 에피소드 (로봇 위치 · 카메라 자세 · 256×256 이미지)
      ├─ 경로 'gt' : GT 로봇 xy + navmesh 바닥 z   -> 06 episode_poses (같은 함수)  -> 우리 카메라
      ├─ 경로 'wp' : nav3d 03 wp 경로 (노은역 04 와 같은 종류)  -> 06 episode_poses -> 우리 카메라
      └─ GT 카메라 자세 그대로                                   -> 같은 렌더러 (기준선)
    -> 숫자 판정 (자세) + Isaac 렌더 (06v_render_isaac.py) -> 이미지 판정 + 리포트

카메라 값(높이·앞 거리·pitch)은 GT 에서 잰다: 06 이 D455 값을 넣는 자리에 vln_pe 값을 넣는다.

## 용어
- **짝 프레임**: 우리 프레임마다 xy 가 가장 가까운 GT 프레임 중 yaw 차가 가장 작은 것(0.3 m 안). 이산 action 이 달라 프레임 번호로는 못 맞춘다.
- **카메라 높이 차**: 짝 프레임끼리 카메라 z 차 (우리 − GT). 바닥이 같으므로 = 바닥 위 높이 차.
- **mesh 바닥 위 높이**: 카메라에서 아래로 쏜 광선이 장면 mesh 에 닿는 거리. navmesh 가 mesh 보다 떠 있으면 우리 쪽이 커진다(06b R4).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 |
|---|---|---|
| V1 위치 (gt 경로) | 06 이 만든 카메라가 GT 카메라 자리에 있나 | 짝 거리 중앙 ≤ 0.10 m · p95 ≤ 0.25 m |
| V2 높이 | 카메라 높이가 GT 와 같은가 (navmesh z 를 쓴 영향 포함) | gt·wp 각각 |중앙 차| ≤ 0.03 m · |차| p95 ≤ 0.08 m |
| V3 방향 | 같은 쪽을 보나 | 짝 |Δyaw| 중앙 ≤ 5° · |Δpitch| 중앙 ≤ 3° |
| V4 계단 | 계단 에피소드에서도 V2 | 계단 ep 의 |차| p95 ≤ 0.10 m |
| V5 이미지 | 06 자세 렌더가 GT 자세 렌더만큼 GT 이미지와 닮았나 (같은 세션 렌더) | 짝 SSIM 중앙 ≥ 기준선 SSIM 중앙 − 0.03 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06v_vlnpe_check.py --stage prepare
    (isaac) /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06v_render_isaac.py
    /usr/bin/python scripts/dataset_converters/nav3d/06v_vlnpe_check.py --stage report
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
GS = HERE.parent / 'gs_vlnpe'
for p in (str(HERE), str(GS), str(GS / 'apply_real'), str(HERE.parent / 'instruction_labeling' / 'render')):
    if p not in sys.path:
        sys.path.insert(0, p)
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from floor_mesh import mesh_floor_z  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['prepare', 'report'])
    ap.add_argument('--scenes', default='17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX,1pXnuDYAj8r', help='05 와 같은 4 씬')
    ap.add_argument('--gt_root', default=f'{D}/vln_pe/traj_data/r2r')
    ap.add_argument('--scene_root', default=f'{D}/scene_data/mp3d_ce/mp3d', help='navmesh · glb (habitat 좌표)')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/03_refine_paths/scenes')
    ap.add_argument('--max_eps', type=int, default=8, help='씬당 평지 에피소드 상한 (계단 에피소드는 전부)')
    ap.add_argument('--stairs_scenes', default='2n8kARJN3HM,b8cTxDM8gDG,uNb9QFRL6hY',
                    help='계단 에피소드만 더 가져올 씬 (vln_pe 2,813 ep 중 로봇 z 범위 > 0.5 m 는 이 3 씬의 3 ep 뿐)')
    ap.add_argument('--stairs_dz', type=float, default=0.5, help='GT 로봇 z 범위가 이보다 크면 계단 에피소드')
    ap.add_argument('--waypoint_spacing_m', type=float, default=0.5, help='06 과 같게')
    ap.add_argument('--pair_r', type=float, default=0.3, help='짝 프레임 찾는 반경')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/06v_vlnpe')
    return ap.parse_args()


def load06():
    sp = importlib.util.spec_from_file_location('m06', HERE / '06_render_noeun_vlnce.py')
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def yaw_of(P):
    f = P[:, :3, 2]
    return np.arctan2(f[:, 1], f[:, 0])


def pitch_of(P):
    return np.degrees(np.arcsin(np.clip(-P[:, 2, 2], -1, 1)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pairs(ours: np.ndarray, gt: np.ndarray, r: float) -> np.ndarray:
    """우리 프레임 -> 짝 GT 프레임 번호 (-1 = 없음): r 안에서 |Δyaw| 최소."""
    oy, gy = yaw_of(ours), yaw_of(gt)
    d = np.linalg.norm(ours[:, None, :2, 3] - gt[None, :, :2, 3], axis=2)
    out = np.full(len(ours), -1)
    for i in range(len(ours)):
        c = np.where(d[i] <= r)[0]
        if len(c):
            out[i] = c[np.argmin(np.abs(wrap(gy[c] - oy[i])))]
    return out


def dedupe(xy: np.ndarray, z: np.ndarray, tol=0.01):
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > tol:
            keep.append(i)
    return xy[keep], z[keep]


def prepare(args):
    import pyarrow.parquet as pq
    import trimesh
    import dataset_utils as du
    from release_io import load_cfg, match_raw, release_episodes
    m06 = load06()
    rc = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], {}
    rig_all = []
    stairs_only = [x for x in args.stairs_scenes.split(',') if x]
    for s in args.scenes.split(',') + stairs_only:
        pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
        tris, _ = nu.navmesh_triangles_islands(pf)
        surf = nu.NavSurface(tris)
        mesh = trimesh.load(f'{args.scene_root}/{s}/{s}.glb', force='mesh')
        # MP3D glb 파일 자체가 z-up = nav3d·Isaac 좌표 (habitat 이 올릴 때만 y-up 으로 돌린다; 실측 정점 z −0.13~2.68 m)
        Z = np.load(Path(args.nav03_dir) / f'{s}_traj.npz')
        ce = {}
        for e in release_episodes(rc, s):
            r = match_raw(rc, s, e['task'])
            if r:
                ce[r['episode_id']] = e['episode_index']
        pe = [json.loads(l) for l in open(Path(args.gt_root) / s / 'meta' / 'episodes.jsonl')]
        flat_n, chosen = 0, []
        for p in pe:
            ei = p['episode_index']
            t = pq.read_table(Path(args.gt_root) / s / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet',
                              columns=['observation.robot_position', 'observation.robot_yaw']).to_pydict()
            rob = np.asarray(t['observation.robot_position'])
            stairs = float(np.ptp(rob[:, 2])) > args.stairs_dz
            r = match_raw(rc, s, p['tasks'][0])
            ci = ce.get(r['episode_id']) if r else None
            has_wp = ci is not None and f'{ci}_wp_ref' in Z.files
            if stairs or (s not in stairs_only and (has_wp or flat_n < args.max_eps)):
                chosen.append((ei, stairs, ci if has_wp else None, rob, np.asarray(t['observation.robot_yaw'])))
                flat_n += not stairs
        jobs[s] = {'gt': {}, 'ours_gt': {}, 'ours_wp': {}, 'ours_gt_mesh': {}, 'ours_wp_mesh': {}}
        for ei, stairs, ci, rob, ryaw in chosen:
            g = du.load_gt_episode(args.gt_root, s, ei, 'vln_pe')
            G = g['poses_c2w']
            z0 = float(rob[0, 2] - du.VLNPE_TORSO_STANDING_M)
            fz = surf.track(rob[:, :2], z0, max_step_dz=0.5)
            if np.isnan(fz).any():
                ok = ~np.isnan(fz)
                fz = np.interp(np.arange(len(fz)), np.where(ok)[0], fz[ok])
            # GT 카메라 값: 바닥(navmesh) 위 높이, 로봇 방향으로 앞선 거리, pitch
            fwd = np.c_[np.cos(ryaw), np.sin(ryaw)]
            rig = {'h': float(np.median(G[:, 2, 3] - fz)), 'fwd': float(np.median(((G[:, :2, 3] - rob[:, :2]) * fwd).sum(1))),
                   'pitch': float(np.median(pitch_of(G)))}
            rig_all.append(rig)
            jobs[s]['gt'][str(ei)] = G
            rows.append({'scene': s, 'ep': ei, 'stairs': stairs, 'wp_ci': ci, 'gt_frames': len(G), 'rig': rig,
                         'rob': rob, 'fz': fz, 'G': G, 'mesh': None})
        # 카메라 값 = 이 씬까지 모은 GT 중앙 (06 의 D455 자리)
        rows_s = [r for r in rows if r['scene'] == s]
        for r in rows_s:
            r['mesh_h_gt'] = mesh_height(mesh, r['G'])
        print(f'[06v] {s}: {len(rows_s)} ep (계단 {sum(r["stairs"] for r in rows_s)}, wp {sum(r["wp_ci"] is not None for r in rows_s)})', flush=True)
        for r in rows_s:
            r['_mesh'] = mesh
            r['_Z'] = Z
    H = float(np.median([r['h'] for r in rig_all]))
    FW = float(np.median([r['fwd'] for r in rig_all]))
    PI = float(np.median([r['pitch'] for r in rig_all]))
    HM = float(np.nanmedian(np.concatenate([r['mesh_h_gt'] for r in rows])))
    profs = {'navmesh': SimpleNamespace(mount_height_m=H, pitch_down_deg=PI), 'mesh': SimpleNamespace(mount_height_m=HM, pitch_down_deg=PI)}
    print(f'[06v] GT 카메라 값 (06 에 넣는 값): 높이 navmesh 위 {H:.3f} / mesh 위 {HM:.3f} m · 앞 {FW:.3f} m · pitch {PI:.2f}°', flush=True)
    res = []
    for r in rows:
        s, ei = r['scene'], r['ep']
        xy, z = dedupe(r['rob'][:, :2], r['fz'])
        T = nu.densify(np.c_[xy, z], 0.05)
        paths = {'gt': (T, np.zeros(len(T), bool))}
        if r['wp_ci'] is not None:
            W = r['_Z'][f"{r['wp_ci']}_wp_ref"].astype(np.float64)
            st = r['_Z'][f"{r['wp_ci']}_wp_ref_st"].astype(bool) if f"{r['wp_ci']}_wp_ref_st" in r['_Z'].files else np.zeros(len(W), bool)
            paths['wp'] = (W, st)
        for base, (traj0, st) in list(paths.items()):   # 바닥 z: navmesh(06 지금) / mesh (floor_mesh.py, 04b)
            fm = mesh_floor_z(traj0, st, r['_mesh'])
            paths[base] = (traj0, 'navmesh')
            paths[f'{base}_mesh'] = (np.c_[traj0[:, :2], fm['z']], 'mesh')
        for mode, (traj, floor) in paths.items():
            disc, zb, P = m06.episode_poses(traj, profs[floor], FW, args.waypoint_spacing_m)
            k = pairs(P, r['G'], args.pair_r)
            v = k >= 0
            Gm = r['G'][k[v]]
            d = np.linalg.norm(P[v, :2, 3] - Gm[:, :2, 3], axis=1)
            dz = P[v, 2, 3] - Gm[:, 2, 3]
            dyaw = np.degrees(np.abs(wrap(yaw_of(P[v]) - yaw_of(Gm))))
            dpit = np.abs(pitch_of(P[v]) - pitch_of(Gm))
            mh = mesh_height(r['_mesh'], P)
            jobs[s][f'ours_{mode}'][str(ei)] = P
            res.append({'scene': s, 'ep': ei, 'mode': mode, 'floor': floor, 'stairs': r['stairs'], 'frames': len(P), 'gt_frames': r['gt_frames'],
                        'paired_frac': float(v.mean()), 'pair_idx': k.tolist(), 'd': d.tolist(), 'dz': dz.tolist(), 'dyaw': dyaw.tolist(),
                        'dpitch': dpit.tolist(), 'mesh_h_ours': mh.tolist(), 'mesh_h_gt': r['mesh_h_gt'].tolist(), 'rig': r['rig'],
                        'n_forward': int(disc['n_forward']), 'n_turn': int(disc['n_turn']), 'end_gap_m': float(disc['end_gap_m']),
                        'gt_cam_xyz': r['G'][:, :3, 3].round(4).tolist(), 'ours_cam_xyz': P[:, :3, 3].round(4).tolist()})
            print(f'  {s} ep{ei:>3} {mode:7s} {"계단" if r["stairs"] else "평지"} {len(P):>4}f 짝 {v.mean():.2f} '
                  f'거리 {np.median(d):.3f} Δz {np.median(dz):+.3f} Δyaw {np.median(dyaw):.1f}', flush=True)
    np.savez_compressed(out / 'jobs.npz', **{f'{s}|{kind}|{ei}': P for s, J in jobs.items() for kind, E in J.items() for ei, P in E.items()})
    json.dump({'rig': {'h': H, 'h_mesh': HM, 'fwd': FW, 'pitch': PI, 'K': du.VLNPE_K.tolist(), 'wh': list(du.VLNPE_IMG_WH)}, 'args': vars(args), 'episodes': res},
              open(out / 'prepare.json', 'w'))
    print(f'[06v] -> {out}/prepare.json · jobs.npz ({len(res)} 경로)')


def mesh_height(mesh, P: np.ndarray) -> np.ndarray:
    """카메라에서 아래로 mesh 까지 거리 (못 맞추면 nan)."""
    loc, idx, _ = mesh.ray.intersects_location(P[:, :3, 3], np.tile([0, 0, -1.0], (len(P), 1)), multiple_hits=False)
    h = np.full(len(P), np.nan)
    h[idx] = P[idx, 2, 3] - loc[:, 2]
    return h


def report(args):
    from skimage.metrics import structural_similarity as ssim
    out = Path(args.out_dir)
    pr = json.load(open(out / 'prepare.json'))
    rig, E = pr['rig'], pr['episodes']
    rend = out / 'render'
    gt_root = Path(args.gt_root)
    for e in E:                      # 이미지: 짝 프레임에서 SSIM (우리 렌더 vs GT 이미지) · 기준선 (GT 자세 렌더 vs GT 이미지)
        f_o, f_b = rend / f"{e['scene']}__ours_{e['mode']}__{e['ep']}.npz", rend / f"{e['scene']}__gt__{e['ep']}.npz"
        if not (f_o.exists() and f_b.exists()):
            e['ssim'] = None
            continue
        o, b = np.load(f_o)['rgb'], np.load(f_b)['rgb']
        g = np.load(gt_root / e['scene'] / 'videos' / 'chunk-000' / 'observation.images.rgb' / f"episode_{e['ep']:06d}.npy")
        k = np.array(e['pair_idx'])
        gray = lambda a: a.mean(-1)  # noqa: E731
        e['ssim'] = [float(ssim(gray(o[i]), gray(g[j]), data_range=255)) for i, j in enumerate(k) if j >= 0]
        e['ssim_base'] = [float(ssim(gray(b[j]), gray(g[j]), data_range=255)) for j in sorted(set(k[k >= 0].tolist()))]
    cat = lambda key, sel: np.concatenate([np.asarray(e[key], float) for e in E if sel(e)] or [np.array([np.nan])])  # noqa: E731
    has_img = any(e.get('ssim') for e in E)

    def gates_for(floor: str) -> tuple:
        """floor = navmesh (06 지금) | mesh (floor_mesh.py 로 고친 것). 같은 기준."""
        fl = lambda e: e['floor'] == floor  # noqa: E731
        d_gt = cat('d', lambda e: fl(e) and e['mode'].startswith('gt'))
        v1 = bool(np.median(d_gt) <= 0.10 and np.percentile(d_gt, 95) <= 0.25)
        dz = {m: cat('dz', (lambda e, m=m: fl(e) and e['mode'].split('_')[0] == m)) for m in ('gt', 'wp')}
        v2 = all(abs(np.nanmedian(dz[m])) <= 0.03 and np.nanpercentile(np.abs(dz[m]), 95) <= 0.08 for m in dz)
        dy, dp = cat('dyaw', fl), cat('dpitch', fl)
        v3 = bool(np.median(dy) <= 5 and np.median(dp) <= 3)
        dzs = cat('dz', lambda e: fl(e) and e['stairs'])
        has_st = bool(np.isfinite(dzs).any())
        v4 = bool(has_st and np.nanpercentile(np.abs(dzs), 95) <= 0.10)
        so = cat('ssim', lambda e: fl(e) and bool(e.get('ssim'))) if has_img else np.array([np.nan])
        sb = cat('ssim_base', lambda e: fl(e) and bool(e.get('ssim'))) if has_img else np.array([np.nan])
        v5 = bool(has_img and np.median(so) >= np.median(sb) - 0.03)
        mho = cat('mesh_h_ours', fl)
        tag = f' [{floor}]'
        g = [
            {'검증': 'V1 위치 (gt 경로)' + tag, '결과': R.pill(v1), '값': f'짝 거리 중앙 {np.median(d_gt):.3f} · p95 {np.percentile(d_gt, 95):.3f} m', '기준': '≤ 0.10 · ≤ 0.25 m'},
            {'검증': 'V2 높이' + tag, '결과': R.pill(v2), '값': ' · '.join(f'{m}: 중앙 {np.nanmedian(dz[m]):+.3f} · |p95| {np.nanpercentile(np.abs(dz[m]), 95):.3f} m' for m in dz), '기준': '|중앙| ≤ 0.03 · p95 ≤ 0.08 m'},
            {'검증': 'V3 방향' + tag, '결과': R.pill(v3), '값': f'|Δyaw| 중앙 {np.median(dy):.1f}° · |Δpitch| 중앙 {np.median(dp):.1f}°', '기준': '≤ 5° · ≤ 3°'},
            {'검증': 'V4 계단' + tag, '결과': R.pill(v4) if has_st else R.pill('계단 없음'), '값': f'|Δz| p95 {np.nanpercentile(np.abs(dzs), 95):.3f} m' if has_st else '이 씬들의 vln_pe 에 계단 에피소드 없음', '기준': '≤ 0.10 m'},
            {'검증': 'V5 이미지' + tag, '결과': R.pill(v5) if has_img else R.pill('렌더 전'), '값': f'SSIM 중앙 우리 {np.median(so):.3f} vs GT 자세 렌더 {np.median(sb):.3f}' if has_img else '-', '기준': '≥ 기준선 − 0.03'},
        ]
        return g, [v1, v2, v3, v4, v5], float(np.nanmedian(mho))

    g_nav, p_nav, mh_nav = gates_for('navmesh')
    g_mesh, p_mesh, mh_mesh = gates_for('mesh')
    gates = g_nav + g_mesh
    mhg = cat('mesh_h_gt', lambda e: e['floor'] == 'navmesh')
    tab = [{'scene': e['scene'], 'ep': e['ep'], '경로': e['mode'], '바닥 z': e['floor'], '종류': '계단' if e['stairs'] else '평지', '프레임 (GT)': f"{e['frames']} ({e['gt_frames']})",
            '짝 비율': f"{e['paired_frac']:.2f}", '짝 거리 중앙': f"{np.median(e['d']):.3f}" if e['d'] else '-',
            'Δz 중앙': f"{np.median(e['dz']):+.3f}" if e['dz'] else '-', 'mesh 위 높이 우리/GT': f"{np.nanmedian(e['mesh_h_ours']):.3f} / {np.nanmedian(e['mesh_h_gt']):.3f}",
            '|Δyaw|': f"{np.median(e['dyaw']):.1f}" if e['dyaw'] else '-', 'SSIM 우리/기준': f"{np.median(e['ssim']):.3f} / {np.median(e['ssim_base']):.3f}" if e.get('ssim') else '-'} for e in E]
    figs = make_figs(out, E, args, rend, gt_root)
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': {'navmesh': p_nav, 'mesh': p_mesh}, 'rig': rig,
               'mesh_h': {'ours_navmesh': mh_nav, 'ours_mesh': mh_mesh, 'gt': float(np.nanmedian(mhg))}},
              open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False)
    lead = (f'06 이 쓰는 함수(<span class="mono">episode_poses</span>: 이산화 → 몸 바닥 z → 카메라 자세)에 vln_pe 카메라 값을 넣어 MP3D {len(set(e["scene"] for e in E))} 씬 '
            f'{len(set((e["scene"], e["ep"]) for e in E))} 에피소드(계단 {len(set((e["scene"], e["ep"]) for e in E if e["stairs"]))})를 만들고 vln_pe GT 와 비교했다. '
            f'카메라 값(GT 에서 잼): 높이 navmesh 위 {rig["h"]:.3f} / mesh 위 {rig["h_mesh"]:.3f} m · 앞 {rig["fwd"]:.3f} m · pitch {rig["pitch"]:.1f}°. '
            f'mesh 바닥 위 카메라 높이 중앙: GT {np.nanmedian(mhg):.3f} · 06 지금(navmesh z) {mh_nav:.3f} · 고친 것(mesh z) {mh_mesh:.3f} m. '
            f'<b>판정 navmesh {sum(p_nav)}/5 · mesh {sum(p_mesh)}/5 통과.</b>')
    body = R.header('nav3d 06v · vln_pe GT 로 06 방식 검증 (Isaac)', '06 방식 검증 — vln_pe GT', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다. 짝 프레임 = 우리 프레임마다 0.3 m 안의 GT 프레임 중 yaw 차가 가장 작은 것. '
                      '바닥 z: navmesh = 06 지금 방식, mesh = floor_mesh.py 로 장면 mesh 바닥에 맞춘 것(계단은 navmesh 경사 + 양 끝 차이 보간). gt 경로 = GT 로봇 xy, wp 경로 = nav3d 03 wp (노은역 04 와 같은 종류). 이미지 기준선 = GT 카메라 자세를 같은 세션·같은 렌더러로 그린 것.</p>'
                      + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('에피소드 별', R.table(tab, list(tab[0].keys())))
    body += R.section('그림', ''.join(R.figure(f, c) for f, c in figs))
    R.write(out, '06 방식 검증 vln_pe', body)
    print('gates:', [f"{g['검증']} {'PASS' if p else 'FAIL'}" for g, p in zip(gates, p_nav + p_mesh)])
    print(f'-> {out}/report.html')


def make_figs(out, E, args, rend, gt_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = []
    pick = [e for e in E if e['stairs']][:3] + [e for e in E if not e['stairs'] and e['mode'] == 'wp'][:2]
    for e in pick:
        G, O = np.array(e['gt_cam_xyz']), np.array(e['ours_cam_xyz'])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(G[:, 0], G[:, 1], 'k.-', ms=2, lw=0.8, label='GT camera')
        ax[0].plot(O[:, 0], O[:, 1], '.-', color='#2b5fa8', ms=2, lw=0.8, label=f'ours ({e["mode"]})')
        ax[0].set_aspect('equal')
        ax[0].legend(fontsize=7)
        ax[1].plot(np.array(e['mesh_h_gt']), 'k', lw=1, label='GT: camera height above mesh')
        ax[1].plot(np.linspace(0, len(G) - 1, len(O)), np.array(e['mesh_h_ours']), color='#2b5fa8', lw=1, label='ours')
        ax[1].set_xlabel('GT frame (ours stretched to same length)')
        ax[1].legend(fontsize=7)
        fig.suptitle(f"{e['scene']} ep{e['ep']} {e['mode']} {'stairs' if e['stairs'] else 'flat'}")
        fig.tight_layout()
        fn = f"path_{e['scene']}_{e['ep']}_{e['mode']}.png"
        fig.savefig(out / fn, dpi=70)
        plt.close(fig)
        figs.append((fn, f"{e['scene']} ep{e['ep']} ({e['mode']}, {'계단' if e['stairs'] else '평지'}) — 왼쪽: 위에서 본 카메라 위치 (검정 GT, 파랑 우리) · 오른쪽: mesh 바닥 위 카메라 높이"))
        f_o, f_b = rend / f"{e['scene']}__ours_{e['mode']}__{e['ep']}.npz", rend / f"{e['scene']}__gt__{e['ep']}.npz"
        if f_o.exists() and f_b.exists():
            o, b = np.load(f_o)['rgb'], np.load(f_b)['rgb']
            g = np.load(gt_root / e['scene'] / 'videos' / 'chunk-000' / 'observation.images.rgb' / f"episode_{e['ep']:06d}.npy")
            k = np.array(e['pair_idx'])
            ii = [i for i in np.linspace(0, len(k) - 1, 6).astype(int) if k[i] >= 0]
            fig, ax = plt.subplots(3, len(ii), figsize=(2.2 * len(ii), 6.8), squeeze=False)
            for c, i in enumerate(ii):
                for r_, (im, t) in enumerate(((g[k[i]], f'GT image f{k[i]}'), (b[k[i]], 'GT pose, our render'), (o[i], f'ours f{i}'))):
                    ax[r_, c].imshow(im)
                    ax[r_, c].set_title(t, fontsize=7)
                    ax[r_, c].axis('off')
            fig.tight_layout()
            fn = f"img_{e['scene']}_{e['ep']}_{e['mode']}.jpg"
            fig.savefig(out / fn, dpi=70)
            plt.close(fig)
            figs.append((fn, f"{e['scene']} ep{e['ep']} ({e['mode']}) — 위: vln_pe GT 이미지 · 가운데: GT 자세를 우리 렌더러로 · 아래: 06 방식 자세 (짝 프레임)"))
    return figs


def main():
    args = parse()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'report':
        report(args)
    else:
        assert False, f'unreachable stage={args.stage!r}'


if __name__ == '__main__':
    main()
