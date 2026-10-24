"""02 — vln_ce GT 와 직접 비교해 **좌표·지도·경로 계획**이 맞는지 확인한다 (계단 에피소드 포함).

vln_ce GT 는 habitat **공식 MP3D navmesh** 위에서 만들어졌다: R2R `reference_path`(사람이 고른 경유점)를
최단경로 추종기(GreedyGeodesicFollower — 전진 0.25 m / 회전 15° 이산 action)로 이은 것이다. 그래서 같은 지도·같은
도구로 다시 만들면 GT 가 나와야 한다. 단계마다 하나씩만 바꿔 어디서 어긋나는지 가른다.

| 검사 | 묻는 것 | 무엇을 비교 |
|---|---|---|
| A 좌표 | GT 궤적(우리 좌표로 올린 것)이 공식 navmesh 표면 위에 있나 | GT 발 위치 → navmesh 수직 투영: 투영 성공률, 높이 차 |
| A2 시작점 | 우리 좌표 변환이 raw 의 habitat 시작점과 같나 | ‖hab(GT[0]) − start_position‖ |
| B 지도+계획 | 공식 navmesh `find_path` 거리 = 데이터셋이 적어 둔 `info.geodesic_distance` 인가 | 절대차 |
| C GT 재현 | reference_path 를 추종기로 이으면 GT 궤적·action 이 나오나 | GT 점 → 추종기 경로 거리, action 열 일치율 |
| C2 경유점+계획 | reference_path 이웃 경유점끼리 `find_path` 로 이으면 GT 와 같은 길인가 (경유점을 줄 때의 nav3d) | GT 점 → 경로 거리, 같은 루트 |
| D nav3d 계획 | start→goal 만 주고 `find_path` 한 경로가 GT 와 같은 길인가 (nav3d 가 쓸 방식) | Fréchet(xy) < 0.8 m = 같은 루트, 길이비, 높이 범위 |
| E 우리 빌더 | glb 에서 우리가 다시 만든 navmesh 가 공식본과 같은가 (`--recompute`) | 면적비, B·D 를 다시 |

"같은 루트" = 두 경로의 이산 Fréchet 거리(순서를 지키며 짝지은 점 사이 최대 거리) < 0.8 m (문 폭) — gs_vlnpe 03 과 같은 정의.
층 이동 에피소드 = GT 시작·끝 높이 차 > 0.5 m (gs_vlnpe 03u 와 같은 정의).

사용 (habitat 컨테이너, 저장소 루트):
    # 씬 하나
    /usr/bin/python scripts/dataset_converters/nav3d/common/compare_vlnce_gt.py --scene 17DRP5sb8fy
    # 전체 (씬별 프로세스, 동시 4개) 후 집계
    ls data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r | xargs -P 4 -I{} /usr/bin/python scripts/dataset_converters/nav3d/common/compare_vlnce_gt.py --scene {} --recompute
    /usr/bin/python scripts/dataset_converters/nav3d/common/compare_vlnce_gt.py --aggregate
"""

import argparse
import difflib
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GS))
import nav3d_utils as nu  # noqa: E402
import vlnce_gt  # noqa: E402  (gs_vlnpe — 읽기 전용으로 import)
from esdf_utils import discrete_frechet_distance  # noqa: E402

DOOR_WIDTH_M = 0.8           # 같은 루트 판정 (gs_vlnpe 03 DOOR_WIDTH_M)
MULTIFLOOR_DZ_M = 0.5        # 층 이동 판정 (gs_vlnpe 03u --dz_m)
ACT = {0: 'S', 1: 'F', 2: 'L', 3: 'R'}   # VLN-CE: 0 STOP · 1 MOVE_FORWARD · 2 TURN_LEFT · 3 TURN_RIGHT


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--aggregate', action='store_true', help='씬별 결과를 모아 요약·그림·report.html')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default=vlnce_gt.DEFAULT_RAW_ROOT)
    ap.add_argument('--scene_root', default='data/scene_data/mp3d_ce/mp3d', help='<scene>/<scene>.{glb,navmesh}')
    ap.add_argument('--rig', default='125cm_30deg', help='GT 카메라 rig — 높이(h_b)만 쓴다')
    ap.add_argument('--goal_radius', type=float, default=0.25, help='추종기 도달 반경 (생성값 미상 — 3dloader 04 와 같게)')
    ap.add_argument('--recompute', action='store_true', help='E: glb 에서 habitat 기본 설정으로 navmesh 를 다시 만들어 비교')
    ap.add_argument('--num_episodes', type=int, default=0, help='0 = 전부')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/common/compare_vlnce_gt')
    return ap.parse_args()


# ---------------------------------------------------------------------------
def follower_rollout(pf, raw, goal_radius, fwd_m=0.25, turn_deg=15.0):
    """reference_path 를 이산 action 으로 잇는다 (3dloader_vlnce/04_calibrate_map.follower_path 와 같은 구성).

    ⚠️ `move_filter_fn = pf.try_step` 이 없으면 벽을 통과한다(3dloader 실측). -> (habitat 위치 (N,3), action 문자열)
    """
    import habitat_sim
    from habitat_sim.agent import Agent, AgentConfiguration, AgentState
    spec, act = habitat_sim.ActionSpec, habitat_sim.ActuationSpec
    sg = habitat_sim.SceneGraph()
    agent = Agent(sg.get_root_node().create_child(), AgentConfiguration(action_space={
        0: spec('stop'), 1: spec('move_forward', act(amount=fwd_m)),
        2: spec('turn_left', act(amount=turn_deg)), 3: spec('turn_right', act(amount=turn_deg))}))
    agent.controls.move_filter_fn = pf.try_step
    fol = habitat_sim.GreedyGeodesicFollower(pf, agent, goal_radius, stop_key=0, forward_key=1, left_key=2, right_key=3)
    agent.set_state(AgentState(position=np.asarray(raw['start_position'], dtype=np.float32),
                               rotation=np.asarray(raw['start_rotation'], dtype=np.float32)))
    pos, acts, ok = [np.asarray(agent.state.position, dtype=np.float64)], [], True
    for goal in np.asarray(raw['reference_path'], dtype=np.float64)[1:]:
        try:
            seq = fol.find_path(goal)
        except Exception:  # noqa: BLE001 — habitat 이 도달 불가면 예외
            ok = False
            continue
        for a in seq:
            if a is None or a == 0:
                continue
            agent.act(a)
            acts.append(ACT[a])
            pos.append(np.asarray(agent.state.position, dtype=np.float64))
    del fol, agent, sg
    return np.stack(pos), ''.join(acts), ok


def shortest(pf, a_hab, b_hab):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = np.asarray(a_hab, np.float32), np.asarray(b_hab, np.float32)
    if not pf.find_path(sp):
        return None, float('nan')
    return np.asarray(sp.points, dtype=np.float64), float(sp.geodesic_distance)


def point_to_polyline(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """각 p 에서 꺾은선 poly 까지 최소 3D 거리."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(nu.densify(poly, 0.02) if len(poly) > 1 else poly).query(p)
    return d


def plan_metrics(S: np.ndarray, gt: np.ndarray) -> dict:
    L = lambda X: float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())
    fr = discrete_frechet_distance(S[:, :2], gt[:, :2])
    return {'frechet_xy_m': fr, 'same_route': bool(fr < DOOR_WIDTH_M), 'len_ratio': L(S) / max(L(gt), 1e-6),
            'z_range_err_m': float(max(abs(S[:, 2].max() - gt[:, 2].max()), abs(S[:, 2].min() - gt[:, 2].min())))}


def run_scene(args):
    s = args.scene
    out = Path(args.out_dir) / 'scenes'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pf = nu.load_pathfinder(f'{args.scene_root}/{s}/{s}.navmesh')
    tris, _ = nu.navmesh_triangles_islands(pf)
    surf = nu.NavSurface(tris)
    scene_row = {'scene': s, 'official_area_m2': float(pf.navigable_area), 'official_islands': int(pf.num_islands)}
    pf2 = surf2 = None
    if args.recompute:
        cfg = {}   # habitat 기본값 (공식본이 기본값으로 만들어졌다 — 3dloader 03 확인)
        tmp = out / f'{s}_recomputed.navmesh'
        pf2 = nu.build_navmesh(f'{args.scene_root}/{s}/{s}.glb', cfg, str(tmp))
        scene_row.update(recomputed_area_m2=float(pf2.navigable_area), recomputed_islands=int(pf2.num_islands))
    rawidx = vlnce_gt.raw_index(args.raw_root)
    h_b = vlnce_gt.vlnce_rig.rig_height_m(args.rig)
    n_ep = len(glob.glob(f'{args.data_root}/{s}/data/chunk-000/*.parquet'))
    if args.num_episodes:
        n_ep = min(n_ep, args.num_episodes)
    rows, arrays = [], {}
    for ep in range(n_ep):
        row = {'scene': s, 'ep': ep}
        try:
            g = vlnce_gt.load_gt_episode_vlnce(args.data_root, s, ep, args.rig, args.raw_root)
            raw = rawidx[(s, vlnce_gt.episode_instruction(args.data_root, s, ep))]
        except KeyError as e:
            row['error'] = str(e)
            rows.append(row)
            continue
        gt = g['cam_xyz'] - [0.0, 0.0, h_b]                 # 발 위치 (우리 좌표 Z-up)
        acts_gt = ''.join(ACT.get(int(a), '?') for a in pq.read_table(
            f'{args.data_root}/{s}/data/chunk-000/episode_{ep:06d}.parquet', columns=['action'])['action'].to_pylist()
            if int(a) >= 0)
        row['multifloor'] = bool(abs(gt[-1, 2] - gt[0, 2]) > MULTIFLOOR_DZ_M)
        row['gt_dz_m'] = float(gt[-1, 2] - gt[0, 2])
        row['n_frames'] = int(len(gt))
        # A 좌표: GT 발 위치를 공식 navmesh 표면에 수직 투영 (연속성 추적)
        zt = surf.track(gt[:, :2], z0=gt[0, 2])
        ok = np.isfinite(zt)
        row['A_on_frac'] = float(ok.mean())
        row['A_dz_med_m'] = float(np.median(np.abs(zt[ok] - gt[ok, 2]))) if ok.any() else None
        row['A_dz_max_m'] = float(np.max(np.abs(zt[ok] - gt[ok, 2]))) if ok.any() else None
        # 투영 실패 점(삼각형 경계 위 등)은 navmesh 까지 3D 거리로 다시 본다 — 0.10 m 넘게 떨어진 점만 '진짜 밖'
        sd = np.zeros(len(gt))
        for i in np.where(~ok)[0]:
            q = np.asarray(pf.snap_point(nu.ours_to_hab(gt[i])))
            sd[i] = np.linalg.norm(nu.hab_to_ours(q) - gt[i]) if np.isfinite(q).all() else np.inf
        row['A_far_frac'] = float((sd > 0.10).mean())
        row['A_far_max_m'] = float(sd.max())
        # A2 시작점
        row['A2_start_err_m'] = float(np.linalg.norm(nu.ours_to_hab(gt[0]) - np.asarray(raw['start_position'])))
        # B 지도+계획: 공식 navmesh 최단거리 vs 데이터셋 기록
        goal_hab = np.asarray(raw['goals'][0]['position'])
        _, geo = shortest(pf, raw['start_position'], goal_hab)
        row['B_geo_m'], row['B_geo_info_m'] = geo, float(raw['info']['geodesic_distance'])
        row['B_geo_abs_err_m'] = abs(geo - row['B_geo_info_m'])
        # C GT 재현: reference_path -> 추종기
        fpos, facts, fok = follower_rollout(pf, raw, args.goal_radius)
        F = nu.hab_to_ours(fpos)
        d = point_to_polyline(gt, F)
        row.update(C_ok=fok, C_med_m=float(np.median(d)), C_p95_m=float(np.percentile(d, 95)), C_max_m=float(d.max()),
                   C_end_m=float(np.linalg.norm(F[-1] - gt[-1])),
                   C_act_equal=bool(facts == acts_gt.rstrip('S')),
                   C_act_ratio=float(difflib.SequenceMatcher(None, facts, acts_gt.rstrip('S'), autojunk=False).ratio()),
                   C_n_act=len(facts), C_n_act_gt=len(acts_gt))
        # C2 경유점 + 우리 계획: reference_path 의 이웃 경유점끼리 find_path 로 이어 붙인 경로 ↔ GT
        wps = np.asarray(raw['reference_path'], dtype=np.float64)
        legs, c2ok = [], True
        for wa, wb in zip(wps[:-1], wps[1:]):
            Pl, _ = shortest(pf, wa, wb)
            if Pl is None:
                c2ok = False
                continue
            legs.append(Pl if not legs else Pl[1:])
        if legs:
            W = nu.densify(nu.hab_to_ours(np.vstack(legs)), 0.05)
            zw = surf.track(W[:, :2], z0=W[0, 2])
            W[np.isfinite(zw), 2] = zw[np.isfinite(zw)]
            d2 = point_to_polyline(gt, W)
            row.update(C2_ok=c2ok, C2_med_m=float(np.median(d2)), C2_p95_m=float(np.percentile(d2, 95)),
                       **{f'C2_{k}': v for k, v in plan_metrics(W, gt).items()})
            arrays[f'{ep}_wp'] = W.astype(np.float32)
        else:
            row['C2_ok'] = False
        # D nav3d 계획: GT 시작·끝만 주고 공식 navmesh find_path
        P, geo_d = shortest(pf, nu.ours_to_hab(gt[0]), nu.ours_to_hab(gt[-1]))
        if P is None:
            row['D_found'] = False
        else:
            Pd = nu.densify(nu.hab_to_ours(P), 0.05)
            z = surf.track(Pd[:, :2], z0=Pd[0, 2])
            S = Pd.copy()
            S[np.isfinite(z), 2] = z[np.isfinite(z)]
            row.update(D_found=True, D_on_frac=float(np.isfinite(z).mean()), **{f'D_{k}': v for k, v in plan_metrics(S, gt).items()})
            arrays[f'{ep}_ours'] = S.astype(np.float32)
        if pf2 is not None:
            _, g2 = shortest(pf2, raw['start_position'], goal_hab)
            row['E_geo_abs_err_m'] = abs(g2 - row['B_geo_info_m'])
            P2, _ = shortest(pf2, nu.ours_to_hab(gt[0]), nu.ours_to_hab(gt[-1]))
            row['E_found'] = P2 is not None
            if P2 is not None:
                row['E_same_route'] = plan_metrics(nu.densify(nu.hab_to_ours(P2), 0.05), gt)['same_route']
        arrays[f'{ep}_gt'] = gt.astype(np.float32)
        arrays[f'{ep}_fol'] = F.astype(np.float32)
        rows.append(row)
    scene_row['sec'] = time.time() - t0
    json.dump({'scene': scene_row, 'episodes': rows}, open(out / f'{s}.json', 'w'), indent=1)
    np.savez_compressed(out / f'{s}_traj.npz', **arrays)
    good = [r for r in rows if 'error' not in r]
    print(f"[02] {s}: {len(good)}/{len(rows)} ep · A on {np.mean([r['A_on_frac'] for r in good]):.3f} · "
          f"B |Δgeo| med {np.median([r['B_geo_abs_err_m'] for r in good]):.4f} m · "
          f"C med {np.median([r['C_med_m'] for r in good]):.3f} m act= {np.mean([r['C_act_equal'] for r in good]):.2f} · "
          f"D same {np.mean([r.get('D_same_route', False) for r in good]):.3f} · {scene_row['sec']:.0f} s")


# ---------------------------------------------------------------------------
def aggregate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.out_dir)
    files = sorted((out / 'scenes').glob('*.json'))
    scenes, eps = [], []
    for f in files:
        j = json.load(open(f))
        scenes.append(j['scene'])
        eps += j['episodes']
    err = [r for r in eps if 'error' in r]
    E = [r for r in eps if 'error' not in r]
    groups = {'all': E, 'flat': [r for r in E if not r['multifloor']], 'multifloor': [r for r in E if r['multifloor']]}

    def q(v, p):
        v = [x for x in v if x is not None and np.isfinite(x)]
        return float(np.percentile(v, p)) if v else None

    summ = {'n_scenes': len(scenes), 'n_episodes': len(eps), 'n_error': len(err)}
    for gname, G in groups.items():
        if not G:
            continue
        n = len(G)
        summ[gname] = {
            'n': n,
            'A_on_frac_min': min(r['A_on_frac'] for r in G),
            'A_eps_fully_on': sum(r['A_on_frac'] == 1.0 for r in G),
            'A_eps_no_far_point': sum(r['A_far_frac'] == 0.0 for r in G),
            'A_eps_far_gt_10pct': sum(r['A_far_frac'] > 0.10 for r in G),
            'A_dz_med_m': q([r['A_dz_med_m'] for r in G], 50), 'A_dz_max_p99_m': q([r['A_dz_max_m'] for r in G], 99),
            'A2_start_err_max_m': max(r['A2_start_err_m'] for r in G),
            'B_geo_abs_err_med_m': q([r['B_geo_abs_err_m'] for r in G], 50),
            'B_geo_abs_err_p99_m': q([r['B_geo_abs_err_m'] for r in G], 99),
            'B_eps_err_lt_1cm': sum(r['B_geo_abs_err_m'] < 0.01 for r in G),
            'C_med_of_med_m': q([r['C_med_m'] for r in G], 50), 'C_p90_of_med_m': q([r['C_med_m'] for r in G], 90),
            'C_eps_med_lt_10cm': sum(r['C_med_m'] < 0.10 for r in G),
            'C_outlier_med_gt_1m': sum(r['C_med_m'] > 1.0 for r in G),
            'C_act_equal': sum(r['C_act_equal'] for r in G), 'C_act_ratio_med': q([r['C_act_ratio'] for r in G], 50),
            'C2_ok': sum(r.get('C2_ok', False) for r in G),
            'C2_med_of_med_m': q([r.get('C2_med_m') for r in G], 50), 'C2_p90_of_med_m': q([r.get('C2_med_m') for r in G], 90),
            'C2_eps_med_lt_10cm': sum((r.get('C2_med_m') is not None and r['C2_med_m'] < 0.10) for r in G),
            'C2_same_route': sum(r.get('C2_same_route', False) for r in G),
            'D_found': sum(r.get('D_found', False) for r in G),
            'D_same_route': sum(r.get('D_same_route', False) for r in G),
            'D_frechet_med_m': q([r.get('D_frechet_xy_m') for r in G], 50),
            'D_len_ratio_med': q([r.get('D_len_ratio') for r in G], 50),
            'D_z_range_ok': sum(r.get('D_z_range_err_m') is not None and r['D_z_range_err_m'] < 0.5 for r in G),
        }
        if 'E_geo_abs_err_m' in G[0]:
            summ[gname].update(E_geo_abs_err_med_m=q([r['E_geo_abs_err_m'] for r in G], 50),
                               E_same_route=sum(r.get('E_same_route', False) for r in G))
    if scenes and 'recomputed_area_m2' in scenes[0]:
        ar = [s['recomputed_area_m2'] / s['official_area_m2'] for s in scenes]
        summ['E_area_ratio'] = {'median': float(np.median(ar)), 'min': float(min(ar)), 'max': float(max(ar))}
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    figs = {}
    # (1) 분포
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for g, c in [('flat', 'tab:blue'), ('multifloor', 'tab:red')]:
        G = groups[g]
        if not G:
            continue
        axs[0][0].hist(np.clip([r['B_geo_abs_err_m'] for r in G], 1e-5, None), bins=np.logspace(-5, 1, 60), alpha=0.6, color=c, label=g)
        axs[0][1].hist([r['C_med_m'] for r in G], bins=np.linspace(0, 1.0, 60), alpha=0.6, color=c, label=g)
        axs[1][0].hist([r['C_act_ratio'] for r in G], bins=np.linspace(0, 1, 50), alpha=0.6, color=c, label=g)
        axs[1][1].hist(np.clip([r.get('D_frechet_xy_m', 20) for r in G], 0, 10), bins=np.linspace(0, 10, 60), alpha=0.6, color=c, label=g)
    axs[0][0].set_xscale('log'); axs[0][0].set_title('B |find_path geodesic - dataset info.geodesic| [m]')
    axs[0][1].set_title('C GT point -> follower path distance, per-episode median [m]')
    axs[1][0].set_title('C action sequence similarity (1 = identical)')
    axs[1][1].axvline(DOOR_WIDTH_M, color='k', ls=':'); axs[1][1].set_title('D Frechet(xy) start->goal shortest vs GT [m] (<0.8 = same route)')
    for a in axs.ravel():
        a.legend()
    figs['dist'] = 'dist.png'; fig.savefig(out / figs['dist'], dpi=90, bbox_inches='tight'); plt.close(fig)

    # (2) 예시: 층 이동 (D 같은 루트 · 다른 루트) + 평지 C 최악
    mf = groups['multifloor']
    pick = []
    pick += [('multifloor, D same route', r) for r in sorted([r for r in mf if r.get('D_same_route')], key=lambda r: -abs(r['gt_dz_m']))[:3]]
    pick += [('multifloor, D different route', r) for r in sorted([r for r in mf if r.get('D_found') and not r.get('D_same_route')],
                                                                   key=lambda r: -r['D_frechet_xy_m'])[:2]]
    pick += [('flat, worst C (follower vs GT)', r) for r in sorted(groups['flat'], key=lambda r: -r['C_med_m'])[:2]]
    if pick:
        fig, axs = plt.subplots(len(pick), 2, figsize=(15, 4 * len(pick)), squeeze=False, gridspec_kw={'width_ratios': [1, 1.6]})
        cache = {}
        for i, (title, r) in enumerate(pick):
            if r['scene'] not in cache:
                cache[r['scene']] = np.load(out / 'scenes' / f"{r['scene']}_traj.npz")
            A = cache[r['scene']]
            gt, fol = A[f"{r['ep']}_gt"], A[f"{r['ep']}_fol"]
            ours = A[f"{r['ep']}_ours"] if f"{r['ep']}_ours" in A else None
            pf = nu.load_pathfinder(f"{args.scene_root}/{r['scene']}/{r['scene']}.navmesh")
            tris, _ = nu.navmesh_triangles_islands(pf)
            zr = (min(gt[:, 2].min(), -1e9 if ours is None else ours[:, 2].min()) - 0.5, gt[:, 2].max() + 0.5)
            m = (tris.mean(1)[:, 2] > zr[0]) & (tris.mean(1)[:, 2] < max(zr[1], gt[:, 2].max() + 0.5))
            from matplotlib.collections import PolyCollection
            ax = axs[i][0]
            ax.add_collection(PolyCollection(tris[m][:, :, :2], facecolors='0.85', edgecolors='none'))
            ax.plot(gt[:, 0], gt[:, 1], color='orange', lw=3, label='GT')
            ax.plot(fol[:, 0], fol[:, 1], color='green', lw=1.2, label='C follower')
            if ours is not None:
                ax.plot(ours[:, 0], ours[:, 1], color='blue', lw=1.0, ls='--', label='D start->goal')
            ax.plot(*gt[0, :2], 'ko'); ax.plot(*gt[-1, :2], 'r*', ms=12)
            ax.autoscale(); ax.set_aspect('equal'); ax.legend(fontsize=7)
            ax.set_title(f"{title}\n{r['scene']} ep{r['ep']} dz {r['gt_dz_m']:+.2f} m", fontsize=9)
            ax = axs[i][1]
            for X, col, nm in [(gt, 'orange', 'GT'), (fol, 'green', 'follower'), (ours, 'blue', 'D')]:
                if X is None:
                    continue
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1))]
                ax.plot(s, X[:, 2], color=col, label=nm, lw=2 if nm == 'GT' else 1)
            ax.set_xlabel('xy path length [m]'); ax.set_ylabel('floor z [m]'); ax.legend(fontsize=7)
            ax.set_title(f"C med {r['C_med_m']:.3f} m, act sim {r['C_act_ratio']:.2f} | D Frechet {r.get('D_frechet_xy_m', float('nan')):.2f} m", fontsize=9)
        figs['examples'] = 'examples.png'; fig.savefig(out / figs['examples'], dpi=85, bbox_inches='tight'); plt.close(fig)

    def tbl(d):
        keys = sorted({k for g in ('all', 'flat', 'multifloor') if g in d for k in d[g]})
        h = '<tr><th>지표</th>' + ''.join(f'<th>{g}</th>' for g in ('all', 'flat', 'multifloor') if g in d) + '</tr>'
        for k in keys:
            h += f'<tr><td>{k}</td>' + ''.join(
                f"<td>{d[g].get(k) if not isinstance(d[g].get(k), float) else round(d[g][k], 4)}</td>" for g in ('all', 'flat', 'multifloor') if g in d) + '</tr>'
        return f'<table>{h}</table>'
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>nav3d 02 vln_ce GT</title>
<style>body{{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}
img{{max-width:100%}}pre{{background:#f6f6f6;padding:8px}}</style></head><body>
<h1>nav3d 02 — vln_ce GT 비교</h1><p>씬 {summ['n_scenes']} · 에피소드 {summ['n_episodes']} (매칭 실패 {summ['n_error']})</p>
{tbl(summ)}<pre>{json.dumps({k: v for k, v in summ.items() if k not in ('all', 'flat', 'multifloor')}, indent=1)}</pre>
<img src="{figs.get('dist', '')}"><img src="{figs.get('examples', '')}"></body></html>"""
    (out / 'report.html').write_text(html)
    print(f'  -> {out}/report.html')


def main():
    args = parse()
    if args.aggregate:
        aggregate(args)
    else:
        assert args.scene, '--scene 또는 --aggregate'
        run_scene(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
