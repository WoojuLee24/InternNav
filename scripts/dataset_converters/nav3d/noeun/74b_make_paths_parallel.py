"""noeun 74b — 74_make_paths 를 **여러 CPU 코어로** 나눠 돌린다. 판정·출력 모양은 74 와 같다.

74 와 무엇이 다른가 (판정 규칙은 그대로, **난수 쓰는 법만** 다르다):
    74 는 난수 하나를 처음부터 끝까지 이어 쓴다 → 앞 목적지의 시도 횟수가 뒤 목적지의 출발점을 바꾼다.
    그대로 쪼개면 코어 수에 따라 결과가 달라진다.
    74b 는 **목적지마다 자기 난수**(seed, 목적지 번호)를 쓴다. 목적지 하나의 경로들은 그 목적지 안에서만 정해지므로
    어느 코어가 언제 풀든 같다. 결과는 74 와 같은 순서(pick 순서)로 모으고, 개수 채우기 규칙도 74 와 같다.
    → `--workers 1` 과 `--workers N` 의 결과가 **바이트까지 같다** (검증: 두 번 돌려 noeun_station_paths.json 비교).
    74 의 결과와는 난수가 달라 같은 경로가 아니다 (분포만 같다).

나누기: 목적지를 pick 순서로 `--batch` 개씩 잘라 풀(pool)에 넣는다. 필요한 수를 채우면 멈춘다.
     마지막 묶음에서 남는 계산은 버린다 (결과에는 영향 없음).

주의: CPU 코어를 너무 많이 쓰면 서버가 느려진다. `--workers` 기본 8.

사용 (habitat 컨테이너, 저장소 루트):
    python3 scripts/dataset_converters/nav3d/noeun/74b_make_paths_parallel.py --goals <goals.json> --out_dir <폴더> --workers 8
"""

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
import goal_rules as gr  # noqa: E402
from path_quality import load_holes, segments, judge, one, stairs_points, near_stairs  # noqa: E402

W = {}   # 워커가 쓰는 공유 상태 (fork 로 물려받는다)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- 74 와 같은 인자 (뜻·기본값 같음) ----
    ap.add_argument('--goals', default='logs/goal_candidates/goals.json', help='73 결과')
    ap.add_argument('--out_dir', default='logs/noeun/paths_landmark', help='22 와 **다른** 폴더여야 한다')
    ap.add_argument('--n_eval', type=int, default=80, help='평가 에피소드')
    ap.add_argument('--n_train', type=int, default=1200, help='학습 에피소드')
    ap.add_argument('--per_landmark', type=int, default=3, help='랜드마크당 쓸 목적지 수')
    ap.add_argument('--min_m', type=float, default=3.0, help='출발점~목적지 직선 거리 하한')
    ap.add_argument('--max_m', type=float, default=40.0, help='상한')
    ap.add_argument('--max_path_m', type=float, default=60.0, help='경로 길이(걸어가는 길) 상한')
    ap.add_argument('--multifloor_frac', type=float, default=0.25, help='층을 넘는 경로의 비율')
    ap.add_argument('--tries', type=int, default=40, help='경로 하나당 출발점 시도 횟수')
    ap.add_argument('--paths_per_goal', type=int, default=2, help='학습: 목적지 하나에 만들 경로 수 (평가는 1)')
    ap.add_argument('--min_start_sep', type=float, default=5.0, help='같은 목적지 경로들의 출발점 사이 최소 거리 [m]')
    ap.add_argument('--pass_r', type=float, default=3.0)
    ap.add_argument('--max_pass', type=int, default=2)
    ap.add_argument('--succ_r', type=float, default=3.0)
    ap.add_argument('--holes_csv', default='scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv')
    ap.add_argument('--hole_kinds', default='mesh,gs')
    ap.add_argument('--navmesh', default='scripts/dataset_converters/nav3d/logs/noeun/12_build_navmesh/noeun_station/noeun_station.navmesh')
    ap.add_argument('--stairs_buffer_m', type=float, default=None, help='74 와 같음 (None = RefineCfg.min_run_m)')
    ap.add_argument('--eval_over', type=float, default=1.5, help='74 와 같음')
    ap.add_argument('--seed', type=int, default=0)
    # ---- 74b 만의 인자 ----
    ap.add_argument('--workers', type=int, default=8, help='동시에 쓰는 CPU 프로세스 수. 1 = 한 코어')
    ap.add_argument('--batch', type=int, default=64, help='한 번에 풀에 넣는 목적지 수 (결과에는 영향 없음)')
    return ap.parse_args()


def storey_of(z):
    sc = W['sc']
    k = int(np.argmin([abs(L['z'] - z) for L in sc.levels])) if sc.levels else 0
    return int(sc.lv2st[k]) if len(sc.lv2st) else 0


def make(rng, goal_xyz, want_multifloor, lm, sp, starts, why):
    """74 의 make 와 같다 (rng 만 인자로 받는다)."""
    a, sc, pf = W['a'], W['sc'], W['pf']
    q_goal = nu.ours_to_hab(goal_xyz)
    isl = int(pf.get_island(pf.snap_point(q_goal)))
    cand = W['by_island'].get(isl)
    if not cand:
        why['NO_PATH'] += 1
        return None
    g_st = storey_of(goal_xyz[2])
    for _ in range(a.tries):
        q = cand[rng.integers(len(cand))]
        s = nu.hab_to_ours(q)
        if want_multifloor and storey_of(s[2]) == g_st:
            why['STOREY_MISMATCH'] += 1
            continue
        if not want_multifloor and storey_of(s[2]) != g_st:
            why['STOREY_MISMATCH'] += 1
            continue
        if not (a.min_m <= float(np.linalg.norm(s[:2] - goal_xyz[:2])) <= a.max_m):
            why['DIST_RANGE'] += 1
            continue
        if any(float(np.linalg.norm(s[:2] - t[:2])) < a.min_start_sep for t in starts):
            why['START_TOO_CLOSE'] += 1
            continue
        P = sc.find_path(s, goal_xyz)
        if P is None:
            why['NO_PATH'] += 1
            continue
        row, r = one(sc, P, W['md'], W['r_b'], W['cfg'])
        X = r['traj'].astype(np.float64)
        st_mask = np.asarray(r['stairs'], bool)
        if st_mask[0] or st_mask[-1]:
            why['START_ON_STAIRS' if st_mask[0] else 'GOAL_ON_STAIRS'] += 1
            continue
        if float(row['ref_length_m']) > a.max_path_m:
            why['TOO_LONG'] += 1
            continue
        passed = [] if sp.get('anchor') else gr.passed_same(W['objs'], lm['oid'], X, goal_xyz, a.pass_r, a.succ_r)
        if len(passed) > a.max_pass:
            why['DISTRACTOR_MANY'] += 1
            continue
        kind = 'multifloor' if len({storey_of(z) for z in X[:, 2]}) > 1 else 'flat'
        bad = judge(row, X, kind, W['holes'], W['r_b'], a)
        if bad:
            for b in bad:
                why[gr.JUDGE_CODE[b.split()[0]]] += 1
            continue
        return dict(kind=kind, X=X, st=st_mask, row=row, passed=passed,
                    from_storey=storey_of(X[0, 2]), to_storey=g_st)
    return None


def solve(gi):
    """목적지 gi 하나: 자기 난수로 경로 k 개 (평가 1 · 학습 paths_per_goal). 실패하면 거기서 멈춘다 (74 와 같음)."""
    a = W['a']
    li, lm, sp = W['goals'][gi]
    is_eval = int(li) in W['eval_lm']
    k_goal = 1 if is_eval else a.paths_per_goal
    rng = np.random.default_rng([a.seed, 74, gi])
    g = np.asarray(sp['xyz'], float)
    why, starts, paths, failed = collections.Counter(), [], [], False
    for _ in range(k_goal):
        want_mf = bool(rng.random() < a.multifloor_frac)
        got1 = make(rng, g, want_mf, lm, sp, starts, why) or make(rng, g, not want_mf, lm, sp, starts, why)
        if got1 is None:
            failed = True
            break
        starts.append(got1['X'][0])
        paths.append(got1)
    return gi, paths, failed, dict(why)


def main():
    a = parse()
    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)        # 나누기·pick 순서는 74 와 같은 방식

    spec = scenes.SCENES['noeun_station']
    pf = nu.load_pathfinder(a.navmesh)
    sc = npth.NavScene(pf)
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    cfg = npth.RefineCfg()
    r_b = sc.agent_radius
    hole_kinds = tuple(k.strip() for k in a.hole_kinds.split(',') if k.strip())
    holes = load_holes(a.holes_csv, hole_kinds)
    print(f'[74b] U4 구멍 {len(holes)} 개 (종류 {",".join(hole_kinds)}) · workers {a.workers}', flush=True)

    G = json.load(open(a.goals))
    lms = G['landmarks']
    objs = [{**o, 'c': np.asarray(o['c'], float), 'dir': np.asarray(o['dir'], float)} for o in G['objects']]
    goals = [(li, lm, sp) for li, lm in enumerate(lms) for sp in lm['spots'][:a.per_landmark]]
    print(f'[74b] 랜드마크 {len(lms)} · 목적지 {len(goals)} (랜드마크당 최대 {a.per_landmark})', flush=True)

    def on_stairs(p):
        return bool(stairs_points(sc, np.asarray(p, float).reshape(1, 3))[0])
    buf = cfg.min_run_m if a.stairs_buffer_m is None else a.stairs_buffer_m
    n0 = len(goals)
    goals = [g for g in goals if not near_stairs(sc, g[2]['xyz'], r=buf)]
    print(f'[74b] 계단 위·계단 {buf:g} m 안 목적지 제외 {n0 - len(goals)} -> 목적지 {len(goals)}', flush=True)

    need = a.n_eval + a.n_train
    order = rng.permutation(len(lms))
    per = {li: sum(1 for g in goals if g[0] == li) for li in range(len(lms))}
    eval_lm, got = set(), 0
    for li in order:
        if got >= a.n_eval * a.eval_over:
            break
        eval_lm.add(int(li)); got += per.get(int(li), 0)
    n_tr_goal = sum(1 for g in goals if g[0] not in eval_lm)
    print(f'[74b] 나누기 — 평가 랜드마크 {len(eval_lm)} (목적지 {got}) · 학습 랜드마크 {len(lms) - len(eval_lm)} '
          f'(목적지 {n_tr_goal} -> 최대 경로 {n_tr_goal * a.paths_per_goal})', flush=True)

    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, per_bucket=400, seed=a.seed + 1)
    by_island = {}
    for (st, isl), v in pts.items():
        by_island.setdefault(isl, []).extend(v)
    a.flat_min_m = a.min_m                     # judge 가 보는 U5 인자 (74 와 같음)

    W.update(a=a, sc=sc, pf=pf, md=md, cfg=cfg, r_b=r_b, holes=holes, objs=objs, goals=goals,
             eval_lm=eval_lm, by_island=by_island)

    # ── 만들기: pick 순서로 묶음을 풀고, 74 와 같은 규칙으로 모은다 ──────────────
    pick = [int(x) for x in rng.permutation(len(goals))]
    eps, fail, fail_log, total = [], 0, {}, collections.Counter()
    n_have = {'eval': 0, 'train': 0}
    pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None
    done = False
    for b0 in range(0, len(pick), a.batch):
        # 이미 다 찬 쪽(평가/학습) 목적지는 풀지 않는다 — 어차피 아래에서 건너뛰므로 결과는 같다 (헛계산만 없앰)
        chunk = [gi for gi in pick[b0:b0 + a.batch]
                 if n_have['eval' if int(goals[gi][0]) in eval_lm else 'train']
                 < (a.n_eval if int(goals[gi][0]) in eval_lm else a.n_train)]
        res = pool.map(solve, chunk, chunksize=1) if pool else [solve(gi) for gi in chunk]
        for n, (gi, paths, failed, why) in zip(range(b0, b0 + len(chunk)), res):
            li, lm, sp = goals[gi]
            is_eval = int(li) in eval_lm
            sp_name = 'eval' if is_eval else 'train'
            tgt = a.n_eval if is_eval else a.n_train
            if n_have[sp_name] >= tgt:
                continue
            if len(eps) >= need:
                done = True
                break
            m = min(1 if is_eval else a.paths_per_goal, tgt - n_have[sp_name])
            for p in paths[:m]:
                p.update(split=sp_name, landmark=li, lm=lm, spot=sp)
                eps.append(p)
                n_have[sp_name] += 1
            if len(paths) < m:
                fail += 1
            if why:
                fail_log[f"{lm['id']}:{lm['spots'].index(sp)}"] = why
                total.update(why)
            if (n + 1) % 100 == 0 or len(eps) == need:
                print(f'  {len(eps)}/{need}  (평가 {n_have["eval"]} · 학습 {n_have["train"]} · 실패 {fail}) '
                      f'{time.time() - t0:.0f}s', flush=True)
        if done or len(eps) >= need:
            break
    if pool:
        pool.close(); pool.join()

    # ── 내보내기 (74 와 같은 모양) ──────────────────────────────
    episodes = []
    for k, e in enumerate(eps):
        X, lm, sp = e['X'], e['lm'], e['spot']
        episodes.append({
            'episode_id': k, 'split': e['split'], 'kind': e['kind'],
            'from_storey': e['from_storey'], 'to_storey': e['to_storey'], 'source': '74b',
            'start': X[0].tolist(), 'goal': X[-1].tolist(),
            'trajectory': X.tolist(), 'trajectory_z': X[:, 2].tolist(), 'floor_z': float(X[0, 2]),
            'segments': segments(e['st'], X[:, 2]),
            'path_length_m': float(e['row']['ref_length_m']),
            'mesh_clear_min_m': float(e['row']['ref_mesh_clear_min_m']),
            'margin_min_m': float(e['row']['ref_margin_min_m']),
            'stairs_frac': float(np.mean(e['st'])),
            'goal_yaw': float(sp['yaw']),
            'stop_cue': {'landmark_id': int(e['landmark']), 'landmark_ids': lm.get('ids', [lm['id']]),
                         'part': sp.get('part', 'whole'), 'target_xyz': sp.get('target_xyz', lm['xyz']),
                         'anchor': sp.get('anchor'),
                         'passed_same': e['passed'], 'ordinal': len(e['passed']) + 1,
                         'name': lm['name'], 'names': lm.get('names', [lm['name']]),
                         'common': lm.get('common'),
                         'xyz': lm['xyz'], 'dist_m': float(sp['dist_m']),
                         'yaw_off_deg': float(sp.get('yaw_off_deg', 0.0))},
        })
    doc = {'scene_id': 'noeun_station', 'coords': 'nav3d (Z-up) = usdz world', 'r_b': r_b,
           'agent_height': sc.agent_height, 'navmesh': a.navmesh,
           'criteria': 'U1~U5 (22_select_paths.py docstring) · hole_kinds=' + a.hole_kinds,
           'goals_from': a.goals, 'episodes': episodes}
    (out / 'noeun_station_paths.json').write_text(json.dumps(doc))
    ev = [e for e in episodes if e['split'] == 'eval']
    tr = [e for e in episodes if e['split'] == 'train']
    (out / 'split.json').write_text(json.dumps({
        'eval_episode_ids': [e['episode_id'] for e in ev],
        'train_episode_ids': [e['episode_id'] for e in tr],
        'eval_landmark_ids': sorted(eval_lm),
        'note': '같은 랜드마크가 양쪽에 들어가지 않는다 (평가 누수 방지)'}, indent=1))

    for e in eps:                                # 자체 검사 (74 와 같음)
        assert not (e['st'][0] or e['st'][-1]), '출발/도착이 계단 위'
        assert not on_stairs(e['X'][-1]), '도착이 계단 위'
        assert float(e['row']['ref_length_m']) <= a.max_path_m, '경로가 너무 김'
    assert not ({e['landmark'] for e in eps if e['split'] == 'eval'} & {e['landmark'] for e in eps if e['split'] == 'train'}), \
        '평가·학습 랜드마크 겹침'

    (out / 'fail_reasons.json').write_text(json.dumps(
        {'reasons': gr.REASONS, 'total': dict(total), 'per_goal': fail_log}, ensure_ascii=False, indent=1))
    d = [e['stop_cue']['dist_m'] for e in episodes]
    L = [e['path_length_m'] for e in episodes]
    print(f'\n[74b] 경로 {len(episodes)} (평가 {len(ev)} · 학습 {len(tr)}) · 출발점 못 찾음 {fail} · {time.time() - t0:.0f}s')
    if d:
        print(f'[74b] 목적지~멈출 대상 거리  중앙 {np.median(d):.2f} m · 최대 {max(d):.2f} m')
        print(f'[74b] 경로 길이              중앙 {np.median(L):.1f} m · 최대 {max(L):.1f} m')
        print(f'[74b] 층 이동 {sum(1 for e in episodes if e["kind"] == "multifloor")} · '
              f'평지 {sum(1 for e in episodes if e["kind"] == "flat")}')
        print(f'[74b] 쓰인 랜드마크 {len({e["stop_cue"]["landmark_id"] for e in episodes})} 개')
    print('[74b] 실패 이유 (시도 횟수):')
    for k, v in total.most_common():
        print(f'       {k:<16} {v:>6}  {gr.REASONS[k]}')
    print(f'[74b] 저장 {out}/noeun_station_paths.json · split.json · fail_reasons.json')


if __name__ == '__main__':
    main()
