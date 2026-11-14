"""noeun 74c — 74b 결과(경로 묶음)를 **지우지 않고 늘린다**: 평가를 두 종류로 나누고, 걷다 넘어질 몫만큼 여유 경로를 더한다.

나누기 (split 값)
    eval        = 평가 A · 안 본 랜드마크 — 학습에 없는 랜드마크. 목적지 자리도 학습 목적지와 `--share_m` 안에 겹치지 않는다.
    eval_seen   = 평가 B · 본 랜드마크 — 학습 랜드마크로 가지만 출발점이 그 목적지의 학습 출발점들과 `--min_start_sep` 이상 떨어진 새 경로.
    train       = 학습.
74b 에서 eval 이었는데 목적지 자리가 학습 랜드마크 목적지와 `--share_m` 안에 겹치는 것은 eval_seen 으로 옮긴다 (누수 정리).
출발·목적지가 똑같은 학습 경로 쌍은 뒤의 것을 뺀다.

더하는 순서 (모두 74b 의 make 와 같은 판정 U1~U5)
    1단계: eval 보충 `--add_eval` 개 (평가 랜드마크의, 아직 안 쓴 목적지부터) + eval_seen `--n_seen` 개 (학습 랜드마크 목적지 하나에 1 개, 랜드마크당 최대 `--seen_per_landmark`)
    2단계: train 보충 `--add_train` 개 (목적지당 `--paths_per_goal` 개까지, 출발점은 그 목적지의 기존·1단계 출발점과 떨어지게)
난수: 목적지마다 [seed, 74, 단계, 목적지 번호] → 74b 처럼 `--workers` 와 무관하게 같은 결과.
기존 episode_id 0..N-1 은 그대로, 새 경로는 N 부터. 걷기 뒤 split 마다 **episode_id 순서로 앞에서부터** 성공한 것을 쓰면 기존 경로가 먼저 쓰인다.

사용 (habitat 컨테이너, 저장소 루트):
    python3 scripts/dataset_converters/nav3d/noeun/74c_extend_paths.py --base logs/noeun/paths_landmark/noeun_station_paths.json \
        --goals logs/goal_candidates_spots6_buf50/goals.json --per_landmark 6 --stairs_buffer_m 0.5 --paths_per_goal 3 \
        --add_eval 25 --n_seen 100 --add_train 350 --workers 24 --out_dir logs/noeun/paths_landmark_ext
"""

import argparse
import collections
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('m74b', Path(__file__).resolve().parent / '74b_make_paths_parallel.py')
m = importlib.util.module_from_spec(_spec)
sys.modules['m74b'] = m
_spec.loader.exec_module(m)
npth, nu, scenes, gr = m.npth, m.nu, m.scenes, m.gr
W = m.W


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='74b noeun_station_paths.json')
    ap.add_argument('--add_eval', type=int, default=25, help='평가 A 보충 수')
    ap.add_argument('--n_seen', type=int, default=100, help='평가 B (본 랜드마크) 수 (옮겨온 것 포함)')
    ap.add_argument('--add_train', type=int, default=350, help='학습 보충 수')
    ap.add_argument('--seen_per_landmark', type=int, default=2, help='평가 B 에서 랜드마크 하나당 최대')
    ap.add_argument('--share_m', type=float, default=0.3, help='목적지 자리가 이 안이면 같은 자리로 본다')
    k, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    a = m.parse()                                # 나머지는 74b 인자 그대로 (seed · workers · paths_per_goal ...)
    for kk, v in vars(k).items():
        setattr(a, kk, v)
    return a


def solve(job):
    """job = (단계, 목적지 번호, 만들 수, 피할 출발점들) → 경로들 (74b solve 와 같은 방식, 실패하면 멈춤)."""
    phase, gi, k, avoid = job
    a = W['a']
    li, lm, sp = W['goals'][gi]
    rng = np.random.default_rng([a.seed, 74, phase, gi])
    g = np.asarray(sp['xyz'], float)
    why, starts, paths = collections.Counter(), [np.asarray(s, float) for s in avoid], []
    for _ in range(k):
        want_mf = bool(rng.random() < a.multifloor_frac)
        got = m.make(rng, g, want_mf, lm, sp, starts, why) or m.make(rng, g, not want_mf, lm, sp, starts, why)
        if got is None:
            break
        starts.append(got['X'][0]); paths.append(got)
    return gi, paths, dict(why)


def run(pool, jobs, need, a, tag, t0):
    """jobs 순서대로 풀어 need 개를 채운다 (묶음 단위, 순서대로 모음 → workers 와 무관)."""
    out, total = [], collections.Counter()
    for b0 in range(0, len(jobs), a.batch):
        if len(out) >= need:
            break
        chunk = jobs[b0:b0 + a.batch]
        res = pool.map(solve, chunk, chunksize=1) if pool else [solve(j) for j in chunk]
        for gi, paths, why in res:
            total.update(why)
            for p in paths:
                if len(out) < need:
                    out.append((gi, p))
        print(f'  [{tag}] {len(out)}/{need}  {time.time() - t0:.0f}s', flush=True)
    return out, total


def main():
    a = parse()
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    B = json.load(open(a.base)); E = B['episodes']

    spec = scenes.SCENES['noeun_station']
    pf = nu.load_pathfinder(a.navmesh)
    sc = npth.NavScene(pf)
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    cfg = npth.RefineCfg()
    holes = m.load_holes(a.holes_csv, tuple(k.strip() for k in a.hole_kinds.split(',') if k.strip()))
    G = json.load(open(a.goals)); lms = G['landmarks']
    objs = [{**o, 'c': np.asarray(o['c'], float), 'dir': np.asarray(o['dir'], float)} for o in G['objects']]
    goals = [(li, lm, sp) for li, lm in enumerate(lms) for sp in lm['spots'][:a.per_landmark]]
    buf = cfg.min_run_m if a.stairs_buffer_m is None else a.stairs_buffer_m
    goals = [g for g in goals if not m.near_stairs(sc, g[2]['xyz'], r=buf)]       # 74b 와 같은 목록
    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, per_bucket=400, seed=a.seed + 1)
    by_island = {}
    for (st, isl), v in pts.items():
        by_island.setdefault(isl, []).extend(v)
    a.flat_min_m = a.min_m
    W.update(a=a, sc=sc, pf=pf, md=nu.MeshDistance(mesh), cfg=cfg, r_b=sc.agent_radius, holes=holes, objs=objs,
             goals=goals, by_island=by_island)

    # ── 기존 경로를 목적지 자리(key)에 붙이기 ─────────────────────────
    XY = np.array([g[2]['xyz'] for g in goals], float)
    key_of = {}                                   # 목적지 번호 -> 자리 key (같은 랜드마크 안 같은 자리는 한 key)
    for gi, (li, lm, sp) in enumerate(goals):
        key_of[gi] = (li, tuple(np.round(sp['xyz'], 2)))
    eval_lm = {e['stop_cue']['landmark_id'] for e in E if e['split'] == 'eval'}
    train_lm = {e['stop_cue']['landmark_id'] for e in E if e['split'] == 'train'}
    tr_spots = XY[[gi for gi, g in enumerate(goals) if g[0] not in eval_lm]]

    def near_train(p):
        return bool(len(tr_spots)) and float(np.linalg.norm(tr_spots - np.asarray(p, float), axis=1).min()) < a.share_m

    starts_at = collections.defaultdict(list)     # key -> 기존 출발점들
    n_at = collections.Counter()
    for e in E:
        li = e['stop_cue']['landmark_id']
        cand = [gi for gi, g in enumerate(goals) if g[0] == li]
        gi = min(cand, key=lambda j: np.linalg.norm(XY[j] - np.asarray(e['goal'])))
        assert np.linalg.norm(XY[gi] - np.asarray(e['goal'])) < 0.05, ('목적지 매칭 실패', e['episode_id'])
        e['_key'] = key_of[gi]
        starts_at[e['_key']].append(e['start']); n_at[e['_key']] += 1

    # ── 정리: 누수 eval → eval_seen, 똑같은 학습 쌍 빼기 ─────────────────
    moved, dropped, seen_pair = [], [], set()
    for e in E:
        if e['split'] == 'eval' and near_train(e['goal']):
            e['split'] = 'eval_seen'; moved.append(e['episode_id'])
        pk = (e['_key'], tuple(np.round(e['start'], 1)))
        if pk in seen_pair:
            dropped.append(e['episode_id']); e['_drop'] = True
        seen_pair.add(pk)
    E = [e for e in E if not e.get('_drop')]
    print(f'[74c] 기존 {len(B["episodes"])} · eval→eval_seen {moved} · 똑같은 쌍 뺌 {dropped}', flush=True)

    rng = np.random.default_rng([a.seed, 74, 0])
    order = [int(x) for x in rng.permutation(len(goals))]
    pool = mp.get_context('fork').Pool(a.workers) if a.workers > 1 else None

    # ── 1단계: eval 보충 + eval_seen ──────────────────────────────────
    used_keys = set(n_at)
    ev_jobs, ks = [], set()
    for rnd in (0, 1):                            # 안 쓴 자리 먼저, 그다음 쓴 자리
        for gi in order:
            li, k = goals[gi][0], key_of[gi]
            if li in eval_lm and k not in ks and (k in used_keys) == bool(rnd) and not near_train(XY[gi]):
                ks.add(k); ev_jobs.append((1, gi, 1, starts_at[k]))
    add_ev, tot1 = run(pool, ev_jobs, a.add_eval, a, 'eval', t0)
    n_seen_need = a.n_seen - len(moved)
    per_lm, se_jobs, ks = collections.Counter(), [], set()
    for gi in order:
        li, k = goals[gi][0], key_of[gi]
        if li in train_lm and k not in ks and per_lm[li] < a.seen_per_landmark:
            ks.add(k); per_lm[li] += 1; se_jobs.append((1, gi, 1, starts_at[k]))
    add_seen, tot2 = run(pool, se_jobs, n_seen_need, a, 'eval_seen', t0)
    for gi, p in add_ev + add_seen:
        starts_at[key_of[gi]].append(p['X'][0].tolist())

    # ── 2단계: train 보충 ─────────────────────────────────────────────
    tr_jobs, ks = [], set()
    for gi in order:
        li, k = goals[gi][0], key_of[gi]
        room = a.paths_per_goal - n_at[k]
        if li in train_lm and room > 0 and k not in ks:
            ks.add(k); tr_jobs.append((2, gi, room, list(starts_at[k])))
    add_tr, tot3 = run(pool, tr_jobs, a.add_train, a, 'train', t0)
    if pool:
        pool.close(); pool.join()

    # ── 내보내기: 기존(그대로) + 새 경로 (74b 와 같은 모양) ────────────────
    nid = max(e['episode_id'] for e in B['episodes']) + 1
    new = []
    for split, items in (('eval', add_ev), ('eval_seen', add_seen), ('train', add_tr)):
        for gi, p in items:
            li, lm, sp = goals[gi]
            X = p['X']
            new.append({
                'episode_id': nid, 'split': split, 'kind': p['kind'],
                'from_storey': p['from_storey'], 'to_storey': p['to_storey'], 'source': '74c',
                'start': X[0].tolist(), 'goal': X[-1].tolist(),
                'trajectory': X.tolist(), 'trajectory_z': X[:, 2].tolist(), 'floor_z': float(X[0, 2]),
                'segments': m.segments(p['st'], X[:, 2]),
                'path_length_m': float(p['row']['ref_length_m']),
                'mesh_clear_min_m': float(p['row']['ref_mesh_clear_min_m']),
                'margin_min_m': float(p['row']['ref_margin_min_m']),
                'stairs_frac': float(np.mean(p['st'])),
                'goal_yaw': float(sp['yaw']),
                'stop_cue': {'landmark_id': int(li), 'landmark_ids': lm.get('ids', [lm['id']]),
                             'part': sp.get('part', 'whole'), 'target_xyz': sp.get('target_xyz', lm['xyz']),
                             'anchor': sp.get('anchor'), 'passed_same': p['passed'], 'ordinal': len(p['passed']) + 1,
                             'name': lm['name'], 'names': lm.get('names', [lm['name']]), 'common': lm.get('common'),
                             'xyz': lm['xyz'], 'dist_m': float(sp['dist_m']),
                             'yaw_off_deg': float(sp.get('yaw_off_deg', 0.0))},
            })
            nid += 1
    for e in E:
        e.pop('_key', None)
    episodes = E + new
    doc = dict(B, episodes=episodes, extended_by='74c', base=a.base)
    (out / 'noeun_station_paths.json').write_text(json.dumps(doc))
    cnt = collections.Counter(e['split'] for e in episodes)
    (out / 'split.json').write_text(json.dumps({
        **{f'{s}_episode_ids': [e['episode_id'] for e in episodes if e['split'] == s] for s in ('eval', 'eval_seen', 'train')},
        'eval_landmark_ids': sorted(eval_lm), 'moved_to_seen': moved, 'dropped_same_pair': dropped,
        'note': 'eval = 안 본 랜드마크 (학습 목적지와 share_m 안 겹침 없음) · eval_seen = 학습 랜드마크, 새 출발점. '
                '걷기 뒤 split 마다 episode_id 순서로 앞에서부터 성공한 것을 쓴다.'}, indent=1))
    total = tot1 + tot2 + tot3
    (out / 'fail_reasons.json').write_text(json.dumps({'reasons': gr.REASONS, 'total': dict(total)}, ensure_ascii=False, indent=1))

    # 자체 검사
    ev_lm2 = {e['stop_cue']['landmark_id'] for e in episodes if e['split'] == 'eval'}
    assert not (ev_lm2 & train_lm), '평가 A 랜드마크가 학습에 있음'
    assert not any(near_train(e['goal']) for e in episodes if e['split'] == 'eval'), '평가 A 목적지가 학습 자리와 겹침'
    for e in new:
        assert not (e['segments'][0]['kind'] == 'stairs' or e['segments'][-1]['kind'] == 'stairs')
    print(f'\n[74c] 새 경로 eval {len(add_ev)} · eval_seen {len(add_seen)} · train {len(add_tr)} · {time.time() - t0:.0f}s')
    print(f'[74c] 합계 {dict(cnt)}')
    for k, v in total.most_common():
        print(f'       {k:<16} {v:>6}  {gr.REASONS[k]}')
    print(f'[74c] 저장 {out}/noeun_station_paths.json · split.json · fail_reasons.json')


if __name__ == '__main__':
    main()
