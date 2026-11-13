"""noeun 74 — **73 이 고른 목적지까지** 가는 경로를 만든다.

22_select_paths.py 와 무엇이 다른가:
    22 는 출발점·도착점을 **둘 다 무작위**로 찍는다. 그래서 도착점 옆에 멈출 물건이 있는지는
    나중에 찾아야 했고, 평가셋 40개 중 21개가 "대상이 2 m 밖" 또는 "범위 3 m 초과"로 못 쓰게 됐다.
    74 는 **도착점이 이미 정해져 있다** — 73 이 랜드마크 옆 0.6~2 m 에서 고른 자리다.
    출발점만 찾으면 된다. 설계상 "멈출 물건이 없는 목적지"가 생기지 않는다.

    22 는 그대로 둔다 (기존 산출물·재현성 때문). 74 는 새 파일이고 출력 폴더도 다르다.

입력: 73_goal_candidates.py 의 goals.json  (랜드마크마다 spots = [{xyz, yaw, dist_m, ...}])
출력: <out_dir>/noeun_station_paths.json   (22 와 같은 모양 + goal 정보)
      <out_dir>/split.json                  (eval / train 나눈 결과)

## 용어
- **목적지(goal)**: 로봇이 멈춰 서는 자리. 73 이 고른 spot.
- **멈출 대상(landmark)**: "저기서 멈춰라"가 가리키는 물건. 목적지에서 2 m 안에 있다.
- **goal yaw**: 그 자리에서 랜드마크를 바라보는 방향. 에피소드의 마지막 방향이 된다.
- **r_b**: 로봇 몸 반지름 0.25 m. **mesh 여유**: 경로 점에서 mesh 까지 3D 거리.

## 사용 가능 기준 — 22 와 **같은** U1~U5 (path_quality.judge)
U1 경로 있음 · U2 navmesh 위 100 % · U3 mesh 여유 ≥ r_b · U4 큰 구멍 옆 아님 · U5 평지면 한 층·3 m 이상

구멍 종류는 `--hole_kinds mesh,gs` 가 기본이다 (22 의 기본값 mesh 와 다르다).
`gs` 는 렌더 이미지에 바닥이 비치는 곳이라, 학습 사진이 깨진다.

## eval / train 나누기 — **랜드마크 단위**로 자른다
같은 랜드마크가 양쪽에 들어가면 평가가 새어 나간다("이 자판기 옆"을 학습에서 이미 봤다).
그래서 목적지가 아니라 **랜드마크를 먼저 나누고** 그 랜드마크의 목적지를 각 쪽에 준다.

사용 (habitat 컨테이너, 저장소 루트):
    python3 scripts/dataset_converters/nav3d/noeun/74_make_paths.py \
      --goals logs/goal_candidates/goals.json --out_dir logs/noeun/paths_landmark
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
sys.path.insert(0, str(HERE))
import nav3d_path as npth  # noqa: E402
import nav3d_utils as nu  # noqa: E402
import nav3d_scenes as scenes  # noqa: E402
from path_quality import load_holes, segments, judge, one, stairs_points  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collections  # noqa: E402
import goal_rules as gr  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--goals', default='logs/goal_candidates/goals.json', help='73 결과')
    ap.add_argument('--out_dir', default='logs/noeun/paths_landmark',
                    help='22 와 **다른** 폴더여야 한다. 기존 경로 파일을 덮어쓰지 않는다')
    ap.add_argument('--n_eval', type=int, default=80, help='평가 에피소드')
    ap.add_argument('--n_train', type=int, default=1200, help='학습 에피소드')
    ap.add_argument('--per_landmark', type=int, default=3, help='랜드마크당 쓸 목적지 수 (73 이 최대 3개 뽑는다)')
    ap.add_argument('--min_m', type=float, default=3.0, help='출발점~목적지 직선 거리 하한')
    ap.add_argument('--max_m', type=float, default=40.0, help='상한 (R2R 수준 길이)')
    ap.add_argument('--max_path_m', type=float, default=60.0,
                    help='경로 길이(걸어가는 길) 상한. VLN-PE 600 걸음 ≈ 88 m (실주행 6.6~6.9 걸음/m, '
                         '261112_noeun_train_deploy_gap_result.md) → 약 2/3 로 여유를 둔다')
    ap.add_argument('--multifloor_frac', type=float, default=0.25,
                    help='층을 넘는 경로의 비율. 0 이면 전부 같은 층')
    ap.add_argument('--tries', type=int, default=40, help='경로 하나당 출발점 시도 횟수')
    ap.add_argument('--paths_per_goal', type=int, default=2, help='학습: 목적지 하나에 출발점을 달리해 만들 경로 수 (평가는 1)')
    ap.add_argument('--min_start_sep', type=float, default=5.0, help='같은 목적지 경로들의 출발점 사이 최소 거리 [m]')
    ap.add_argument('--pass_r', type=float, default=3.0, help='경로에서 이 거리 안의 같은 종류 물체 = 지나간 것')
    ap.add_argument('--max_pass', type=int, default=2, help='지나가도 되는 같은 종류 수 (기준점이 있는 목적지는 안 셈)')
    ap.add_argument('--succ_r', type=float, default=3.0, help='성공 반경. 목적지에서 이 안의 같은 종류는 지나간 것으로 안 셈')
    ap.add_argument('--holes_csv', default='scripts/dataset_converters/nav3d/logs/noeun/11_check_assets/noeun_station/holes.csv')
    ap.add_argument('--hole_kinds', default='mesh,gs',
                    help="U4 가 피할 구멍 종류. 'mesh'=로봇이 빠지는 곳 · 'gs'=렌더에 바닥이 비치는 곳")
    ap.add_argument('--navmesh', default='scripts/dataset_converters/nav3d/logs/noeun/12_build_navmesh/noeun_station/noeun_station.navmesh')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def main():
    a = parse()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    spec = scenes.SCENES['noeun_station']
    pf = nu.load_pathfinder(a.navmesh)
    sc = npth.NavScene(pf)
    mesh = nu.add_box_blockers(nu.load_mesh_zup(spec['mesh']), spec.get('blockers', []))
    md = nu.MeshDistance(mesh)
    cfg = npth.RefineCfg()
    r_b = sc.agent_radius

    hole_kinds = tuple(k.strip() for k in a.hole_kinds.split(',') if k.strip())
    holes = load_holes(a.holes_csv, hole_kinds)
    print(f'[74] U4 구멍 {len(holes)} 개 (종류 {",".join(hole_kinds)})', flush=True)

    # ── 목적지 읽기 ─────────────────────────────────────────────
    G = json.load(open(a.goals))
    lms = G['landmarks']
    objs = [{**o, 'c': np.asarray(o['c'], float), 'dir': np.asarray(o['dir'], float)} for o in G['objects']]
    goals = []                       # (랜드마크 번호, 랜드마크, spot)
    for li, lm in enumerate(lms):
        for sp in lm['spots'][:a.per_landmark]:
            goals.append((li, lm, sp))
    print(f'[74] 랜드마크 {len(lms)} · 목적지 {len(goals)} (랜드마크당 최대 {a.per_landmark})', flush=True)

    # 계단 한가운데는 목적지가 될 수 없다 (계단 바로 앞·뒤 바닥은 된다). 계단 = NavScene 의 계단 삼각형 위.
    def on_stairs(p):
        return bool(stairs_points(sc, np.asarray(p, float).reshape(1, 3))[0])
    n0 = len(goals)
    goals = [g for g in goals if not on_stairs(g[2]['xyz'])]
    print(f'[74] 계단 위 목적지 제외 {n0 - len(goals)} -> 목적지 {len(goals)}', flush=True)

    need = a.n_eval + a.n_train
    if len(goals) < need:
        print(f'[74] 경고 — 목적지 {len(goals)} 개가 필요한 {need} 개보다 적다. 만들 수 있는 만큼만 만든다.')

    # ── eval / train 을 **랜드마크 단위**로 나눈다 (같은 랜드마크가 양쪽에 들어가면 평가가 샌다) ──
    order = rng.permutation(len(lms))
    per = {li: sum(1 for g in goals if g[0] == li) for li in range(len(lms))}
    eval_lm, got = set(), 0
    for li in order:
        if got >= a.n_eval:
            break
        eval_lm.add(int(li)); got += per.get(int(li), 0)
    print(f'[74] 나누기 — 평가 랜드마크 {len(eval_lm)} (목적지 {got}) · 학습 랜드마크 {len(lms) - len(eval_lm)}', flush=True)

    # ── 출발점 후보 (층·섬별) ───────────────────────────────────
    pts = scenes.sample_storey_points(pf, sc.levels, sc.lv2st, per_bucket=400, seed=a.seed + 1)
    by_island = {}                   # 섬 -> [habitat 점]
    for (st, isl), v in pts.items():
        by_island.setdefault(isl, []).extend(v)
    st_of = {}                       # 섬 -> 거기 있는 층들
    for (st, isl) in pts:
        st_of.setdefault(isl, set()).add(st)

    def storey_of(z):
        k = int(np.argmin([abs(L['z'] - z) for L in sc.levels])) if sc.levels else 0
        return int(sc.lv2st[k]) if len(sc.lv2st) else 0

    def make(goal_xyz, want_multifloor, lm, sp, starts, why):
        """목적지까지 오는 경로 하나. 못 만들면 None. 실패 이유는 why(Counter) 에 goal_rules.REASONS 코드로 센다."""
        q_goal = nu.ours_to_hab(goal_xyz)
        isl = int(pf.get_island(pf.snap_point(q_goal)))
        cand = by_island.get(isl)
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
            row, r = one(sc, P, md, r_b, cfg)
            X = r['traj'].astype(np.float64)
            st_mask = np.asarray(r['stairs'], bool)
            if st_mask[0] or st_mask[-1]:          # 출발·도착이 계단 한가운데
                why['START_ON_STAIRS' if st_mask[0] else 'GOAL_ON_STAIRS'] += 1
                continue
            if float(row['ref_length_m']) > a.max_path_m:
                why['TOO_LONG'] += 1
                continue
            # 가는 길에 같은 종류 — 기준점이 있는 목적지(반복 물체)는 기준점으로 가리키므로 세지 않는다
            passed = [] if sp.get('anchor') else gr.passed_same(objs, lm['oid'], X, goal_xyz, a.pass_r, a.succ_r)
            if len(passed) > a.max_pass:
                why['DISTRACTOR_MANY'] += 1
                continue
            # 층 이동 = 경로가 다른 층(storey)을 지난다. 갔다가 돌아오는 경로도 층 이동이다.
            # 높이차 0.5 m 로 가르면 기울어진 바닥이 층 이동으로 잡힌다 (층 사이는 storey_gap 1.5 m 이상).
            kind = 'multifloor' if len({storey_of(z) for z in X[:, 2]}) > 1 else 'flat'
            bad = judge(row, X, kind, holes, r_b, a)
            if bad:
                for b in bad:
                    why[gr.JUDGE_CODE[b.split()[0]]] += 1
                continue
            return dict(kind=kind, X=X, st=st_mask, row=row, passed=passed,
                        from_storey=storey_of(X[0, 2]), to_storey=g_st)
        return None

    # judge 가 보는 U5 인자
    a.flat_min_m = a.min_m

    # ── 만들기 ──────────────────────────────────────────────────
    eps, fail = [], 0
    fail_log = {}                    # '랜드마크번호:spot번호' -> {코드: 횟수}
    total = collections.Counter()
    pick = rng.permutation(len(goals))
    for n, gi in enumerate(pick):
        li, lm, sp = goals[int(gi)]
        is_eval = int(li) in eval_lm
        tgt = a.n_eval if is_eval else a.n_train
        have = sum(1 for e in eps if e['split'] == ('eval' if is_eval else 'train'))
        if have >= tgt:
            continue
        if len(eps) >= need:
            break
        k_goal = 1 if is_eval else a.paths_per_goal       # 평가는 목적지당 1 개 (한 장소에 점수가 쏠리지 않게)
        g = np.asarray(sp['xyz'], float)
        why, starts = collections.Counter(), []
        for _ in range(min(k_goal, tgt - have)):
            want_mf = bool(rng.random() < a.multifloor_frac)
            got1 = make(g, want_mf, lm, sp, starts, why) or make(g, not want_mf, lm, sp, starts, why)
            if got1 is None:
                fail += 1
                break
            starts.append(got1['X'][0])
            got1.update(split='eval' if is_eval else 'train', landmark=li, lm=lm, spot=sp)
            eps.append(got1)
        if why:
            fail_log[f"{lm['id']}:{lm['spots'].index(sp)}"] = dict(why)
            total.update(why)
        if (n + 1) % 100 == 0 or len(eps) == need:
            ne = sum(1 for e in eps if e['split'] == 'eval')
            print(f'  {len(eps)}/{need}  (평가 {ne} · 학습 {len(eps) - ne} · 실패 {fail})', flush=True)

    # ── 내보내기 (22 와 같은 모양 + goal 정보) ───────────────────
    episodes = []
    for k, e in enumerate(eps):
        X, lm, sp = e['X'], e['lm'], e['spot']
        episodes.append({
            'episode_id': k, 'split': e['split'], 'kind': e['kind'],
            'from_storey': e['from_storey'], 'to_storey': e['to_storey'], 'source': '74',
            'start': X[0].tolist(), 'goal': X[-1].tolist(),
            'trajectory': X.tolist(), 'trajectory_z': X[:, 2].tolist(), 'floor_z': float(X[0, 2]),
            'segments': segments(e['st'], X[:, 2]),
            'path_length_m': float(e['row']['ref_length_m']),
            'mesh_clear_min_m': float(e['row']['ref_mesh_clear_min_m']),
            'margin_min_m': float(e['row']['ref_margin_min_m']),
            'stairs_frac': float(np.mean(e['st'])),
            # 멈출 대상 — 이것 때문에 74 를 만들었다
            'goal_yaw': float(sp['yaw']),
            'stop_cue': {'landmark_id': int(e['landmark']), 'landmark_ids': lm.get('ids', [lm['id']]),
                         'part': sp.get('part', 'whole'), 'target_xyz': sp.get('target_xyz', lm['xyz']),
                         'anchor': sp.get('anchor'),
                         # 가는 길에 지나친 같은 종류 (순서대로). ordinal = 목표가 몇 번째인가
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

    # 자체 검사: 위 규칙이 결과에 지켜졌는가
    for e in eps:
        assert not (e['st'][0] or e['st'][-1]), '출발/도착이 계단 위'
        assert not on_stairs(e['X'][-1]), '도착이 계단 위'
        assert float(e['row']['ref_length_m']) <= a.max_path_m, '경로가 너무 김'

    (out / 'fail_reasons.json').write_text(json.dumps(
        {'reasons': gr.REASONS, 'total': dict(total), 'per_goal': fail_log}, ensure_ascii=False, indent=1))
    d = [e['stop_cue']['dist_m'] for e in episodes]
    L = [e['path_length_m'] for e in episodes]
    print(f'\n[74] 경로 {len(episodes)} (평가 {len(ev)} · 학습 {len(tr)}) · 출발점 못 찾음 {fail}')
    if d:
        print(f'[74] 목적지~멈출 대상 거리  중앙 {np.median(d):.2f} m · 최대 {max(d):.2f} m')
        print(f'[74] 경로 길이              중앙 {np.median(L):.1f} m · 최대 {max(L):.1f} m')
        print(f'[74] 층 이동 {sum(1 for e in episodes if e["kind"] == "multifloor")} · '
              f'평지 {sum(1 for e in episodes if e["kind"] == "flat")}')
        print(f'[74] 쓰인 랜드마크 {len({e["stop_cue"]["landmark_id"] for e in episodes})} 개')
    print('[74] 실패 이유 (시도 횟수):')
    for k, v in total.most_common():
        print(f'       {k:<16} {v:>6}  {gr.REASONS[k]}')
    na = sum(1 for e in episodes if e['stop_cue']['anchor'])
    no = collections.Counter(e['stop_cue']['ordinal'] for e in episodes)
    print(f'[74] 기준점 붙은 경로 {na} · 목표가 n 번째 {dict(sorted(no.items()))} · 양 끝 목표 '
          f'{sum(1 for e in episodes if e["stop_cue"]["part"] == "end")}')
    print(f'[74] 저장 {out}/noeun_station_paths.json · split.json · fail_reasons.json')


if __name__ == '__main__':
    main()
