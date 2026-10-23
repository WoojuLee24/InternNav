"""03u `--multifloor_mode stairs` 전용 — 층 이동 에피소드를 버리지 않고 판정한다.

03u 의 기본 동작(off)은 그대로 두고, `main()` 맨 앞 guard 가 여기로 넘긴다.
평지 판정은 03u 와 **같은 규칙**(같은 `fail_cell`, `multifloor_episodes`, 같은 순서)으로 하고,
층 이동 에피소드는 03mf 의 `<out_dir>/paths_mf/<scene><sfx>.json` 으로 판정한다.

층 이동 칸 (서로 겹치지 않음, 합 = 03u 의 '층 이동' 칸 수)
| 칸                 | 뜻                                                                 |
|--------------------|--------------------------------------------------------------------|
| mf_degenerate      | GT 5 프레임 미만                                                   |
| mf_entry_blocked   | 계단 입구/출구를 GT 따라 물려도 그 층 지도에서 통행 가능한 점 없음   |
| mf_sg_blocked      | 층 지도에서 A* 시작/끝(GT start·goal, 물린 입구·출구)이 통행 불가     |
| mf_astar_failed    | 층 지도에서 연결 경로 없음                                          |
| mf_no_floor_plan   | 평지 구간이 전부 짧아 계획한 곳이 없음(GT 복사본)                    |
| mf_stairs_no_path  | (03mf --mf_stairs plan) 높이 지도에서 계단 아래 -> 위 경로 없음       |
| mf_no_result       | paths_mf 에 없음(03mf 미실행 등)                                    |
| mf_hard            | 생성됨, 최소 여유 <= 0 (gt 모드: 평지 부분만 / plan 모드: 평지+계단)   |
| mf_below_rb        | 생성됨, 0 < 평지 부분 최소 여유 < r_b                               |
| **mf_usable**      | 생성됨, 평지 부분 최소 여유 >= r_b (계단 구간은 GT 라 통과로 본다)   |
"""

import ast
import glob
import importlib.util
import json
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('m03u', HERE / '03u_filter_usable.py')
m03u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03u)

MF_CELLS = ['mf_degenerate', 'mf_entry_blocked', 'mf_sg_blocked', 'mf_astar_failed', 'mf_no_floor_plan',
            'mf_stairs_no_path', 'mf_no_result', 'mf_hard', 'mf_below_rb', 'mf_usable']
MF_LABELS = {'mf_degenerate': '퇴화 GT', 'mf_entry_blocked': '계단 입구 막힘', 'mf_sg_blocked': '층 끝점 막힘',
             'mf_astar_failed': '층별 A* 실패', 'mf_no_floor_plan': '평지 계획 없음',
             'mf_stairs_no_path': '계단 길 못 찾음', 'mf_no_result': '결과 없음',
             'mf_hard': '관통', 'mf_below_rb': 'r_b 미달', 'mf_usable': '사용 가능'}


def mf_fail_cell(reason: str) -> str:
    if reason.startswith('degenerate'):
        return 'mf_degenerate'
    if reason in ('mf_entry_blocked', 'mf_sg_blocked', 'mf_astar_failed', 'mf_no_floor_plan', 'mf_stairs_no_path'):
        return reason
    assert False, f'unreachable multifloor failure reason {reason!r}'


def main_stairs(args) -> int:
    out_dir = Path(args.out_dir)
    paths_dir, logs_dir, dst = out_dir / 'paths', out_dir / 'run_logs', out_dir / 'usable'
    mf_dir = out_dir / 'paths_mf'
    gt_root = Path(args.gt_root or m03u.default_data_root(args.dataset))
    sfx = '_vlnce' if args.dataset == 'vln_ce' else '_vlnpe'
    scenes = args.scenes.split(',') if args.scenes else sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    (dst / 'paths').mkdir(parents=True, exist_ok=True)

    if args.multifloor:
        mf_eps = json.load(open(args.multifloor))['episodes']
    else:
        mf_eps = m03u.multifloor_episodes(gt_root, scenes, args.dataset, args.rig, args.dz_m)
        json.dump({'rule': f'|z_goal - z_start| > {args.dz_m} m (camera z)', 'dataset': args.dataset,
                   'episodes': mf_eps}, open(dst / 'multifloor_episodes.json', 'w'))
    mf = {s: set(v) for s, v in mf_eps.items()}

    cnt, mcnt = Counter(), Counter()
    usable, usable_mf, same, mins, turns, r_bs = {}, {}, 0, [], [], set()
    m_same, m_mins, m_stairs = 0, [], Counter()
    for s in scenes:
        n_gt = len(glob.glob(str(gt_root / s / 'data' / 'chunk-000' / '*.parquet')))
        m = mf.get(s, set())
        cnt['multifloor'] += len(m)
        seen = set(m)
        # --- 평지: 03u main() 과 같은 판정 ---
        p = paths_dir / f'{s}{sfx}.json'
        j = None
        if p.is_file():
            j = json.load(open(p))
            fails, eps = j['stats']['failures'], j['episodes']
            r_b = args.r_b if args.r_b is not None else j['params']['r_b']
            r_bs.add(r_b)
        else:
            log = logs_dir / f'{s}.log'
            hit = re.search(r'실패 사유: (\{.*\})', log.read_text()) if log.is_file() else None
            fails, eps = (ast.literal_eval(hit.group(1)) if hit else {}), []
        for k, v in fails.items():
            k = int(k)
            if k in seen:
                continue
            seen.add(k); cnt[m03u.fail_cell(v)] += 1
        for e in eps:
            k = e['episode_id']
            if k in seen:
                continue
            seen.add(k)
            t = e['check_trajectory']['points_min_m']
            if t <= 0:
                cnt['hard'] += 1
            elif t < r_b - 1e-9:
                cnt['below_rb'] += 1
            else:
                cnt['usable'] += 1
                usable.setdefault(s, []).append(k)
                same += bool(e['gt_compare']['same_route'])
                mins.append(t); turns.append(e['gt_compare']['turn_per_m_deg'])
        cnt['no_result'] += n_gt - len(seen)
        cnt['n_gt'] += n_gt

        # --- 층 이동: paths_mf 로 판정 ---
        q_mf = mf_dir / f'{s}{sfx}.json'
        jm = json.load(open(q_mf)) if q_mf.is_file() else None
        m_seen, keep_mf = set(), []
        if jm is not None:
            r_b_m = args.r_b if args.r_b is not None else jm['params']['r_b']
            r_bs.add(r_b_m)
            for k, v in jm['stats']['failures'].items():
                k = int(k)
                if k in m and k not in m_seen:
                    m_seen.add(k); mcnt[mf_fail_cell(v)] += 1
            for e in jm['episodes']:
                k = e['episode_id']
                if k not in m or k in m_seen:
                    continue
                m_seen.add(k)
                t = e['check_trajectory']['points_min_m']
                if t <= 0:
                    mcnt['mf_hard'] += 1
                elif t < r_b_m - 1e-9:
                    mcnt['mf_below_rb'] += 1
                else:
                    mcnt['mf_usable'] += 1
                    usable_mf.setdefault(s, []).append(k)
                    keep_mf.append(e)
                    m_same += bool(e['gt_compare']['same_route'])
                    m_mins.append(t); m_stairs[e['multifloor']['n_stairs']] += 1
        mcnt['mf_no_result'] += len(m) - len(m_seen)

        # --- 출력: 평지 사용 가능 + 층 이동 사용 가능 ---
        q = dst / 'paths' / f'{s}{sfx}.json'
        if usable.get(s) or keep_mf:
            if j is None:
                j = {k: v for k, v in jm.items() if k != 'episodes'}
                j['episodes'] = []
            keep = set(usable.get(s, []))
            j['episodes'] = sorted([e for e in j['episodes'] if e['episode_id'] in keep] + keep_mf,
                                   key=lambda e: e['episode_id'])
            j['usable_filter'] = {'rule': f'check_trajectory.points_min_m >= r_b ({r_b_txt_of(r_bs)})',
                                  'multifloor_excluded': False, 'source': str(p),
                                  'multifloor_source': str(q_mf), 'n_kept': len(j['episodes']),
                                  'n_kept_multifloor': len(keep_mf)}
            json.dump(j, open(q, 'w'))
        elif q.is_file():
            q.unlink()

    flat = cnt['n_gt'] - cnt['multifloor']
    assert sum(cnt[c] for c in m03u.CELLS) == cnt['n_gt'], dict(cnt)
    assert sum(mcnt[c] for c in MF_CELLS) == cnt['multifloor'], (dict(mcnt), cnt['multifloor'])
    assert len(r_bs) <= 1, f'씬마다 r_b 가 다르다 {sorted(r_bs)} — --r_b 로 지정'
    r_b_txt = r_b_txt_of(r_bs)
    print(f'[03u_filter_usable --multifloor_mode stairs] {args.dataset} · {len(scenes)}씬 · GT {cnt["n_gt"]} · '
          f'평지 {flat} · 층 이동 {cnt["multifloor"]} · r_b {r_b_txt}')
    print(' 평지')
    for c in m03u.CELLS:
        if c == 'multifloor':
            continue
        print(f'  {m03u.LABELS[c]:<12}: {cnt[c]:>6}  ({100 * cnt[c] / max(flat, 1):5.1f}%)')
    if mins:
        near = sum(x < next(iter(r_bs)) + 0.05 - 1e-9 for x in mins)
        print(f'  사용 가능 중 같은루트(참고) {same} ({100 * same / len(mins):.1f}%) · 최소 여유 중앙 '
              f'{statistics.median(mins):.3f} m · 여유 < r_b+0.05 {near} ({100 * near / len(mins):.0f}%) · '
              f'회전 중앙 {statistics.median(turns):.1f} °/m')
    print(' 층 이동')
    for c in MF_CELLS:
        print(f'  {MF_LABELS[c]:<12}: {mcnt[c]:>6}  ({100 * mcnt[c] / max(cnt["multifloor"], 1):5.1f}%)')
    if m_mins:
        print(f'  사용 가능 중 같은루트(참고) {m_same} ({100 * m_same / len(m_mins):.1f}%) · 최소 여유 중앙(gt=평지만, plan=평지+계단) '
              f'{statistics.median(m_mins):.3f} m · 계단 개수 {dict(sorted(m_stairs.items()))}')
    n_all = cnt['usable'] + mcnt['mf_usable']
    stairs_mode = 'plan' if any(json.load(open(f))['params']['multifloor'].get('stairs') == 'plan'
                                for f in sorted(mf_dir.glob(f'*{sfx}.json'))[:1]) else 'gt'
    all_usable = {s: sorted(usable.get(s, []) + usable_mf.get(s, [])) for s in sorted(set(usable) | set(usable_mf))}
    json.dump({'dataset': args.dataset, 'r_b': next(iter(r_bs)) if r_bs else None, 'paths_dir': str(paths_dir),
               'counts': {c: cnt[c] for c in m03u.CELLS} | {'n_gt': cnt['n_gt'], 'flat': flat},
               'usable': all_usable,
               'multifloor_mode': 'stairs', 'paths_mf_dir': str(mf_dir),
               'counts_multifloor': {c: mcnt[c] for c in MF_CELLS},
               'stairs_mode': stairs_mode,
               'usable_rate': {'all': n_all / max(cnt['n_gt'], 1), 'flat_of_flat': cnt['usable'] / max(flat, 1),
                               'multifloor_of_multifloor': mcnt['mf_usable'] / max(cnt['multifloor'], 1)},
               'usable_multifloor': usable_mf}, open(dst / 'usable.json', 'w'), indent=1)
    print(f' 전체 (분모 = GT {cnt["n_gt"]}, 계단 {stairs_mode} 모드): 사용 가능 {n_all} ({100 * n_all / max(cnt["n_gt"], 1):.1f}%) = '
          f'평지 {cnt["usable"]} ({100 * cnt["usable"] / max(cnt["n_gt"], 1):.1f}%) + '
          f'층 이동 {mcnt["mf_usable"]} ({100 * mcnt["mf_usable"] / max(cnt["n_gt"], 1):.1f}%)')
    print(f'  -> 사용 가능만 남긴 paths: {dst / "paths"} ({len(all_usable)}씬 · 평지 {cnt["usable"]} + '
          f'층 이동 {mcnt["mf_usable"]} = {n_all} ep) · {dst / "usable.json"}')
    return 0


def r_b_txt_of(r_bs) -> str:
    return f'{next(iter(r_bs))}' if r_bs else '-'
