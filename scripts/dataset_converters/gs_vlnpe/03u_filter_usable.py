"""03u — 03 결과에서 **사용 가능한 경로만 남기고 나머지는 버린다**. 렌더(04)는 이 결과만 읽는다.

    03:  occupancy  -> paths/<scene>.json           (생성 시도 전부 — 분석용, 그대로 둔다)
    03u: paths      -> usable/paths/<scene>.json    (사용 가능만 — 렌더 입력)

판정 (GT 에피소드 하나는 정확히 한 칸에 들어간다)
------------------------------------------------
| 칸         | 조건                                                            |
|------------|-----------------------------------------------------------------|
| 층 이동    | GT 카메라 |z_끝 - z_시작| > `--dz_m` (2D 계획으로 못 다룸 — 제외) |
| 퇴화 GT    | 03 이 `degenerate` 로 건너뜀 (5 프레임 미만)                    |
| 끝점 막힘  | `start_or_goal_not_navigable`                                   |
| A* 실패    | `astar_failed`                                                  |
| 결과 없음  | paths json 도 run log 도 없어 사유를 모름                       |
| 관통       | 생성됨, `check_trajectory.points_min_m <= 0`                    |
| r_b 미달   | 생성됨, `0 < points_min_m < r_b`                                |
| **사용 가능** | 생성됨, `points_min_m >= r_b`                                |

`points_min_m` 은 03 이 `esdf_utils.check_path_navigable` 로 잰 값이다 — 0.05 m ESDF 에서
궤적을 셀의 1/5(1 cm) 간격으로 촘촘히 찍은 **벽까지 최소 거리**. `r_b` 는 기본으로 03 이 paths json
`params.r_b` 에 남긴 값을 쓴다. "같은루트"는 참고 지표로만 센다(instruction 을 새로 붙이므로 필터 아님).

층 이동: 카메라 높이는 에피소드 안에서 일정하므로 z 차가 곧 바닥 높이 차다. vln_ce 61씬 1,592 / 10,684
(14.9 %), vln_pe 3 / 2,813 (2026-10-13). InternNav 도 VLN-PE 에서만 `filter_stairs` 로 계단을 뺀다.

분모는 GT parquet 수다. 03 은 성공 0건인 씬의 paths json 을 남기지 않으므로 `<out_dir>/run_logs/<scene>.log`
(03 stdout) 의 "실패 사유" 로 세고, 없으면 "결과 없음" 으로 센다(조용히 빠지지 않게).

출력 (`<out_dir>/usable/`)
    paths/<scene><sfx>.json     사용 가능 에피소드만 남긴 03 json (+ `usable_filter`). 0건 씬은 쓰지 않는다
                                (이전 실행의 같은 씬 파일은 지운다).
    usable.json                 칸별 개수 + `{"usable": {scene: [episode_id, ...]}}`
    multifloor_episodes.json    층 이동 목록

다음 단계: `04_render_obs_isaac.py --mode reproduce --out_dir <out_dir>/usable`.

사용:
    python 03u_filter_usable.py --dataset vln_ce --rig 125cm_30deg --out_dir <03 의 --out_dir>
    python 03u_filter_usable.py --dataset vln_pe --out_dir <03 의 --out_dir>
    python 03u_filter_usable.py --dataset vln_ce --rig 125cm_30deg --out_dir <out> --multifloor_mode stairs
        (03mf_plan_multifloor.py 의 <out>/paths_mf/ 로 층 이동도 판정 — multifloor_filter.py)
"""

import argparse
import ast
import glob
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset_utils import default_data_root  # noqa: E402

_spec = importlib.util.spec_from_file_location('m03', HERE / '03_sample_gt_paths.py')
m03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m03)

CELLS = ['multifloor', 'degenerate', 'sg_blocked', 'astar_failed', 'no_result', 'hard', 'below_rb', 'usable']
LABELS = {'multifloor': '층 이동(제외)', 'degenerate': '퇴화 GT', 'sg_blocked': '끝점 막힘',
          'astar_failed': 'A* 실패', 'no_result': '결과 없음', 'hard': '관통', 'below_rb': 'r_b 미달',
          'usable': '사용 가능'}


def fail_cell(reason: str) -> str:
    if reason.startswith('degenerate'):
        return 'degenerate'
    if reason.startswith('start_or_goal'):
        return 'sg_blocked'
    if reason.startswith('astar'):
        return 'astar_failed'
    assert False, f'unreachable failure reason {reason!r}'


def multifloor_episodes(gt_root: Path, scenes, dataset: str, rig, dz_m: float) -> dict:
    out = {}
    for s in scenes:
        out[s] = []
        for ep in range(len(glob.glob(str(gt_root / s / 'data' / 'chunk-000' / '*.parquet')))):
            xyz = m03.load_gt_episode(str(gt_root), s, ep, dataset, rig)[0]
            if abs(float(xyz[-1, 2] - xyz[0, 2])) > dz_m:
                out[s].append(ep)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=['vln_ce', 'vln_pe'])
    ap.add_argument('--out_dir', required=True, help='03 의 --out_dir (paths/, run_logs/ 를 읽는다)')
    ap.add_argument('--rig', default=None, help='vln_ce 전용 카메라 rig (03 과 같게)')
    ap.add_argument('--r_b', type=float, default=None, help='생략 시 paths json 의 params.r_b (03 의 --r_b)')
    ap.add_argument('--dz_m', type=float, default=0.5, help='층 이동 판정 z 차 [m]')
    ap.add_argument('--multifloor', default=None, help='이미 만든 층 이동 목록 json — 주면 다시 계산하지 않는다')
    ap.add_argument('--gt_root', default=None, help='생략 시 dataset_utils.default_data_root')
    ap.add_argument('--scenes', default=None, help='콤마 구분. 생략 시 gt_root 의 모든 씬')
    ap.add_argument('--multifloor_mode', default='off', choices=['off', 'stairs'],
                    help="off(기본)=층 이동 제외(종전 동작). stairs=03mf 의 paths_mf/ 로 층 이동도 판정해 "
                         '사용 가능을 포함한다(multifloor_filter.py)')
    args = ap.parse_args()
    if args.multifloor_mode == 'stairs':
        import multifloor_filter
        return multifloor_filter.main_stairs(args)

    out_dir = Path(args.out_dir)
    paths_dir, logs_dir, dst = out_dir / 'paths', out_dir / 'run_logs', out_dir / 'usable'
    gt_root = Path(args.gt_root or default_data_root(args.dataset))
    sfx = '_vlnce' if args.dataset == 'vln_ce' else '_vlnpe'
    scenes = args.scenes.split(',') if args.scenes else sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    (dst / 'paths').mkdir(parents=True, exist_ok=True)

    if args.multifloor:
        mf_eps = json.load(open(args.multifloor))['episodes']
    else:
        mf_eps = multifloor_episodes(gt_root, scenes, args.dataset, args.rig, args.dz_m)
        json.dump({'rule': f'|z_goal - z_start| > {args.dz_m} m (camera z)', 'dataset': args.dataset,
                   'episodes': mf_eps}, open(dst / 'multifloor_episodes.json', 'w'))
    mf = {s: set(v) for s, v in mf_eps.items()}

    cnt = Counter()
    usable, same, mins, turns, r_bs = {}, 0, [], [], set()
    for s in scenes:
        n_gt = len(glob.glob(str(gt_root / s / 'data' / 'chunk-000' / '*.parquet')))
        m = mf.get(s, set())
        cnt['multifloor'] += len(m)
        seen = set(m)
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
            seen.add(k); cnt[fail_cell(v)] += 1
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
        q = dst / 'paths' / f'{s}{sfx}.json'
        if usable.get(s):
            keep = set(usable[s])
            j['episodes'] = [e for e in j['episodes'] if e['episode_id'] in keep]
            j['usable_filter'] = {'rule': f'check_trajectory.points_min_m >= r_b ({r_b})',
                                  'multifloor_excluded': True, 'source': str(p), 'n_kept': len(j['episodes'])}
            json.dump(j, open(q, 'w'))
        elif q.is_file():
            q.unlink()

    flat = cnt['n_gt'] - cnt['multifloor']
    assert sum(cnt[c] for c in CELLS) == cnt['n_gt'], dict(cnt)
    assert len(r_bs) <= 1, f'씬마다 r_b 가 다르다 {sorted(r_bs)} — --r_b 로 지정'
    r_b_txt = f'{next(iter(r_bs))}' if r_bs else '-'
    print(f'[03u_filter_usable] {args.dataset} · {len(scenes)}씬 · GT {cnt["n_gt"]} · 평지 {flat} · r_b {r_b_txt}')
    for c in CELLS:
        base = cnt['n_gt'] if c == 'multifloor' else flat
        print(f'  {LABELS[c]:<12}: {cnt[c]:>6}  ({100 * cnt[c] / max(base, 1):5.1f}%)')
    if mins:
        near = sum(x < next(iter(r_bs)) + 0.05 - 1e-9 for x in mins)
        print(f'  사용 가능 중 같은루트(참고) {same} ({100 * same / len(mins):.1f}%) · 최소 여유 중앙 '
              f'{statistics.median(mins):.3f} m · 여유 < r_b+0.05 {near} ({100 * near / len(mins):.0f}%) · '
              f'회전 중앙 {statistics.median(turns):.1f} °/m')
    json.dump({'dataset': args.dataset, 'r_b': next(iter(r_bs)) if r_bs else None, 'paths_dir': str(paths_dir),
               'counts': {c: cnt[c] for c in CELLS} | {'n_gt': cnt['n_gt'], 'flat': flat},
               'usable': usable}, open(dst / 'usable.json', 'w'), indent=1)
    print(f'  -> 사용 가능만 남긴 paths: {dst / "paths"} ({len(usable)}씬 · {cnt["usable"]} ep) · {dst / "usable.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
