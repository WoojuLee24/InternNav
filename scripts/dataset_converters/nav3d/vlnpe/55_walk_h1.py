"""06r — nav3d 경로를 **H1 로봇이 물리 시뮬레이션으로 걸어** vln_pe 포맷으로 저장한다 (06p 의 카메라만 놓는 방식과 나란히 둘 다 쓸 수 있게).

vln_pe 릴리스는 H1 이 GT reference_path 를 move_by_flash(전진 0.25 m / 회전 15°)로 따라가며 찍은 것이다
(`scripts/visualization/eval_gt_collect.py` 와 같은 규칙). 여기서는 같은 규칙을 nav3d 쪽에 두고, 경로만 바꾼다.

    stage prepare (habitat 컨테이너): 같은 에피소드로 raw split 두 개
        nav3d_gt : GT reference_path 그대로            (같은 세션 기준선)
        nav3d_sg : nav3d 03 sg 경로를 0.5 m 경유점으로  (시작 위치·시작 heading = GT 그대로)
    stage walk (isaac 컨테이너): split 마다 H1 을 걷게 해 vln_pe 포맷으로 저장 + 결과 json
    stage compare (habitat 컨테이너): 성공률 · 도착 오차 · H1 궤적 ↔ 경로 · sg 와 gt 비교

## 용어
- **H1 걷기**: InternUtopia H1 로봇 + move_by_flash 제어기. 각 경유점까지 "목표 방향과 7.5° 넘게 어긋나면 15° 회전, 아니면 0.25 m 전진".
- **성공**: 멈춘 위치가 목표 3 m 안 (vln_pe 평가와 같음). **NE**: 멈춘 위치 ↔ 목표 거리.

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 |
|---|---|---|
| W1 sg 완주 | H1 이 sg 경로를 GT 경로만큼 완주하나 (같은 세션) | 성공률(sg) ≥ 성공률(gt) − 10 %p |
| W2 sg 따라가기 | 성공한 sg ep 에서 H1 궤적이 sg 경로를 따르나 | H1↔sg 같은 루트(Fréchet < 0.8 m) ≥ 90 % |
| W3 형식 | 저장 결과가 학습 로더로 읽히나 | 08 vln_pe 검사 통과 |

사용 (저장소 루트):
    (habitat) /usr/bin/python scripts/dataset_converters/nav3d/vlnpe/55_walk_h1.py --stage prepare
    (isaac)   /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/vlnpe/55_walk_h1.py --stage walk --split nav3d_gt
    (isaac)   /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/vlnpe/55_walk_h1.py --stage walk --split nav3d_sg
    (habitat) /usr/bin/python scripts/dataset_converters/nav3d/vlnpe/55_walk_h1.py --stage compare
"""

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / 'third_party' / 'diffusion-policy'), str(HERE), str(HERE.parent / 'gs_vlnpe'),
          str(HERE.parent / 'instruction_labeling' / 'render')):
    if p not in sys.path:
        sys.path.insert(0, p)

D = 'data/InternData-N1-v0.5-mini'
LOG = 'scripts/dataset_converters/nav3d/logs/vlnpe/55_walk_h1'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['prepare', 'walk', 'compare'])
    ap.add_argument('--split', default='nav3d_sg', help='walk: nav3d_gt | nav3d_sg')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/common/episodes.json', help='(씬, vln_ce 릴리스 episode_index)')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths/scenes')
    ap.add_argument('--waypoint_m', type=float, default=0.5, help='sg 경로 -> H1 경유점 간격 (06 이산화와 같은 0.5 m)')
    ap.add_argument('--raw_root', default=f'{D}/vln_pe_nav3d_raw/r2r', help='prepare 출력 = walk 의 base_data_dir')
    ap.add_argument('--config', default='scripts/dataset_converters/nav3d/vlnpe/configs/h1_vlnpe_walk_cfg.py',
                    help='H1 env 설정. 기본 = vln_pe 카메라(h1_pano_camera_0 256²) · 물리 모드')
    ap.add_argument('--robot_flash', default=None, choices=['0', '1'],
                    help='None = 설정 파일 값 / 1 = 위치를 바로 옮김(물리 없음) / 0 = 물리로 걸음. 릴리스 vln_pe 는 물리 (몸 흔들림·이탈이 있다)')
    ap.add_argument('--filter_stairs', action='store_true', help='vln_pe 평가처럼 계단 ep 를 뺀다 (기본: 안 뺀다 — 계단도 본다)')
    ap.add_argument('--ne_threshold', type=float, default=3.0)
    ap.add_argument('--h1_height_clip', type=float, default=None,
                    help='None = 원본 컨트롤러 (발밑 높이 바닥 ±0.2 m 밖은 바닥으로 덮음). 값 = 그 폭으로 (h1_tunable_controller.py) — 계단 진단용')
    ap.add_argument('--h1_steps_per_action', type=int, default=None,
                    help='None = 50 (릴리스: action 당 50 step @ 200 Hz → 1.0 m/s · 60°/s). 100 = 같은 0.25 m / 15° 를 0.5 m/s · 30°/s 로')
    ap.add_argument('--only_stairs', action='store_true', help='walk: raw split 에서 계단 ep (경로 높이 범위 > 0.5 m) 만')
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--out_root', default=f'{D}/vln_pe_nav3d_h1/traj_data', help='<out_root>/<split>/<scene>/ (vln_pe 레이아웃)')
    ap.add_argument('--log_dir', default=LOG)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def resample(P, step):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    s = np.r_[np.arange(0, L[-1], step), L[-1]]
    return np.stack([np.interp(s, L, P[:, k]) for k in range(3)], 1)


def prepare(args):
    import nav3d_utils as nu
    from release_io import load_cfg, match_raw, release_episodes
    rc = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')
    sel = json.loads(Path(args.episodes_json).read_text())
    out = {'nav3d_gt': [], 'nav3d_sg': []}
    for scene, ids in sel.items():
        rel = {e['episode_index']: e for e in release_episodes(rc, scene)}
        Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        for ep in ids:
            if f'{ep}_sg_ref' not in Z.files:
                continue
            raw = match_raw(rc, scene, rel[ep]['task'])
            e = {k: raw[k] for k in ('episode_id', 'trajectory_id', 'scene_id', 'start_position', 'start_rotation', 'info', 'goals', 'instruction', 'reference_path')}
            e['scene_id'] = f'mp3d/{scene}/{scene}.glb'           # vln_pe raw 규약
            e['nav3d'] = {'scene': scene, 'ce_ep': int(ep)}
            out['nav3d_gt'].append(e)
            W = resample(Z[f'{ep}_sg_ref'].astype(np.float64), args.waypoint_m)
            Wh = nu.ours_to_hab(W)
            Wh[0] = np.asarray(raw['start_position'])            # 시작 위치 = GT 그대로
            es = dict(e, reference_path=Wh.tolist())
            out['nav3d_sg'].append(es)
    for split, eps in out.items():
        d = Path(args.raw_root) / split
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / f'{split}.json.gz', 'wt') as f:
            json.dump({'episodes': eps}, f)
        print(f'[06r] {split}: {len(eps)} ep -> {d}/{split}.json.gz', flush=True)


# ---------------------------------------------------------------------------
# walk (isaac) — eval_gt_collect.py 와 같은 제어 규칙
# ---------------------------------------------------------------------------

def walk(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location('eval_config_module', REPO / args.config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    eval_cfg = mod.eval_cfg
    from internnav.configs.evaluator.vln_default_config import get_config
    eval_cfg.task.task_name = f'nav3d_h1_{args.split}'
    ds = eval_cfg.dataset.dataset_settings
    ds['base_data_dir'] = str((REPO / args.raw_root).resolve())
    split = args.split
    if args.only_stairs:                                   # 계단 ep 만 담은 split 을 옆에 만든다
        src = json.load(gzip.open(REPO / args.raw_root / args.split / f'{args.split}.json.gz'))['episodes']
        keep = [e for e in src if np.ptp([p[1] for p in e['reference_path']]) > 0.5]
        split = f'{args.split}_stairs'
        (REPO / args.raw_root / split).mkdir(parents=True, exist_ok=True)
        with gzip.open(REPO / args.raw_root / split / f'{split}.json.gz', 'wt') as f:
            json.dump({'episodes': keep}, f)
        print(f'[06r] 계단 ep {len(keep)}/{len(src)} -> split {split}', flush=True)
    ds['split_data_types'] = [split]
    ds['filter_stairs'] = bool(args.filter_stairs)
    if args.robot_flash is not None:
        eval_cfg.task.robot_flash = args.robot_flash == '1'
    eval_cfg = get_config(eval_cfg)
    _tune_controllers(eval_cfg, args)
    for k, v in (('rank', 0), ('local_rank', 0), ('world_size', 1)):
        eval_cfg.env.env_settings[k] = v
    eval_cfg.env.env_settings['dataset'] = eval_cfg.dataset
    robot = eval_cfg.task.robot_name
    from internnav.env import Env
    import vlnpe_writer as vw
    env = Env.init(eval_cfg.env, eval_cfg.task)
    schema, stats_fn = vw.load_schema(REPO / vw.DEFAULT_SCHEMA_REF), vw.load_stats_fn()
    raw = {str(e['episode_id']): e for e in json.load(gzip.open(REPO / args.raw_root / split / f'{split}.json.gz'))['episodes']}
    out_root = Path(args.out_root) / args.split
    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    results, counts, frames = [], {}, {}
    # 물리 모드 = move_by_discrete (H1 걷기 정책, action 당 50 step @ 200 Hz — 릴리스 observation.step = 프레임 × 50 과 같다)
    # flash 모드 = move_by_flash (위치를 바로 옮김)
    ctrl = 'move_by_flash' if getattr(eval_cfg.task, 'robot_flash', False) else 'move_by_discrete'
    print(f'[06r {args.split}] 제어기 {ctrl}', flush=True)
    _, infos = env.reset()
    while infos and infos[0] is not None:
        info = infos[0].data
        if args.max_episodes and len(results) >= args.max_episodes:
            break
        ref = info['reference_path']
        scan = info['scan'] if 'scan' in info else info['scene_id'].split('/')[1]
        obs, _ = _step(env, [{robot: {'stand_still': []}}], robot)
        buf, term = [], False
        traj = [np.asarray(obs['globalgps'], dtype=np.float64).tolist()]
        for wi, wp in enumerate(ref[1:], start=1):
            obs, term = _goto(env, robot, wp, obs, buf, wi / max(len(ref), 1), traj, ctrl)
            if term:
                break
        if not term:
            obs, term = _step(env, [{robot: {'stop': []}}], robot)
        m = obs.get('metrics', {})
        if m:
            mm = m[list(m.keys())[0]][0]
            ne, ok, why = float(mm.get('NE', np.inf)), bool(mm.get('success', 0)), mm.get('fail_reason', '')
        else:
            ne = float(np.linalg.norm(np.asarray(obs['globalgps'])[:2] - np.asarray(ref[-1])[:2]))
            ok, why = ne < args.ne_threshold, 'no_metrics'
        e = raw.get(str(info['episode_id']), {})
        res = {'split': args.split, 'controller': ctrl, 'robot_flash': bool(getattr(eval_cfg.task, 'robot_flash', False)), 'scene': scan, 'episode_id': info['episode_id'], 'nav3d': e.get('nav3d'), 'ne': ne, 'success': ok,
               'fail_reason': why, 'steps': len(buf), 'h1_traj': traj, 'ref': [list(map(float, p)) for p in ref]}
        if ok and why == 'success' and buf:            # 넘어졌지만 3 m 안에서 멈춘 ep (fail_reason=fall) 는 저장 안 함 — 릴리스는 깨끗한 성공만
            ei = counts.get(scan, 0)
            for i, b in enumerate(buf):
                b['progress'] = float(i / max(len(buf) - 1, 1))
                # 릴리스 depth 는 (T,256,256) — 센서는 (256,256,1) 로 준다
                b['depth'] = np.clip(np.asarray(b['depth'], np.float32).reshape(np.asarray(b['depth']).shape[:2]), 1e-6, 1.0) if b['depth'] is not None else None
            if all(b['rgb'] is not None and b['depth'] is not None for b in buf):
                tok = list(info['instruction'].get('instruction_tokens', []))[:200]
                ins = [{'text': info['instruction']['instruction_text'], 'tokens': tok + [0] * (200 - len(tok))}]   # 로더는 200 개 고정
                vw.save_episode_vlnpe(out_root, scan, ei, buf, ins, schema=schema, stats_fn=stats_fn)
                counts[scan] = ei + 1
                frames[scan] = frames.get(scan, 0) + len(buf)
                res['saved_episode_index'] = ei
        results.append(res)
        print(f"[06r {args.split}] {scan} ep{info['episode_id']} NE {ne:.2f} 성공 {ok} steps {len(buf)} {why}", flush=True)
        (log / f'walk_{args.split}.json').write_text(json.dumps(results))
        if counts:                                         # env.close() 가 멈추므로 매 ep 마다 갱신
            vw.finalize_info_json_vlnpe(out_root, {s: {'episode_count': n, 'frame_count': frames.get(s, 0)} for s, n in counts.items()}, robot_type='h1')
        _, infos = env.reset([0])
    print(f'[06r {args.split}] {len(results)} ep · 성공 {sum(r["success"] for r in results)} · 저장 {sum(counts.values())}', flush=True)
    env.close()


def _tune_controllers(eval_cfg, args):
    """--h1_height_clip / --h1_steps_per_action 이 있을 때만 컨트롤러 설정을 바꾼다 (없으면 손대지 않음)."""
    if args.h1_height_clip is None and args.h1_steps_per_action is None:
        return
    if args.h1_height_clip is not None:
        import h1_tunable_controller as htc
        htc.HEIGHT_CLIP_UP = htc.HEIGHT_CLIP_DN = float(args.h1_height_clip)

    def fix(d):
        if isinstance(d, dict):
            if args.h1_height_clip is not None and d.get('type') == 'VlnMoveBySpeedController':
                d['type'] = 'Nav3dMoveBySpeedController'
            if args.h1_steps_per_action is not None and d.get('name') == 'move_by_discrete':
                d['steps_per_action'] = int(args.h1_steps_per_action)
            for v in d.values():
                fix(v)
        elif isinstance(d, list):
            for v in d:
                fix(v)
    for c in eval_cfg.task.robot.controllers:
        fix(c.controller_settings)
    print(f'[06r] 컨트롤러 조정: height_clip {args.h1_height_clip} · steps_per_action {args.h1_steps_per_action}', flush=True)


def _step(env, action, robot):
    while True:
        obs_list, _, terminated, _, _ = env.step(action)
        obs = obs_list[0][robot]
        if obs['finish_action'] or terminated[0]:
            return obs, terminated[0]


def _yaw(q):
    from omni.isaac.core.utils.rotations import quat_to_euler_angles
    return float(quat_to_euler_angles(np.array(q))[2])


def _goto(env, robot, wp, obs, buf, progress, traj, ctrl='move_by_flash'):
    """eval_gt_collect.navigate_to_waypoint 와 같은 규칙 (전진 0.25 / 회전 15° / 7.5° 문턱 / 0.15 m 도착)."""
    FWD, TURN = 0.25, math.radians(15.0)
    term = False
    for _ in range(400):
        cur = np.asarray(obs['globalgps'])[:2]
        if np.linalg.norm(np.asarray(wp)[:2] - cur) < FWD * 0.6:
            break
        dp = np.asarray(wp)[:2] - cur
        delta = ((math.atan2(dp[1], dp[0]) - _yaw(obs['globalrotation']) + math.pi) % (2 * math.pi)) - math.pi
        a = (2 if delta > 0 else 3) if abs(delta) > TURN * 0.5 else 1
        obs, term = _step(env, [{robot: {ctrl: [a]}}], robot)
        rp, ro = np.asarray(obs['globalgps']).copy(), np.asarray(obs['globalrotation']).copy()
        cp, co = np.asarray(obs.get('camera_position', rp)).copy(), np.asarray(obs.get('camera_orientation', ro)).copy()
        buf.append({'camera_position': cp, 'camera_orientation': co, 'camera_yaw': _yaw(co), 'robot_position': rp, 'robot_orientation': ro,
                    'robot_yaw': _yaw(ro), 'action': a, 'progress': progress,
                    'rgb': None if obs.get('rgb') is None else np.asarray(obs['rgb']).copy(),
                    'depth': None if obs.get('depth') is None else np.asarray(obs['depth']).copy()})
        traj.append(rp.tolist())
        if term:
            break
    return obs, term


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare(args):
    import nav3d_report as R
    import nav3d_utils as nu
    from esdf_utils import discrete_frechet_distance
    log = Path(args.log_dir)
    W = {s: {(r['nav3d']['scene'], r['nav3d']['ce_ep']): r for r in json.loads((log / f'walk_{s}.json').read_text()) if r.get('nav3d')}
         for s in ('nav3d_gt', 'nav3d_sg') if (log / f'walk_{s}.json').exists()}
    common = sorted(set.intersection(*[set(v) for v in W.values()])) if len(W) == 2 else []

    def fr(a, b):
        a, b = nu.densify(np.asarray(a)[:, :3], 0.1), nu.densify(np.asarray(b)[:, :3], 0.1)
        return float(discrete_frechet_distance(a[:, :2], b[:, :2]))
    rows = []
    for k in common:
        g, s = W['nav3d_gt'][k], W['nav3d_sg'][k]
        rows.append({'scene': k[0], 'ep': k[1], 'gt_ok': g['success'], 'sg_ok': s['success'], 'gt_ne': g['ne'], 'sg_ne': s['ne'],
                     'gt_why': g['fail_reason'], 'sg_why': s['fail_reason'],
                     'sg_follow': fr(s['h1_traj'], s['ref']) if len(s['h1_traj']) > 1 else np.inf,
                     'h1_sg_vs_h1_gt': fr(s['h1_traj'], g['h1_traj']) if len(s['h1_traj']) > 1 and len(g['h1_traj']) > 1 else np.inf})
    sr = lambda key: 100 * float(np.mean([r[key] for r in rows])) if rows else float('nan')  # noqa: E731
    ok_sg = [r for r in rows if r['sg_ok']]
    w1 = bool(rows) and sr('sg_ok') >= sr('gt_ok') - 10
    w2 = bool(ok_sg) and float(np.mean([r['sg_follow'] < 0.8 for r in ok_sg])) >= 0.9
    s08 = log / 'loader.json'
    w3 = s08.exists() and all(x['ok'] for x in json.loads(s08.read_text())['vln_pe'])
    gates = [
        {'검증': 'W1 sg 완주', '결과': R.pill(w1), '값': f"성공률 sg {sr('sg_ok'):.1f} % vs gt {sr('gt_ok'):.1f} % ({len(rows)} ep) · NE 중앙 sg {np.median([r['sg_ne'] for r in rows]) if rows else np.nan:.2f} / gt {np.median([r['gt_ne'] for r in rows]) if rows else np.nan:.2f} m", '기준': '≥ gt − 10 %p'},
        {'검증': 'W2 sg 따라가기', '결과': R.pill(w2), '값': f"성공 sg {len(ok_sg)} ep 중 H1↔sg 같은 루트 {100 * np.mean([r['sg_follow'] < 0.8 for r in ok_sg]) if ok_sg else np.nan:.1f} %", '기준': '≥ 90 %'},
        {'검증': 'W3 형식', '결과': R.pill(w3) if s08.exists() else R.pill('미실행'), '값': '08 vln_pe 로더', '기준': '통과'},
        {'검증': '(참고) H1(sg) ↔ H1(gt)', '결과': R.pill('참고'), '값': f"같은 루트 {100 * np.mean([r['h1_sg_vs_h1_gt'] < 0.8 for r in rows]) if rows else np.nan:.1f} %", '기준': '-'},
    ]
    fails = {}
    for r in rows:
        for k in ('gt', 'sg'):
            if not r[f'{k}_ok']:
                fails.setdefault(f"{k}: {r[f'{k}_why'] or '?'}", 0)
                fails[f"{k}: {r[f'{k}_why'] or '?'}"] += 1
    tab = [{'scene': r['scene'], 'ep': r['ep'], 'gt 성공/NE': f"{'O' if r['gt_ok'] else 'X'} {r['gt_ne']:.2f}", 'sg 성공/NE': f"{'O' if r['sg_ok'] else 'X'} {r['sg_ne']:.2f}",
            'H1↔sg Fréchet': f"{r['sg_follow']:.2f}", 'H1(sg)↔H1(gt)': f"{r['h1_sg_vs_h1_gt']:.2f}", '실패 사유 gt / sg': f"{r['gt_why']} / {r['sg_why']}"} for r in rows]
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': [w1, w2, w3], 'rows': rows, 'fail_reasons': fails}, open(log / 'summary.json', 'w'), indent=1, default=float)
    lead = f'같은 {len(rows)} ep 를 H1 이 GT 경로와 nav3d sg 경로로 각각 걸었다 (같은 세션, 시작 위치·heading = GT). <b>판정 {sum([w1, w2, w3])}/3 통과.</b>'
    body = R.header('nav3d 06r · H1 로 sg 걷기', 'H1 걷기 — GT 경로 vs nav3d sg', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다.</p>' + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('실패 사유', R.table([{'사유': k, 'ep': v} for k, v in sorted(fails.items())] or [{'사유': '없음', 'ep': 0}], ['사유', 'ep']))
    body += R.section('에피소드 별', R.table(tab, list(tab[0].keys())) if tab else '-')
    R.write(log, 'H1 걷기 GT vs sg', body)
    for g in gates:
        print(' ', g['검증'], '|', g['값'])


def main():
    args = parse()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'walk':
        import traceback
        code = 0
        try:
            walk(args)
        except BaseException:  # noqa: BLE001  — os._exit 가 traceback 을 삼키지 않게 먼저 찍는다
            traceback.print_exc()
            code = 3
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)          # simulation_app.close() 가 오래 멈춘다
    elif args.stage == 'compare':
        compare(args)
    else:
        assert False, f'unreachable stage={args.stage!r}'


if __name__ == '__main__':
    main()
