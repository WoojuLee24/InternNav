"""noeun 84 · 목표 근처 데이터 (A 비스듬히 다가오기 · B 지나쳤다 돌아오기) — 모델 없이 전문가가 일부러 흔들어 걷고, 프레임마다 전문가 정답을 붙인다 (Isaac).

왜: 반경 STOP(r1.0) 은 정답 자세에서 STOP 을 잘 하지만 (열린 루프 26/37), 닫힌 루프에서는 비켜서 · 다른 방향으로 다가가거나
    지나친 장면에서 STOP 을 못 한다 (VLN-PE 지나침 8/40). 그런 장면을 실제로 걸은 궤적(히스토리가 이어짐)으로 만든다.

에피소드마다:
  1. prefix: 전문가(81 /expert)가 시작부터 목표까지 남은 경로 길이 branch_m 인 점까지 걷는다 — 학습 로더의 history
     (linspace(0, i-1, 8)) 가 시작부터의 장면이 되도록. prefix 프레임은 abc.sample = False (history 용, 학습 샘플 아님).
  2. 가지 (prefix 끝 상태에서 시작, 프레임은 abc.sample = True):
     A±: ref 끝에서 a_back_m 앞 점을 옆으로 ±U(a_off) m 옮긴 점까지 간 뒤 목표로 (도착 방향 ±60° 안에서 비스듬히).
     B : 목표까지 간 뒤 보던 방향으로 U(b_over) m 더 걷고 돌아와 목표로.
     걷기 = navmesh 최단 경로 다음 점 추종 (/next, 회전 15° · 전진 0.25 m) + 확률 noise_p 로 전진 대신 아무 회전.
  3. 라벨: 82 와 같은 전문가 /expert + select_goal (STOP = ref 끝 stop_radius_m 안). 반경 STOP 은 학습 로더가
     abc.goal_dist_m · abc.goal_dz_m 으로 다시 정한다 (r · 경계 띠를 학습 config 로).
저장: 가지마다 한 에피소드 (prefix + 가지), scene abc_<ds>_s<형제>, episode_index = raw id x 10 + 가지 번호.
  열은 82 와 같음 (dagger.* 는 전문가 라벨 · 실행 행동) + abc.sample · abc.kind · abc.goal_dist_m · abc.goal_dz_m
  · abc.yaw_to_goal_deg (카메라 방향 - 목표 방향) · abc.yaw_to_arrive_deg (카메라 방향 - 정답 도착 방향) — 로더가 '목표가 안 보이는 STOP' 을 거를 수 있게.
  prefix 는 prefix_every 간격만 저장 (history 용, action 열은 간격 사이 행동이 빠짐).
  prefix 이미지는 가지끼리 하드링크로 나눈다.

사용 (isaac 컨테이너, 저장소 루트; 81 서버 :8041 이 떠 있어야 한다, 모델 서버는 필요 없다):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/84_goal_region.py --raw_json <train.json.gz> --labels_json <62 결과> --ds v6 --out_root <dir> --episodes 0,1,2
"""

import argparse
import gzip
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CONV = REPO / 'scripts' / 'dataset_converters'
HERE = Path(__file__).resolve().parent


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


D82 = _load('dagger82', HERE / '82_dagger_rollout.py')  # trunk · Pitch · expert_label · post 를 그대로 쓴다 (sys.path 도 거기서 잡힌다)
from render_poses import frames_to_poses  # noqa: E402
from vlnce_discretize import to_start_relative  # noqa: E402

STOP, FORWARD, LEFT, RIGHT = D82.STOP, D82.FORWARD, D82.LEFT, D82.RIGHT


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_json', required=True)
    ap.add_argument('--labels_json', default='')
    ap.add_argument('--ds', required=True)
    ap.add_argument('--sib', default='cycle')
    ap.add_argument('--episodes', default='')
    ap.add_argument('--expert_url', default='http://127.0.0.1:8041')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--usd', default='data/GS_USDZ/noeun_station_collision.usdz')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--base_height', type=float, default=0.55)
    ap.add_argument('--pitch_gain', type=float, default=-0.75)
    ap.add_argument('--expert_horizon', type=int, default=60)
    ap.add_argument('--stop_radius_m', type=float, default=0.35, help='전문가 STOP 라벨 (82 와 같음). 반경 STOP 은 로더가 abc.goal_dist_m 로')
    ap.add_argument('--lookahead_m', type=float, default=2.0)
    ap.add_argument('--branch_m', type=float, default=5.0, help='가지 시작 = ref 끝까지 남은 경로 길이')
    ap.add_argument('--branches', default='A+,A-,B', help='만들 가지 (A+ · A- · B)')
    ap.add_argument('--a_back_m', type=float, default=2.5, help='A: 옆으로 옮길 점 = ref 끝에서 이 경로 길이 앞')
    ap.add_argument('--a_off', default='0.6,1.2', help='A: 옆으로 옮기는 거리 범위 (m)')
    ap.add_argument('--b_over', default='0.5,3.0', help='B: 목표 뒤로 더 걷는 거리 범위 (m)')
    ap.add_argument('--noise_p', type=float, default=0.1, help='전진 대신 아무 회전을 할 확률')
    ap.add_argument('--arrive_m', type=float, default=0.3, help='가지 끝 = 목표 xy 이 거리 안')
    ap.add_argument('--max_branch_steps', type=int, default=120)
    ap.add_argument('--prefix_every', type=int, default=4, help='prefix 를 이 간격으로만 렌더 · 저장 (history 는 시작~지금 8 장 고르게라 간격은 무관, 렌더 시간 1/k)')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def ref_remaining(ref):
    """ref 각 점에서 끝까지 남은 경로 길이."""
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


class Walker:
    """운동학 걷기 (82 와 같은 이동: 전진 = 81 /step, 회전 15°). 상태마다 (pos, yaw, pitch, 들어온 행동) 를 쌓는다."""

    def __init__(self, a, pos, yaw, pitch_obj):
        self.a, self.pos, self.yaw, self.pit = a, np.asarray(pos, float), float(yaw), pitch_obj
        self.states = []

    def record(self, kind):
        self.states.append({'pos': self.pos.copy(), 'yaw': self.yaw, 'pitch': self.pit.value, 'kind': kind})

    def act(self, action):
        self.states[-1]['exec'] = action
        if action == FORWARD:
            new = np.array(D82.post(self.a.expert_url + '/step', {'pos': self.pos.tolist(), 'yaw': self.yaw, 'base_height': self.a.base_height})['pos'])
            moved = np.linalg.norm(new[:2] - self.pos[:2]) > 0.02
            self.pos = new
            self.pit.moved(self.pos)
            return moved
        elif action == LEFT:
            self.yaw += math.radians(15)
        elif action == RIGHT:
            self.yaw -= math.radians(15)
        else:
            assert False, f'unreachable action={action!r}'
        return True

    def go_to(self, goal_floor, kind, rng, radius, budget):
        """navmesh 다음 점 추종으로 goal(바닥 점) xy radius 안까지. 잡음: 전진 대신 아무 회전. -> 도착 여부."""
        g = (np.asarray(goal_floor, float) + [0, 0, self.a.base_height]).tolist()
        for _ in range(budget):
            if math.hypot(self.pos[0] - g[0], self.pos[1] - g[1]) < radius:
                return True
            nx = D82.post(self.a.expert_url + '/next', {'pos': self.pos.tolist(), 'goal': g, 'base_height': self.a.base_height, 'yaw': self.yaw})
            act = nx['action']
            if act == STOP:  # planner 반경(0.25) 안이거나 길 없음
                return nx.get('found', False) or nx.get('dist_xy') is not None and nx['dist_xy'] < radius
            if act == FORWARD and rng.random() < self.a.noise_p:
                act = rng.choice([LEFT, RIGHT])
            self.record(kind)
            self.act(act)
        return False


def side_point(ref, rem, back_m, off_m):
    """ref 끝에서 경로 길이 back_m 앞 점을 진행 방향 왼쪽(+)/오른쪽(-) 으로 off_m 옮긴 바닥 점."""
    i = int(np.searchsorted(-rem, -back_m))  # rem 은 내림차순
    i = min(max(i, 1), len(ref) - 1)
    d = ref[i] - ref[i - 1]
    n = np.array([-d[1], d[0], 0.0])
    n /= np.linalg.norm(n) + 1e-9
    return ref[i] + n * off_m


def main():
    a = parse()
    rng = random.Random(a.seed)
    out = Path(a.out_root)
    a_off = [float(x) for x in a.a_off.split(',')]
    b_over = [float(x) for x in a.b_over.split(',')]

    ex = _load('ex05', CONV / 'gs_vlnpe' / 'apply_real' / '05_export_vlnce.py')
    r4 = ex.load_renderer_module()
    prof = ex.load_camera_profiles().get(a.rig)
    cas = r4.build_renderer(prof.width, prof.height, prof.k, REPO / a.usd, 'ambient_only', None, 10.0, 70.0, prof.render_near_m, prof.render_far_m)

    eps = json.load(gzip.open(a.raw_json))['episodes']
    sibs = {}
    if a.labels_json:
        for e in json.load(gzip.open(a.labels_json))['episodes']:
            sibs.setdefault(e['raw_episode_id'], {})[e['sib_idx']] = e['instruction']['instruction_text']
    if a.episodes and ':' in a.episodes:
        lo, hi = a.episodes.split(':')
        keep = set(range(int(lo or 0), int(hi or len(eps))))
    elif a.episodes:
        keep = {int(x) for x in a.episodes.split(',')}
    else:
        keep = None
    eps = [e for e in eps if keep is None or e['episode_id'] in keep]

    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    from geometry_utils import save_jpg

    def render(states):
        """상태들 -> (c2w, rgb, depth_m, valid) 목록 (한 번에 렌더)."""
        frames = [D82.trunk(s['pos'], s['yaw'], s['pitch']) for s in states]
        P = frames_to_poses(frames)
        rgb, depth = r4.render_along(cas, P)
        res = []
        for k in range(len(states)):
            dm = depth[k]
            valid = np.isfinite(dm) & (dm > 0) & (dm < prof.render_far_m * 0.99)
            res.append((P[k], rgb[k], dm, valid))
        return res

    for e in eps:
        rid = e['episode_id']
        k_sib = rid % 3 if a.sib == 'cycle' else int(a.sib)
        instr = sibs[rid][k_sib] if sibs else e['instruction']['instruction_text']
        sc = out / 'traj_data' / 'r2r' / f'abc_{a.ds}_s{k_sib}'
        (sc / 'meta').mkdir(parents=True, exist_ok=True)
        done = sc / 'meta' / f'done_{rid:06d}.json'
        if done.exists():
            continue
        vid = sc / 'videos' / 'chunk-000'
        rgb_dir, dep_dir = vid / f'observation.images.rgb.{a.rig}', vid / f'observation.images.depth.{a.rig}'
        for d in (rgb_dir, dep_dir, sc / 'data' / 'chunk-000'):
            d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        ref_h = json.loads(e['reference_path']) if isinstance(e['reference_path'], str) else e['reference_path']
        ref = np.array([[p[0], -p[2], p[1]] for p in ref_h])
        rem = ref_remaining(ref)
        goal = ref[-1]
        q = e['start_rotation']
        yaw0 = 2 * math.atan2(q[1], q[3]) + math.pi / 2
        pos0 = np.array([ref[0][0], ref[0][1], ref[0][2] + a.base_height])
        ex_req = {'base_height': a.base_height, 'ref': ref.tolist(), 'stop_radius_m': a.stop_radius_m, 'lookahead_m': a.lookahead_m}

        # 1. prefix: 전문가 전체 궤적에서 남은 경로 branch_m 까지
        E_full = D82.post(a.expert_url + '/expert', {**ex_req, 'pos': pos0.tolist(), 'yaw': yaw0, 'max_steps': 2000})['steps']
        nb = next((j for j, s in enumerate(E_full) if rem[s['ref_i']] <= a.branch_m), len(E_full) - 1)
        nb = max(nb, 1)
        pit = D82.Pitch(a.pitch_gain, pos0)
        pre_states = []
        keep_j = set(range(0, nb, a.prefix_every))
        for j in range(nb):
            if j > 0 and E_full[j - 1]['action'] == FORWARD:
                pit.moved(E_full[j]['pos'])
            if j in keep_j:
                    pre_states.append({'pos': np.asarray(E_full[j]['pos']), 'yaw': E_full[j]['yaw'], 'pitch': pit.value, 'kind': 'prefix', 'exec': E_full[j]['action'], 'j': j})
        if E_full[nb - 1]['action'] == FORWARD:
            pit.moved(E_full[nb]['pos'])
        branch_start = (np.asarray(E_full[nb]['pos']), E_full[nb]['yaw'])
        hint_b = E_full[nb]['ref_i']
        i_arr = int(np.searchsorted(-rem, -1.5))  # 도착 방향 = ref 끝 1.5 m 구간
        i_arr = min(i_arr, len(ref) - 2)
        arrive_yaw = math.atan2(ref[-1][1] - ref[i_arr][1], ref[-1][0] - ref[i_arr][0])

        pre_r = render(pre_states)
        pre_rows = []
        for jj, (c2w, rgb, dm, valid) in enumerate(pre_r):
            j = pre_states[jj]['j']
            L, u, v, eacts, estop = D82.expert_label(a, E_full[j:j + a.expert_horizon], np.where(valid, dm, 0.0))
            pre_rows.append({'c2w': c2w, 'rgb': rgb, 'dm': dm, 'valid': valid, 'goal_len': L, 'goal': [u, v], 'eacts': eacts, 'estop': estop,
                             'dev': E_full[j]['dev'], 'ref_i': E_full[j]['ref_i'], 'state': pre_states[jj]})

        # 2. 가지
        metas = []
        for bi, br in enumerate(a.branches.split(',')):
            import copy
            W = Walker(a, branch_start[0], branch_start[1], copy.deepcopy(pit))
            note = ''
            if br in ('A+', 'A-'):
                s = 1.0 if br == 'A+' else -1.0
                off = rng.uniform(*a_off)
                O = side_point(ref, rem, a.a_back_m, s * off)
                ok1 = W.go_to(O, br, rng, 0.4, a.max_branch_steps // 2)
                ok2 = W.go_to(goal, br, rng, a.arrive_m, a.max_branch_steps)
                note = f'옆 {s * off:+.2f} m · 옆점 도착 {ok1} · 목표 도착 {ok2}'
            elif br == 'B':
                ok1 = W.go_to(goal, br, rng, a.arrive_m, a.max_branch_steps // 2)
                over = rng.uniform(*b_over)
                n_over = 0
                for _ in range(int(round(over / 0.25))):
                    W.record(br)
                    if not W.act(FORWARD):
                        break
                    n_over += 1
                ok2 = W.go_to(goal, br, rng, a.arrive_m, a.max_branch_steps)
                note = f'목표 도착 {ok1} · 지나감 {n_over * 0.25:.2f}/{over:.2f} m · 돌아옴 {ok2}'
            else:
                assert False, f'unreachable branch={br!r}'
            W.record(br)  # 마지막 상태 (도착 자리)
            W.states[-1]['exec'] = STOP

            br_r = render(W.states)
            rows = list(pre_rows)
            for s_, (c2w, rgb, dm, valid) in zip(W.states, br_r):
                E = D82.post(a.expert_url + '/expert', {**ex_req, 'pos': s_['pos'].tolist(), 'yaw': s_['yaw'], 'hint': hint_b, 'max_steps': a.expert_horizon})['steps']
                L, u, v, eacts, estop = D82.expert_label(a, E, np.where(valid, dm, 0.0))
                rows.append({'c2w': c2w, 'rgb': rgb, 'dm': dm, 'valid': valid, 'goal_len': L, 'goal': [u, v], 'eacts': eacts, 'estop': estop,
                             'dev': E[0]['dev'], 'ref_i': E[0]['ref_i'], 'state': s_})

            eid = rid * 10 + bi
            for fi, r in enumerate(rows):
                jp, pp = rgb_dir / f'episode_{eid:06d}_{fi}.jpg', dep_dir / f'episode_{eid:06d}_{fi}.png'
                if fi < len(pre_rows) and bi > 0:  # prefix 는 첫 가지 파일을 하드링크
                    src = rid * 10
                    for p_new, p_src in ((jp, rgb_dir / f'episode_{src:06d}_{fi}.jpg'), (pp, dep_dir / f'episode_{src:06d}_{fi}.png')):
                        if p_new.exists():
                            p_new.unlink()
                        os.link(p_src, p_new)
                else:
                    save_jpg(r['rgb'], jp)
                    cv2.imwrite(str(pp), ex.depth_m_to_vlnce_png(r['dm'], r['valid']))

            def gdist(s_):
                p = s_['pos'] - [0, 0, a.base_height]
                return round(float(math.hypot(p[0] - goal[0], p[1] - goal[1])), 3), round(float(p[2] - goal[2]), 3)

            def ang(x):
                return round(math.degrees((x + math.pi) % (2 * math.pi) - math.pi), 1)

            def view(s_):
                """(yaw - 목표 방향, yaw - 도착 방향) 도. 목표 방향은 0.3 m 안이면 0 (정의 안 됨)."""
                p = s_['pos']
                to_g = 0.0 if math.hypot(goal[0] - p[0], goal[1] - p[1]) < 0.3 else ang(s_['yaw'] - math.atan2(goal[1] - p[1], goal[0] - p[0]))
                return to_g, ang(s_['yaw'] - arrive_yaw)

            rel = to_start_relative(np.stack([r['c2w'] for r in rows]), floor_z=ref[0][2]).astype(np.float32)
            execs = [r['state']['exec'] for r in rows]
            gd = [gdist(r['state']) for r in rows]
            vw = [view(r['state']) for r in rows]
            cols = {
                'action': [-1] + execs[:-1],
                f'pose.{a.rig}': [p.tolist() for p in rel],
                f'goal.{a.rig}': [r['goal'] for r in rows],
                f'relative_goal_frame_id.{a.rig}': [r['goal_len'] for r in rows],
                'dagger.expert_actions': [r['eacts'] for r in rows],
                'dagger.expert_stop': [r['estop'] for r in rows],
                'dagger.exec_action': execs,
                'dagger.exec_by': [1] * len(rows),
                'dagger.dev_m': [round(r['dev'], 3) for r in rows],
                'dagger.ref_i': [r['ref_i'] for r in rows],
                'dagger.world_pose': [[*r['state']['pos'].tolist(), r['state']['yaw'], r['state']['pitch']] for r in rows],
                'abc.sample': [r['state']['kind'] != 'prefix' for r in rows],
                'abc.kind': [r['state']['kind'] for r in rows],
                'abc.goal_dist_m': [g[0] for g in gd],
                'abc.goal_dz_m': [g[1] for g in gd],
                'abc.yaw_to_goal_deg': [x[0] for x in vw],
                'abc.yaw_to_arrive_deg': [x[1] for x in vw],
                'frame_index': list(range(len(rows))),
                'episode_index': [eid] * len(rows),
            }
            pq.write_table(pa.table(cols), sc / 'data' / 'chunk-000' / f'episode_{eid:06d}.parquet')
            n_b = len(rows) - len(pre_rows)
            metas.append({'episode_index': eid, 'tasks': [instr], 'length': len(rows), 'raw_episode_id': rid, 'ds': a.ds, 'sib': k_sib,
                          'branch': br, 'n_prefix': len(pre_rows), 'n_branch': n_b, 'note': note,
                          'min_goal_dist_m': min(g[0] for g in gd[len(pre_rows):]), 'end_goal_dist_m': gd[-1][0], 'seed': a.seed})
            print(f'[84] {a.ds} ep {rid} {br}: prefix {len(pre_rows)} + 가지 {n_b} · 끝 {gd[-1][0]:.2f} m · {note}', flush=True)
        done.write_text(json.dumps({'raw_episode_id': rid, 'branches': metas, 'sec': round(time.time() - t0, 1)}, ensure_ascii=False))

    for sc in sorted((out / 'traj_data' / 'r2r').glob(f'abc_{a.ds}_s*')):
        ms = []
        for p in sorted((sc / 'meta').glob('done_*.json')):
            ms += json.loads(p.read_text())['branches']
        with open(sc / 'meta' / 'episodes.jsonl', 'w') as f:
            for m in sorted(ms, key=lambda m: m['episode_index']):
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print('[84] 끝', flush=True)


if __name__ == '__main__':
    import traceback

    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
