"""noeun 82 · DAgger 롤아웃 — S2 모델이 직접 걷게 하고, 지나간 프레임마다 전문가 정답을 붙여 vln_ce 형식으로 저장한다 (Isaac).

한 step:
  1. 지금 자세(navmesh 위 base + RBQ trunk)에서 51_render_rbq 와 같은 렌더러로 D455 이미지 · 깊이를 만든다.
  2. VLN-PE 평가와 같은 agent server(`scripts/eval/start_server.py`, internvla_n1_s2) 에 넣어 행동을 받는다 (입력 형식이 평가와 같다).
  3. 전문가(81_dagger_navmesh_server /expert)가 여기서부터 정답 경로로 복귀해 걷는 궤적을 렌더 없이 계산하고,
     그 궤적으로 학습 데이터와 같은 함수(vlnce_pixel_goal.select_goal)로 pixel goal 을 정한다.
  4. 확률 beta 로 전문가 행동을, 아니면 모델 행동을 실행한다 (운동학 이동: 전진 0.25 m navmesh try_step · 회전 15°).
끝: 실행한 STOP · 정답 경로에서 max_dev_m 넘게 벗어남 · max_steps.
저장 (학습 로더용 열 + 전문가 열): action(실행) · pose.<rig>(시작 기준 상대) · goal.<rig> · relative_goal_frame_id.<rig> (전문가) ·
  dagger.expert_actions · dagger.expert_stop · dagger.exec_by · dagger.dev_m · dagger.world_pose ...
  기존 로더(NavPixelGoalDataset)는 action 열로 턴 · STOP 정답을 만들어 틀린다 → dagger.* 열을 읽는 로더로 학습한다 (H200).
'↓' 다음 제자리(look-down) 관측은 agent 가 히스토리에 넣지 않으므로 저장하지 않는다 (저장 프레임 = agent 히스토리).

사용 (isaac 컨테이너, 저장소 루트; agent server · 81 서버가 떠 있어야 한다):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/82_dagger_rollout.py --raw_json <train.json.gz> --ds hunav_v1 --model_path <ckpt> --out_root <dir>
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
import requests

REPO = Path(__file__).resolve().parents[4]
CONV = REPO / 'scripts' / 'dataset_converters'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CONV / 'nav3d' / 'utils'))
for p in (CONV / 'gs_vlnpe', CONV / 'gs_vlnpe' / 'apply_real'):
    sys.path.insert(0, str(p))
from render_poses import frames_to_poses  # noqa: E402
from vlnce_discretize import to_start_relative  # noqa: E402
import vlnce_pixel_goal as vpg  # noqa: E402

STOP, FORWARD, LEFT, RIGHT, STAND = 0, 1, 2, 3, -1


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_json', required=True, help='R2R 형식 (start · reference_path, habitat 좌표). instruction 은 --labels_json 이 없을 때 쓴다')
    ap.add_argument('--labels_json', default='', help='62_label 결과 (raw_episode_id · sib_idx · instruction) — 형제 문장')
    ap.add_argument('--ds', required=True, help='경로 종류 이름 (v6 | hunav_v1), 저장 scene 폴더 이름에 들어간다')
    ap.add_argument('--sib', default='cycle', help="문장 형제 번호 0|1|2, cycle = 에피소드 번호 % 3")
    ap.add_argument('--episodes', default='', help='raw episode_id 목록 "0,5,7" 또는 범위 "0:50" (빈 값 = 전부)')
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--agent_cfg', default='scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py')
    ap.add_argument('--agent_port', type=int, default=8032)
    ap.add_argument('--expert_url', default='http://127.0.0.1:8041')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--agent_log_dir', default='', help='agent server 로그 (빈 값 = <out_root>/agent). 같은 out_root 에 작업자 여럿이면 따로 준다')
    ap.add_argument('--usd', default='data/GS_USDZ/noeun_station_collision.usdz')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--base_height', type=float, default=0.55, help='RBQ trunk z - 바닥 (51_render_rbq 와 같음)')
    ap.add_argument('--pitch_gain', type=float, default=-0.75, help='trunk pitch = gain x 경사각 (학습 RBQ 자세 실측: 경사 10° -> -5°, 25° -> -19°)')
    ap.add_argument('--beta', type=float, default=0.5, help='전문가 행동을 실행할 확률 (모델 행동 STOP/전진/회전 step 에만)')
    ap.add_argument('--max_dev_m', type=float, default=3.0, help='정답 경로에서 이만큼 벗어나면 끝')
    ap.add_argument('--max_steps_factor', type=float, default=2.0, help='최대 step = 전문가 전체 step x 이 값 + 20')
    ap.add_argument('--expert_horizon', type=int, default=60, help='라벨용 전문가 궤적 길이 (pixel goal 은 전진 13 step 안)')
    ap.add_argument('--stop_radius_m', type=float, default=0.35, help='전문가 STOP: ref 끝에서 이 거리 안')
    ap.add_argument('--lookahead_m', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def post(url, body):
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def quat_wxyz(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def trunk(pos, yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return {'pos': list(map(float, pos)), 'quat_wxyz': quat_wxyz(R)}


def pitch_from(prev, cur, gain):
    """직전 전진 두 점의 경사로 trunk pitch (평지 0)."""
    dxy = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
    return 0.0 if dxy < 1e-3 else gain * math.atan2(cur[2] - prev[2], dxy)


class Pitch:
    """전진할 때만 갱신 (최근 전진 2 번의 시작-끝 경사)."""

    def __init__(self, gain, pos):
        self.gain, self.hist, self.value = gain, [list(pos)], 0.0

    def moved(self, pos):
        self.hist = (self.hist + [list(pos)])[-3:]
        self.value = pitch_from(self.hist[0], self.hist[-1], self.gain)


def expert_label(a, E, depth_m):
    """전문가 궤적 E (지금 상태 포함) -> (goal_len, u, v, expert_actions, stop). 학습 데이터와 같은 select_goal."""
    acts = [e['action'] for e in E]
    if acts[0] == STOP:
        return -1, -1, -1, acts[:16], True
    pit, frames = Pitch(a.pitch_gain, E[0]['pos']), []
    for k, e in enumerate(E):
        if k > 0 and acts[k - 1] == FORWARD:
            pit.moved(e['pos'])
        frames.append(trunk(e['pos'], e['yaw'], pit.value))
    P = frames_to_poses(frames)
    act_in = np.array([-1] + acts[:-1], np.int32)  # action[k] = 프레임 k 로 들어온 동작 (vlnce 규약)
    L, u, v = vpg.select_goal(P, act_in, 0, a.rig, depth_m)
    return L, u, v, acts[:16], False


def main():
    a = parse()
    rng = random.Random(a.seed)
    os.environ.setdefault('NOEUN_MODEL', a.model_path)
    os.environ['NOEUN_PORT'] = str(a.agent_port)
    os.environ['NOEUN_FOLLOWER'] = 'navmesh'
    os.environ['NOEUN_BASE_HEIGHT'] = str(a.base_height)
    from scripts.eval.start_server import load_eval_cfg
    from internnav.utils.comm_utils.client import AgentClient

    cfg = load_eval_cfg(str(REPO / a.agent_cfg)).agent
    out = Path(a.out_root)
    cfg.model_settings['model_path'] = a.model_path
    agent_dir = Path(a.agent_log_dir) if a.agent_log_dir else out / 'agent'
    cfg.model_settings['log_dir'] = str(agent_dir)
    client = AgentClient(cfg)

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

    import pyarrow as pa
    import pyarrow.parquet as pq
    import cv2
    from geometry_utils import save_jpg

    for e in eps:
        rid = e['episode_id']
        k = rid % 3 if a.sib == 'cycle' else int(a.sib)
        instr = sibs[rid][k] if sibs else e['instruction']['instruction_text']
        sc = out / 'traj_data' / 'r2r' / f'dagger_{a.ds}_s{k}'
        (sc / 'meta').mkdir(parents=True, exist_ok=True)
        done = sc / 'meta' / f'done_{rid:06d}.json'
        if done.exists():
            continue
        vid = sc / 'videos' / 'chunk-000'
        rgb_dir, dep_dir = vid / f'observation.images.rgb.{a.rig}', vid / f'observation.images.depth.{a.rig}'
        for d in (rgb_dir, dep_dir, sc / 'data' / 'chunk-000'):
            d.mkdir(parents=True, exist_ok=True)

        ref_h = json.loads(e['reference_path']) if isinstance(e['reference_path'], str) else e['reference_path']
        ref = [[p[0], -p[2], p[1]] for p in ref_h]  # habitat -> world (바닥)
        q = e['start_rotation']
        yaw = 2 * math.atan2(q[1], q[3]) + math.pi / 2
        pos = np.array([ref[0][0], ref[0][1], ref[0][2] + a.base_height])
        ex_req = {'base_height': a.base_height, 'ref': ref, 'stop_radius_m': a.stop_radius_m, 'lookahead_m': a.lookahead_m}
        n_full = len(post(a.expert_url + '/expert', {**ex_req, 'pos': pos.tolist(), 'yaw': yaw, 'max_steps': 2000})['steps'])
        max_steps = int(a.max_steps_factor * n_full) + 20

        client.reset()
        pit, hint, skip_next, rows, t0 = Pitch(a.pitch_gain, pos), 0, False, [], time.time()
        end, n_expert = 'max_steps', 0
        for step in range(max_steps):
            fr = trunk(pos, yaw, pit.value)
            c2w = frames_to_poses([fr])[0]
            rgb, depth = r4.render_along(cas, c2w[None])
            rgb, depth_m = rgb[0], depth[0]
            valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < prof.render_far_m * 0.99)
            R_usd = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenCV -> USD 카메라 (-Z 전방 · +Y 위)
            obs = {'rgb': rgb, 'depth': np.clip(np.where(valid, depth_m, 10.0) / 10.0, 0, 1)[..., None].astype(np.float32),
                   'globalgps': pos.copy(), 'globalrotation': np.array(fr['quat_wxyz']), 'camera_position': c2w[:3, 3].copy(),
                   'camera_orientation': np.array(quat_wxyz(R_usd)), 'instruction': instr}
            a_model = int(client.step([obs])[0]['action'][0])

            E = post(a.expert_url + '/expert', {**ex_req, 'pos': pos.tolist(), 'yaw': yaw, 'hint': hint, 'max_steps': a.expert_horizon})
            hint = E['hint']
            E = E['steps']
            dev = E[0]['dev']
            store = not skip_next
            if store:
                fi = len(rows)
                L, u, v, eacts, estop = expert_label(a, E, np.where(valid, depth_m, 0.0))
                save_jpg(rgb, rgb_dir / f'episode_{rid:06d}_{fi}.jpg')
                cv2.imwrite(str(dep_dir / f'episode_{rid:06d}_{fi}.png'), ex.depth_m_to_vlnce_png(depth_m, valid))
                rows.append({'c2w': c2w, 'world': [*pos.tolist(), yaw, pit.value], 'goal_len': L, 'goal': [u, v],
                             'eacts': eacts, 'estop': estop, 'model': a_model, 'dev': dev, 'ref_i': E[0]['ref_i']})

            exec_by = 0
            a_exec = a_model
            if a_model in (STOP, FORWARD, LEFT, RIGHT) and rng.random() < a.beta:
                a_exec, exec_by = E[0]['action'], 1
                n_expert += 1
            if store:
                rows[-1]['exec'], rows[-1]['exec_by'] = a_exec, exec_by
            # '↓' 다음 제자리 관측은 agent 가 look-down 으로만 쓰고 히스토리에 넣지 않는다. WAIT 의 제자리는 넣는다.
            skip_next = a_exec == STAND and not _last_event_is_wait(agent_dir)
            if a_exec == STOP:
                end = 'expert_stop' if exec_by else ('model_stop_ok' if E[0]['action'] == STOP else 'model_stop_early')
                break
            if a_exec == FORWARD:
                pos = np.array(post(a.expert_url + '/step', {'pos': pos.tolist(), 'yaw': yaw, 'base_height': a.base_height})['pos'])
                pit.moved(pos)
            elif a_exec == LEFT:
                yaw += math.radians(15)
            elif a_exec == RIGHT:
                yaw -= math.radians(15)
            elif a_exec == STAND:
                pass
            else:
                assert False, f'unreachable action={a_exec!r}'
            if dev > a.max_dev_m:
                end = 'deviated'
                break

        floor0 = ref[0][2]  # 51_render_rbq 와 같음 (시작 바닥)
        rel = to_start_relative(np.stack([r['c2w'] for r in rows]), floor_z=floor0).astype(np.float32)
        act_in = [-1] + [r['exec'] for r in rows[:-1]]
        cols = {
            'action': act_in,
            f'pose.{a.rig}': [p.tolist() for p in rel],
            f'goal.{a.rig}': [r['goal'] for r in rows],
            f'relative_goal_frame_id.{a.rig}': [r['goal_len'] for r in rows],
            'dagger.expert_actions': [r['eacts'] for r in rows],
            'dagger.expert_stop': [r['estop'] for r in rows],
            'dagger.model_action': [r['model'] for r in rows],
            'dagger.exec_action': [r['exec'] for r in rows],
            'dagger.exec_by': [r['exec_by'] for r in rows],
            'dagger.dev_m': [round(r['dev'], 3) for r in rows],
            'dagger.ref_i': [r['ref_i'] for r in rows],
            'dagger.world_pose': [r['world'] for r in rows],
            'frame_index': list(range(len(rows))),
            'episode_index': [rid] * len(rows),
        }
        pq.write_table(pa.table(cols), sc / 'data' / 'chunk-000' / f'episode_{rid:06d}.parquet')
        meta = {'episode_index': rid, 'tasks': [instr], 'length': len(rows), 'raw_episode_id': rid, 'ds': a.ds, 'sib': k,
                'end_reason': end, 'n_expert_exec': n_expert, 'steps': step + 1, 'expert_full_steps': n_full, 'beta': a.beta,
                'model_path': a.model_path, 'seed': a.seed, 'max_dev_m': max(r['dev'] for r in rows), 'sec': round(time.time() - t0, 1)}
        done.write_text(json.dumps(meta, ensure_ascii=False))
        print(f'[82] {a.ds} ep {rid} s{k}: {len(rows)} 프레임 · 끝 {end} · 전문가 실행 {n_expert} · 최대 이탈 {meta["max_dev_m"]:.2f} m · {meta["sec"]:.0f}s', flush=True)

    for sc in sorted((out / 'traj_data' / 'r2r').glob(f'dagger_{a.ds}_s*')):
        metas = sorted((json.loads(p.read_text()) for p in (sc / 'meta').glob('done_*.json')), key=lambda m: m['episode_index'])
        with open(sc / 'meta' / 'episodes.jsonl', 'w') as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print('[82] 끝', flush=True)


def _last_event_is_wait(agent_dir):
    """agent 로그(steps.jsonl) 의 마지막 기록이 WAIT 인지 — STAND 가 '↓' 때문인지 WAIT 때문인지 가린다."""
    eps = sorted(Path(agent_dir).glob('ep*'))
    if not eps:
        return False
    with open(eps[-1] / 'steps.jsonl') as f:
        last = f.readlines()[-1]
    return json.loads(last).get('event') == 'wait'


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
