"""06k — RBQ10 사족보행 로봇이 **물리 시뮬레이션으로** nav3d 경로를 걷는다 (H1 대신; H1 RL 정책은 계단을 못 오른다 — 06r).

sim_docker(`modules/sim_docker/workspace/scripts/run_rbq10_sim.py`) 의 보행 루프를 그대로 따른다:
  200 Hz physics (dt 0.005) · 50 Hz 정책 (decimation 4) · 명시 PD 토크 (DCMotor clamp) · action delay 2 step ·
  정책 = ONNX + `camel.policy.v1` 메타데이터 (`policy_runtime.py` 그대로 import) · height scan = trunk +0.5 m 에서 PhysX 하향 raycast.
ROS·센서는 쓰지 않는다. 자산은 sim_docker 것을 `/ws/assets/rbq10/` 에 하드링크해 쓴다 (원본 수정 없음).

명령: vln_pe 와 같은 이산 규칙 (경유점 방향과 7.5° 넘게 어긋나면 15° 회전, 아니면 0.25 m 전진, 0.15 m 안이면 다음 경유점)
을 **속도 명령**으로: 전진 = vx `--fwd_speed` 로 0.25 m 갈 시간, 회전 = wz `--turn_speed` 로 15° 돌 시간.
action 이 끝날 때마다 trunk 자세를 기록 (`--record trigger` 면 걷는 동안 0.25 m 갔거나 15° 돈 순간마다) → 렌더 단계(06 렌더러, D455 = trunk 기준 sensor_configuration.yaml 장착값)에서 쓴다.

입력 `--episodes_json`: {"scene", "background_usd"?, "episodes": [{"key", "waypoints": [[x,y,z],..] (nav3d 좌표, z = 바닥), "start_yaw": rad}]}
출력 `<log_dir>/walk_<tag>.json`: episode 마다 frames [{action, pos, quat_wxyz}] · 넘어짐 · 도착 오차.

사용 (isaac 컨테이너, 저장소 루트, 씬당 프로세스 하나):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06k_rbq_walk.py --episodes_json <json> --tag <이름>
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes_json', required=True)
    ap.add_argument('--tag', required=True, help='출력 파일 이름')
    ap.add_argument('--assets', default='/ws/assets/rbq10', help='rbq10 usd · RBQ/policy · policy_runtime.py')
    ap.add_argument('--policy', default='dv_v3.6.28_full_h1_stair020_39000.onnx', help='RBQ/policy/ 아래. 기본 = sim_docker 의 계단(0.20 m) height-scan 정책')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe', help='MP3D 씬이면 load_scene_model 로 연다')
    ap.add_argument('--fwd_speed', type=float, default=0.5, help='m/s (학습 분포 ±2.0, sim_docker 클램프 ±1.5)')
    ap.add_argument('--turn_speed', type=float, default=0.75, help='rad/s (±1.5)')
    ap.add_argument('--turn_threshold_deg', type=float, default=7.5, help='경유점 방향과 이보다 어긋나면 제자리 회전 (7.5 = vln_pe 규칙)')
    ap.add_argument('--steer_gain', type=float, default=0.0,
                    help='0 = 전진 중 회전 없음 (이전). >0 = 전진 action 동안 wz = gain × 방향 오차 (rad/s, ±turn_speed 로 자름) — 좁은 통로에서 제자리 보정 회전을 줄인다')
    ap.add_argument('--steer_lookahead_m', type=float, default=0.0,
                    help='0 = 방향 보정 목표 = 지금 경유점 (이전, 0.5 m 간격이라 가깝다). >0 = 지금 경유점부터 몸에서 이 거리 이상 떨어진 첫 경유점 (pure pursuit, 계단 지그재그 줄이기)')
    ap.add_argument('--settle_s', type=float, default=0.1, help='action 사이 명령 0 으로 둘 시간 (속도 전환을 부드럽게)')
    ap.add_argument('--warmup_s', type=float, default=0.3, help='스폰 후 PD 로 서기 (sim_docker POLICY_WARMUP_S)')
    ap.add_argument('--policy_settle_s', type=float, default=0.0,
                    help='0 = 이전. >0 = warmup(PD) 뒤 보행 정책을 켠 채 명령 0 으로 이 시간 서 있다가 출발 (출발 직후 회전 때 옆 미끄러짐 줄이기)')
    ap.add_argument('--spawn_h', type=float, default=0.65, help='바닥 위 trunk 스폰 높이 (학습 INIT_Z)')
    ap.add_argument('--fall_h', type=float, default=0.25, help='trunk 바닥 위 높이가 이보다 낮으면 넘어짐')
    ap.add_argument('--fall_tilt_deg', type=float, default=60.0)
    ap.add_argument('--max_actions', type=int, default=600)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/06k_rbq_walk')
    ap.add_argument('--threads', type=int, default=0, help='0 = 제한 없음 (이전). N = onnxruntime·PhysX CPU 스레드 N 개 (병렬 worker 용)')
    ap.add_argument('--gpu_dynamics', type=int, default=1, choices=[0, 1], help='1 = sim_docker 와 같음. 0 = CPU dynamics (로봇 1 대면 더 빠를 수 있다)')
    ap.add_argument('--fast_step', action='store_true', help='sim.step 대신 PhysX simulate/fetch_results 를 직접 (Kit 앱 루프 생략)')
    ap.add_argument('--scan_obstacle_depth', type=float, default=None,
                    help='None = 그대로 (이전). 값 = height_depth(trunk − 지면) 가 이보다 작은 점(지면이 trunk 보다 높음 = 개찰구 함·기둥 위)을 평지 값으로 바꾼다. '
                         '계단은 스캔 범위(앞 0.85 m) 안에서 ≈ 0.0 까지, 함·기둥 위는 −0.3~−0.5 (실측) → −0.2 권장')
    ap.add_argument('--scan_fill', type=float, default=0.55, help='바꿀 값 = 평지에서의 height_depth (서 있는 trunk 높이)')
    ap.add_argument('--record', default='action', choices=['action', 'trigger'],
                    help='action = action 이 끝날 때마다 프레임 (이전). trigger = 걷는 동안 지난 프레임보다 --rec_dist_m 를 갔거나 '
                         '--rec_turn_deg 를 돈 순간마다 프레임 (릴리스 프레임 간격 = 한 걸음). 라벨 = 평가 vln_utils.chunk_token 규칙 '
                         '(이동/0.25 와 회전/15° 중 큰 쪽: 전진 1 · 왼 2 · 오 3)')
    ap.add_argument('--rec_dist_m', type=float, default=0.25)
    ap.add_argument('--rec_turn_deg', type=float, default=15.0)
    ap.add_argument('--shard', default='', help='"i/n": 에피소드를 n 등분한 i 번째만 (병렬 worker 용)')
    return ap.parse_args()


PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_DELAY_STEPS = 2
SCAN_RAY_UP = 0.5
DCMOTOR_EFFORT_LIMIT = 200.0
DCMOTOR_SATURATION_EFFORT = 200.0
DCMOTOR_VELOCITY_LIMIT = 50.0
FWD, TURN = 0.25, math.radians(15.0)


def main() -> int:
    args = parse()
    if args.threads:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    if args.threads:
        import carb.settings
        carb.settings.get_settings().set('/physics/numThreads', int(args.threads))
    import carb
    import omni
    import onnxruntime as ort
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils import stage as stage_utils
    from isaacsim.core.utils.types import ArticulationAction
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    sys.path.insert(0, args.assets)
    from policy_runtime import PolicyRuntime

    spec = json.loads(Path(args.episodes_json).read_text())
    eps = spec['episodes'][:args.max_episodes] if args.max_episodes else spec['episodes']
    if args.shard:
        i_, n_ = map(int, args.shard.split('/'))
        eps = eps[i_::n_]
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Scene.Define(stage, Sdf.Path('/physicsScene'))
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene'))
    phx = PhysxSchema.PhysxSceneAPI.Get(stage, '/physicsScene')
    phx.CreateEnableCCDAttr(True)
    phx.CreateEnableStabilizationAttr(False)
    phx.CreateEnableGPUDynamicsAttr(bool(args.gpu_dynamics))
    phx.CreateBroadphaseTypeAttr('GPU' if args.gpu_dynamics else 'MBP')
    phx.CreateSolverTypeAttr('TGS')
    phx.CreateBounceThresholdAttr(0.5)
    UsdGeom.Xform.Define(stage, Sdf.Path('/World'))

    bg = spec.get('background_usd')
    if not bg:                                            # MP3D: 텍스처 심링크까지 해 주는 렌더러 쪽 로더
        import importlib.util
        GS = Path(__file__).resolve().parent.parent / 'gs_vlnpe'
        sp = importlib.util.spec_from_file_location('r4_06k', GS / 'apply_real' / '04_render_obs_isaac.py')
        sys.path.insert(0, str(GS))
        sys.path.insert(0, str(GS / 'apply_real'))
        r4 = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(r4)
        bg = str(r4.load_scene_model(args.mesh_root, spec['scene']))
    stage_utils.add_reference_to_stage(usd_path=bg, prim_path='/World/Background')
    app.update()
    n_col = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Background')):   # sim_docker _apply_background_collision 와 같음
        if not prim.IsA(UsdGeom.Mesh) or 'groundplane' in str(prim.GetPath()).lower():
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr('none')
        n_col += 1
    stage_utils.add_reference_to_stage(usd_path=f'{args.assets}/usd/rbq10.usd', prim_path='/World/Robot')
    app.update()
    print(f'[06k] 배경 {bg} · 충돌 mesh {n_col} · 정책 {args.policy}', flush=True)

    sim = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=0.05, stage_units_in_meters=1.0)
    robot = SingleArticulation(prim_path='/World/Robot/trunk', name='rbq10')
    sim.reset()
    robot.initialize()
    from omni.physx import get_physx_simulation_interface
    phx_sim = get_physx_simulation_interface()
    dof = list(robot.dof_names)
    n = robot.num_dof
    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = so.inter_op_num_threads = int(args.threads)
    sess = ort.InferenceSession(f'{args.assets}/RBQ/policy/{args.policy}', sess_options=so, providers=['CPUExecutionProvider'])
    pol = PolicyRuntime(sess)
    assert abs(pol.dt - PHYSICS_DT * DECIMATION) < 1e-9, f'policy_dt {pol.dt}'
    pol_to_sim = np.array([dof.index(j) for j in pol.joint_names])
    q_default = np.zeros(n, np.float32)
    kps = np.zeros(n, np.float32)
    kds = np.zeros(n, np.float32)
    q_default[pol_to_sim], kps[pol_to_sim], kds[pol_to_sim] = pol.default_pos, pol.kp, pol.kd
    ctrl = robot.get_articulation_controller()
    ctrl.set_gains(kps=np.zeros(n, np.float32), kds=np.zeros(n, np.float32))
    scan_xy = pol.scan_points() if pol.scan is not None else None
    sqi = get_physx_scene_query_interface()
    best = [0.0]

    def hit(h):
        p = getattr(h, 'rigid_body', '') or getattr(h, 'collision', '') or ''
        if not p.startswith('/World/Robot') and h.distance < best[0]:
            best[0] = h.distance
        return True

    def ground_below(x, y, z0):
        best[0] = 1e30
        sqi.raycast_all(carb.Float3(float(x), float(y), float(z0)), carb.Float3(0.0, 0.0, -1.0), 50.0, hit)
        return best[0] if best[0] < 1e29 else None

    def height_depth(pos, R):
        c, s = R[0, 0], R[1, 0]
        nr = math.hypot(c, s)
        c, s = c / nr, s / nr
        z0 = float(pos[2]) + SCAN_RAY_UP
        out = np.full(len(scan_xy), 10.0, np.float32)
        for i, (dx, dy) in enumerate(scan_xy):
            d = ground_below(pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy, z0)
            if d is not None:
                out[i] = d - SCAN_RAY_UP
        return out

    def rot(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float32)

    prof = {'scan': 0.0, 'policy': 0.0, 'physics': 0.0, 'steps': 0}
    st = {'k': 0, 'cur': q_default.copy(), 'prev': q_default.copy(), 'sub': 0, 'active': False}

    def physics_step(cmd):
        if st['active'] and st['k'] % DECIMATION == 0:
            pos, q = robot.get_world_pose()
            R = rot(np.asarray(q, np.float32))
            src = {'angular_velocity': R.T @ np.asarray(robot.get_angular_velocity(), np.float32),
                   'gravity': R.T @ np.array([0, 0, -1.0], np.float32), 'command': np.asarray(cmd, np.float32),
                   'joint_position_rel': np.asarray(robot.get_joint_positions(), np.float32)[pol_to_sim] - pol.default_pos,
                   'joint_velocity': np.asarray(robot.get_joint_velocities(), np.float32)[pol_to_sim],
                   'payload': np.array([0.0], np.float32)}
            if scan_xy is not None:
                t_ = time.perf_counter()
                hd = height_depth(pos, R)
                if args.scan_obstacle_depth is not None:
                    hd[hd < args.scan_obstacle_depth] = args.scan_fill
                src['height_depth'] = hd
                prof['scan'] += time.perf_counter() - t_
            t_ = time.perf_counter()
            tgt = q_default.copy()
            tgt[pol_to_sim] = pol.step(src)
            prof['policy'] += time.perf_counter() - t_
            st['prev'], st['cur'], st['sub'] = st['cur'].copy(), tgt, 0
        if st['active']:
            act = st['prev'] if st['sub'] < ACTION_DELAY_STEPS else st['cur']
            st['sub'] += 1
        else:
            act = q_default
        qq = np.asarray(robot.get_joint_positions(), np.float32)
        qd = np.asarray(robot.get_joint_velocities(), np.float32)
        tau = kps * (act - qq) - kds * qd
        tmax = DCMOTOR_SATURATION_EFFORT * np.clip(1.0 - qd / DCMOTOR_VELOCITY_LIMIT, 0.0, 1.0)
        tmin = -DCMOTOR_SATURATION_EFFORT * np.clip(1.0 + qd / DCMOTOR_VELOCITY_LIMIT, 0.0, 1.0)
        tau = np.clip(np.clip(tau, tmin, tmax), -DCMOTOR_EFFORT_LIMIT, DCMOTOR_EFFORT_LIMIT)
        ctrl.apply_action(ArticulationAction(joint_efforts=tau))
        t_ = time.perf_counter()
        if args.fast_step:
            phx_sim.simulate(PHYSICS_DT, st['k'] * PHYSICS_DT)
            phx_sim.fetch_results()
        else:
            sim.step(render=False)
        prof['physics'] += time.perf_counter() - t_
        prof['steps'] += 1
        st['k'] += 1

    def reset_at(p, yaw):
        robot.set_joint_positions(q_default)
        robot.set_joint_velocities(np.zeros(n, np.float32))
        robot.set_world_pose(position=np.asarray(p, np.float32), orientation=np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], np.float32))
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))
        pol.reset()
        st.update(k=0, cur=q_default.copy(), prev=q_default.copy(), sub=0, active=False)
        for _ in range(int(args.warmup_s / PHYSICS_DT)):
            physics_step([0, 0, 0])
        st['active'] = True

    def state():
        pos, q = robot.get_world_pose()
        q = np.asarray(q, np.float64)
        R = rot(q.astype(np.float32))
        yaw = math.atan2(R[1, 0], R[0, 0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        g = ground_below(pos[0], pos[1], pos[2] + SCAN_RAY_UP)
        h = (g - SCAN_RAY_UP) if g is not None else np.nan
        return np.asarray(pos, np.float64), q, yaw, tilt, h

    def trigger_recorder():
        """--record trigger: physics_step 을 감싸 정책 주기(50 Hz)마다 trunk 를 보고, 한 걸음(거리·회전) 넘으면 프레임을 남긴다."""
        rec = {'frames': [], 'tick': 0}

        def snap(action):
            p_, q_ = robot.get_world_pose()
            rec['frames'].append({'action': action, 'pos': np.asarray(p_, np.float64).tolist(), 'quat_wxyz': np.asarray(q_, np.float64).tolist()})

        def step(cmd):
            physics_step(cmd)
            rec['tick'] += 1
            if rec['tick'] % DECIMATION:
                return
            p_, q_ = robot.get_world_pose()
            R_ = rot(np.asarray(q_, np.float32))
            last = rec['frames'][-1]
            Rl = rot(np.asarray(last['quat_wxyz'], np.float32))
            dist = float(np.linalg.norm(np.asarray(p_)[:2] - np.asarray(last['pos'])[:2]))
            dyaw = (math.atan2(R_[1, 0], R_[0, 0]) - math.atan2(Rl[1, 0], Rl[0, 0]) + math.pi) % (2 * math.pi) - math.pi
            x_prop, y_prop = dist / args.rec_dist_m, abs(dyaw) / math.radians(args.rec_turn_deg)
            if x_prop < 1.0 and y_prop < 1.0:
                return
            snap(1 if x_prop >= y_prop else (2 if dyaw > 0 else 3))

        return rec, snap, step

    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    out_f = log / f'walk_{args.tag}.json'
    results = []
    t0 = time.time()
    for e in eps:
        W = np.asarray(e['waypoints'], np.float64)
        reset_at([W[0, 0], W[0, 1], W[0, 2] + args.spawn_h], float(e['start_yaw']))
        for _ in range(int(round(args.policy_settle_s / PHYSICS_DT))):
            physics_step([0, 0, 0])
        frames, fell, why = [], False, ''
        pos, q, yaw, tilt, h = state()
        frames.append({'action': -1, 'pos': pos.tolist(), 'quat_wxyz': q.tolist()})
        rec, step = None, physics_step
        if args.record == 'trigger':
            rec, snap, step = trigger_recorder()
            snap(-1)
        wi = 1
        for _ in range(args.max_actions):
            while wi < len(W) and np.linalg.norm(W[wi, :2] - pos[:2]) < FWD * 0.6:
                wi += 1
            if wi >= len(W):
                break
            dp = W[wi, :2] - pos[:2]
            d = (math.atan2(dp[1], dp[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            a = (2 if d > 0 else 3) if abs(d) > math.radians(args.turn_threshold_deg) else 1
            if a == 1:
                cmd, dur = [args.fwd_speed, 0.0, 0.0], FWD / args.fwd_speed
            else:
                cmd, dur = [0.0, 0.0, args.turn_speed * (1 if a == 2 else -1)], TURN / args.turn_speed
            for k in range(int(round(dur / PHYSICS_DT))):
                if a == 1 and args.steer_gain > 0 and k % DECIMATION == 0:       # 걸으면서 방향 보정
                    p_, q_ = robot.get_world_pose()
                    R_ = rot(np.asarray(q_, np.float32))
                    tgt_ = wi
                    if args.steer_lookahead_m > 0:
                        while tgt_ + 1 < len(W) and np.linalg.norm(W[tgt_, :2] - np.asarray(p_)[:2]) < args.steer_lookahead_m:
                            tgt_ += 1
                    dp_ = W[tgt_, :2] - np.asarray(p_)[:2]
                    e_ = (math.atan2(dp_[1], dp_[0]) - math.atan2(R_[1, 0], R_[0, 0]) + math.pi) % (2 * math.pi) - math.pi
                    cmd = [args.fwd_speed, 0.0, float(np.clip(args.steer_gain * e_, -args.turn_speed, args.turn_speed))]
                step(cmd)
            for _ in range(int(round(args.settle_s / PHYSICS_DT))):
                step([0, 0, 0])
            pos, q, yaw, tilt, h = state()
            frames.append({'action': a, 'pos': pos.tolist(), 'quat_wxyz': q.tolist()})
            if tilt > args.fall_tilt_deg or (np.isfinite(h) and h < args.fall_h):
                fell, why = True, f'fall (tilt {tilt:.0f}° · h {h:.2f} m)'
                break
        ne = float(np.linalg.norm(W[-1, :2] - pos[:2]))
        dz = float(abs(W[-1, 2] - (pos[2] - (h if np.isfinite(h) else args.spawn_h))))
        ok = (not fell) and ne < 0.5 and wi >= len(W)
        if rec is not None:
            frames = rec['frames']
        results.append({'key': e['key'], 'success': ok, 'fell': fell, 'why': why or ('success' if ok else 'not_reached'), 'ne': ne, 'goal_dz': dz,
                        'n_actions': len(frames) - 1, 'frames': frames})
        print(f"[06k {args.tag}] {e['key']} {'성공' if ok else '실패'} NE {ne:.2f} · Δz {dz:.2f} · action {len(frames) - 1} · {why} ({time.time() - t0:.0f}s"
              f" | 누적 physics {prof['physics']:.0f}s · scan {prof['scan']:.0f}s · policy {prof['policy']:.0f}s · step {prof['steps']})", flush=True)
        out_f.write_text(json.dumps({'args': vars(args), 'scene': spec.get('scene'), 'background': bg, 'results': results}))
    print(f'[06k {args.tag}] {len(results)} ep · 성공 {sum(r["success"] for r in results)} -> {out_f}', flush=True)
    return 0


if __name__ == '__main__':
    import traceback
    code = 0
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
