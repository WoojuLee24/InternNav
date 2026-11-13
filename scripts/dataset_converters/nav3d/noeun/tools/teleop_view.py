"""teleop_view — 씬 안을 RBQ10 으로 직접 걸어 다니며 D455 RGB(3DGS) 를 보고, 위에서 본 지도에 유리문 상자를 표시한다.

브라우저 http://localhost:<port> :
  왼쪽 = 위에서 본 지도 (paint_bg_<tag>.png) + 로봇 점 + 반투명 시야 부채꼴 + 유리문 상자
  오른쪽 = 로봇 D455 카메라로 렌더한 RGB (usdz 안의 3DGS)
  W/S 전진·후진 · A/D 회전 · Q/E 옆걸음 · R 다시 세우기 · 지도 클릭(순간이동 도구) = 그 자리로 옮겨 다시 서기
  유리문 상자 = 지도에서 드래그 → 저장하면 logs/etri/<씬>/doors_<tag>.json (칠하기 페이지 doors/<tag> 와 같은 형식)

물리: walk_rbq_worker.py 의 보행 루프(200 Hz physics · 50 Hz 정책 · PD 토크 · action delay 2)를 그대로 옮겼다 — 저쪽을 바꾸면 여기도.
카메라: render_poses.trunk_to_cam_cv() (51_render_rbq 와 같은 D455 장착값).

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/tools/teleop_view.py --scene etri12_1f2f
"""

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

N = Path(__file__).resolve().parents[2]   # nav3d/
sys.path.insert(0, str(N / 'utils'))
from render_poses import trunk_to_cam_cv  # noqa: E402

PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_DELAY_STEPS = 2
SCAN_RAY_UP = 0.5
DCMOTOR_EFFORT_LIMIT = 200.0
DCMOTOR_SATURATION_EFFORT = 200.0
DCMOTOR_VELOCITY_LIMIT = 50.0


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True, help='etri12_1f2f · etri12_3fout · etri_bldg1 (logs/etri/<씬>/paint_meta_*.json 이 있어야 지도가 뜬다)')
    ap.add_argument('--usd', default=None, help='기본 /ws/assets/GS_USDZ/ETRI/<씬>_collision.usdz')
    ap.add_argument('--log_root', default=str(N / 'logs/etri'))
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--assets', default='/ws/assets/rbq10')
    ap.add_argument('--policy', default='dv_v3.6.28_full_h1_stair020_39000.onnx', help='walk_rbq_worker 와 같은 기본값')
    ap.add_argument('--scan_obstacle_depth', type=float, default=-0.2, help='ETRI 걷기(etri_regen.sh WALK)와 같은 값')
    ap.add_argument('--scan_fill', type=float, default=0.55)
    ap.add_argument('--spawn_h', type=float, default=0.65)
    ap.add_argument('--warmup_s', type=float, default=0.3)
    ap.add_argument('--fall_h', type=float, default=0.25)
    ap.add_argument('--fall_tilt_deg', type=float, default=60.0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov_deg', type=float, default=87.0, help='보기용 넓은 시야 (D455 RGB ≈ 90°). 데이터셋 vln_ce 칸은 68.7°')
    ap.add_argument('--render_every', type=int, default=10, help='physics step 몇 번마다 한 장 렌더 (10 = 0.05 s)')
    ap.add_argument('--start', type=float, nargs=3, default=None, metavar=('X', 'Y', 'YAW_DEG'), help='기본 = 경로 json 첫 경로의 출발점')
    return ap.parse_args()


def main() -> int:
    args = parse()
    L = Path(args.log_root) / args.scene
    layers = []
    for mp in sorted(L.glob('paint_meta_*.json')):
        m = json.loads(mp.read_text())
        tag = m['tag']
        if not (L / f'paint_bg_{tag}.png').exists():
            continue
        d = L / f'doors_{tag}.json'
        layers.append({k: m[k] for k in ('scene', 'tag', 'zlo', 'zhi', 'x0', 'y1', 'res', 'w', 'h', 'zmin', 'zmax')}
                      | {'doors': json.loads(d.read_text())['boxes'] if d.exists() else []})
    assert layers, f'{L}/paint_meta_*.json + paint_bg_<tag>.png 가 없다 — logs/etri/paint_bg.py 로 먼저 만든다'
    usd = args.usd or f'/ws/assets/GS_USDZ/ETRI/{args.scene}_collision.usdz'
    if args.start is None:
        P = json.loads(next(iter(sorted((L / 'paths').glob(f'{args.scene}_paths*.json')))).read_text())['episodes'][0]['trajectory']
        a, b = np.asarray(P[0]), np.asarray(P[min(3, len(P) - 1)])
        args.start = [a[0], a[1], math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))]

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    import carb
    import omni
    import onnxruntime as ort
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils import stage as stage_utils
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from omni.physx import get_physx_scene_query_interface
    from PIL import Image
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, args.assets)
    from policy_runtime import PolicyRuntime

    # ---- 장면: walk_rbq_worker.py main() 과 같은 순서·설정 ----
    stage = omni.usd.get_context().get_stage()
    UsdPhysics.Scene.Define(stage, Sdf.Path('/physicsScene'))
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/physicsScene'))
    phx = PhysxSchema.PhysxSceneAPI.Get(stage, '/physicsScene')
    phx.CreateEnableCCDAttr(True)
    phx.CreateEnableStabilizationAttr(False)
    phx.CreateEnableGPUDynamicsAttr(True)
    phx.CreateBroadphaseTypeAttr('GPU')
    phx.CreateSolverTypeAttr('TGS')
    phx.CreateBounceThresholdAttr(0.5)
    UsdGeom.Xform.Define(stage, Sdf.Path('/World'))
    stage_utils.add_reference_to_stage(usd_path=usd, prim_path='/World/Background')
    app.update()
    n_col = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Background')):
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

    sim = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=0.05, stage_units_in_meters=1.0)
    cam = Camera(prim_path='/World/ViewCam', resolution=(args.width, args.height))
    robot = SingleArticulation(prim_path='/World/Robot/trunk', name='rbq10')
    sim.reset()
    robot.initialize()
    cam.initialize()
    fx = args.width / 2 / math.tan(math.radians(args.hfov_deg) / 2)
    ap_h = 20.955
    cam.set_horizontal_aperture(ap_h)
    cam.set_vertical_aperture(ap_h * args.height / args.width)
    cam.set_focal_length(fx * ap_h / args.width)
    cam.set_clipping_range(0.05, 200.0)
    print(f'[teleop] 배경 {usd} · 충돌 mesh {n_col} · 층 {[l["tag"] for l in layers]} · 정책 {args.policy}', flush=True)

    dof = list(robot.dof_names)
    n = robot.num_dof
    pol = PolicyRuntime(ort.InferenceSession(f'{args.assets}/RBQ/policy/{args.policy}', providers=['CPUExecutionProvider']))
    assert abs(pol.dt - PHYSICS_DT * DECIMATION) < 1e-9, f'policy_dt {pol.dt}'
    pol_to_sim = np.array([dof.index(j) for j in pol.joint_names])
    q_default, kps, kds = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
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
                hd = height_depth(pos, R)
                hd[hd < args.scan_obstacle_depth] = args.scan_fill
                src['height_depth'] = hd
            tgt = q_default.copy()
            tgt[pol_to_sim] = pol.step(src)
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
        sim.step(render=False)
        st['k'] += 1

    def layer_top(x, y):
        """순간이동할 바닥을 찾는 ray 시작 높이 = 지금 고른 층 바닥 최고점 + 1.5 m (위층 바닥에 걸리지 않게)."""
        return float(shared['layer']['zmax']) + 1.5

    def spawn(x, y, yaw):
        g = ground_below(x, y, layer_top(x, y))
        if g is None:
            return False
        p = np.array([x, y, layer_top(x, y) - g + args.spawn_h], np.float32)
        robot.set_joint_positions(q_default)
        robot.set_joint_velocities(np.zeros(n, np.float32))
        robot.set_world_pose(position=p, orientation=np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], np.float32))
        robot.set_linear_velocity(np.zeros(3, np.float32))
        robot.set_angular_velocity(np.zeros(3, np.float32))
        pol.reset()
        st.update(k=0, cur=q_default.copy(), prev=q_default.copy(), sub=0, active=False)
        for _ in range(int(args.warmup_s / PHYSICS_DT)):
            physics_step([0, 0, 0])
        st['active'] = True
        shared['fallen'] = False
        return True

    # ---- 브라우저와 주고받는 값 (HTTP 스레드 ↔ 메인 루프) ----
    lock = threading.Lock()
    shared = {'cmd': [0.0, 0.0, 0.0], 'cmd_t': 0.0, 'jobs': [], 'jpg': b'', 'fid': 0, 'pose': None, 'fallen': False,
              'layer': layers[0], 'fps': 0.0, 'msg': '',
              'cam_off': [0.0, 0.0, 0.0]}   # 카메라 옮기기 (로봇 앞 · 왼쪽 · 위, m). 0 = 원래 장착 위치
    html_path = Path(__file__).with_suffix('.html')  # read per request so page edits need no restart

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def do_GET(self):
            p = self.path.split('?')[0]
            if p == '/':
                self._send(200, html_path.read_bytes(), 'text/html; charset=utf-8')
            elif p == '/meta':
                self._json({'scene': args.scene, 'hfov_deg': args.hfov_deg, 'layers': layers})
            elif p.startswith('/bg/'):
                tag = p[4:].removesuffix('.png')
                f = L / f'paint_bg_{tag}.png'
                if tag not in {l['tag'] for l in layers} or not f.exists():
                    return self._send(404, b'no layer', 'text/plain')
                self._send(200, f.read_bytes(), 'image/png')
            elif p == '/state':
                with lock:
                    self._json({k: shared[k] for k in ('pose', 'fallen', 'fid', 'fps', 'msg')} | {'tag': shared['layer']['tag']})
            elif p == '/frame.jpg':
                with lock:
                    b = shared['jpg']
                self._send(200, b, 'image/jpeg') if b else self._send(503, b'not yet', 'text/plain')
            else:
                self._send(404, b'not found', 'text/plain')

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            except json.JSONDecodeError:
                return self._json({'error': 'json 이 아니다'}, 400)
            p = self.path
            if p == '/cmd':
                v = [float(np.clip(body.get(k, 0.0), -1.5, 1.5)) for k in ('vx', 'vy', 'wz')]   # 정책 학습 범위 (sim_docker 클램프 ±1.5)
                with lock:
                    shared['cmd'], shared['cmd_t'] = v, time.time()
                self._json({'ok': True})
            elif p == '/cam':      # 카메라 위치: {d: [앞, 왼쪽, 위]} 더하기 또는 {reset: true} (로봇 카메라로 되돌리기)
                with lock:
                    o = shared['cam_off']
                    if body.get('reset'):
                        o[:] = [0.0, 0.0, 0.0]
                    for i, v in enumerate(body.get('d') or [0, 0, 0]):   # 범위 제한 없음 (사용자 결정)
                        o[i] += float(v)
                    self._json({'ok': True, 'cam_off': list(o)})
            elif p in ('/teleport', '/reset', '/layer'):
                with lock:
                    shared['jobs'].append((p, body))
                self._json({'ok': True})
            elif p == '/doors':
                tag = body.get('tag')
                lay = next((l for l in layers if l['tag'] == tag), None)
                boxes = body.get('boxes')
                if lay is None or not isinstance(boxes, list):
                    return self._json({'error': f'모르는 층 {tag!r} 또는 boxes 없음'}, 400)
                boxes = [{k: round(float(b[k]), 3) for k in ('x0', 'y0', 'x1', 'y1')} for b in boxes]
                assert all(b['x0'] < b['x1'] and b['y0'] < b['y1'] for b in boxes), boxes
                f = L / f'doors_{tag}.json'
                f.write_text(json.dumps({'kind': 'doors', 'scene': args.scene, 'tag': tag, 'zlo': lay['zlo'], 'zhi': lay['zhi'],
                                         'zmin': lay['zmin'], 'zmax': lay['zmax'], 'boxes': boxes,
                                         'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, ensure_ascii=False, indent=1))
                os.chown(f, 1000, 1000)   # 컨테이너 root → 호스트 사용자
                lay['doors'] = boxes
                self._json({'ok': True, 'path': str(f), 'n': len(boxes)})
            else:
                self._json({'error': 'not found'}, 404)

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    M_cam = trunk_to_cam_cv()
    spawn(args.start[0], args.start[1], math.radians(args.start[2]))
    for _ in range(5):   # 렌더 파이프라인 워밍업 (04_render_obs_isaac: 첫 몇 프레임은 비어 있다)
        sim.render()
    print(f'[teleop] http://localhost:{args.port}  (출발 {args.start})', flush=True)

    t_wall0, k_wall0, t_fps, n_fps = time.time(), st['k'], time.time(), 0
    while app.is_running():
        with lock:
            jobs, shared['jobs'] = shared['jobs'], []
            cmd = shared['cmd'] if time.time() - shared['cmd_t'] < 0.5 else [0.0, 0.0, 0.0]   # 키를 뗐거나 페이지가 멈추면 정지
        for p, b in jobs:
            if p == '/layer':
                shared['layer'] = next((l for l in layers if l['tag'] == b.get('tag')), shared['layer'])
            elif p == '/teleport':
                if 'tag' in b:
                    shared['layer'] = next((l for l in layers if l['tag'] == b['tag']), shared['layer'])
                ok = spawn(float(b['x']), float(b['y']), float(b.get('yaw', shared['pose']['yaw'] if shared['pose'] else 0.0)))
                shared['msg'] = '' if ok else f'({b["x"]:.1f}, {b["y"]:.1f}) 아래에 바닥이 없다'
            elif p == '/reset':
                pp = shared['pose']
                spawn(pp['x'], pp['y'], pp['yaw'])
            else:
                assert False, f'unreachable job {p!r}'
            t_wall0, k_wall0 = time.time(), st['k']
        if shared['fallen']:
            cmd = [0.0, 0.0, 0.0]
        for _ in range(args.render_every):
            physics_step(cmd)
        pos, q = robot.get_world_pose()
        pos, q = np.asarray(pos, np.float64), np.asarray(q, np.float64)
        R = rot(q.astype(np.float32))
        yaw = math.atan2(R[1, 0], R[0, 0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        g = ground_below(pos[0], pos[1], pos[2] + SCAN_RAY_UP)
        h = (g - SCAN_RAY_UP) if g is not None else float('nan')
        fallen = tilt > args.fall_tilt_deg or (math.isfinite(h) and h < args.fall_h)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        c2w = T @ M_cam
        with lock:
            off = list(shared['cam_off'])
        if any(off):   # 카메라만 옮긴다 (방향 그대로). 앞·왼쪽은 로봇 진행 방향 기준 수평, 위는 세계 위
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            c2w[:3, 3] += [cy_ * off[0] - sy_ * off[1], sy_ * off[0] + cy_ * off[1], off[2]]
        qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        cam.set_world_pose(position=c2w[:3, 3], orientation=np.array([qw, qx, qy, qz]), camera_axes='ros')
        sim.render()
        rgba = cam.get_rgba()
        jpg = None
        if rgba is not None and rgba.size:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(buf, 'JPEG', quality=85)
            jpg = buf.getvalue()
        n_fps += 1
        if time.time() - t_fps > 1.0:
            fps, t_fps, n_fps = n_fps / (time.time() - t_fps), time.time(), 0
        else:
            fps = shared['fps']
        with lock:
            shared['pose'] = {'x': pos[0], 'y': pos[1], 'z': pos[2], 'yaw': yaw, 'tilt': tilt, 'h': h if math.isfinite(h) else None,
                              'cmd': cmd, 'R': R.tolist(), 'C': c2w.tolist(), 'cam_off': off}   # C = 이 프레임을 찍은 카메라 (OpenCV, 카메라→세계)   # R = 몸 회전 전체 (기울기 포함) — 렌더 카메라와 같은 자세로 되찍기용
            shared['fallen'] = shared['fallen'] or fallen
            shared['fps'] = fps
            if jpg:
                shared['jpg'], shared['fid'] = jpg, shared['fid'] + 1
        ahead = (st['k'] - k_wall0) * PHYSICS_DT - (time.time() - t_wall0)   # 실제 시간보다 빠르면 기다린다
        if ahead > 0:
            time.sleep(ahead)
    srv.shutdown()
    app.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
