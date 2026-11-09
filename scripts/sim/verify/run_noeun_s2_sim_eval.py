"""노은역 val 을 live sim(RBQ10, sim_docker) 에서 **물리 보행**으로 평가한다 — VLN-PE(H1 flash) 트랙의 sim 짝.

VLN-PE 트랙과 같은 것: 이산 동작 (0.25 m 전진 · 15 deg 회전 · 제자리), 같은 S2 에이전트(agent server, 같은 config
`scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py` — NOEUN_AGENT=internvla_n1_s2_gguf 면 경량화 모델), 같은 입력
(640x480 · hfov 68.7 — sim 카메라 1280x720 hfov 90.5 에서 hfov 68.7 이 되게 가운데를 잘라 줄인다), 같은 지표.
다른 것: 이산 동작을 RBQ10 보행 정책이 /cmd_vel PID 로 실제로 걸어서 한다 (vln_bridge_node 와 같은 PID ·
control_utils.discrete_step). 순간이동은 ep 시작 자세에 세울 때만 (sim 을 GUIDEDOG_INIT_* 로 띄운다 — 호스트
`scripts/sim/run_noeun_sim_eval.sh` 가 ep 마다 sim 을 다시 띄우고 이 스크립트를 --only 로 부른다).

| policy | 뜻 |
|---|---|
| `gt` | reference_path 를 VLN-PE GT 재생(`scripts/eval/noeun_gt_replay_check.py`)과 같은 규칙으로 따라간다 — 보행 검증 |
| `model` | S2 에이전트 (agent server 가 떠 있어야 한다) |

| 지표 | 뜻 |
|---|---|
| `ne_m` | 멈춘 곳(발) ↔ 도착점 3D 거리 |
| `success` | STOP 했고 xy 거리 ≤ 3 m 이고 높이 차 ≤ success_dz (VLN-PE NoeunVLNPEMetric 과 같은 같은-층 판정) |
| `oracle_success` | 지나간 점 중 하나라도 성공 반경 안 (같은 층) |
| `spl` | success × L / max(L, 걸은 길이) |
| `fall` | trunk 가 45 deg 넘게 기울어짐 (넘어짐) |

좌표: ep 는 habitat Y-up → sim world = [x, -z, y] (run_r2r_eval.py 와 같음). sim odom = world.

사용 (gd_vln 컨테이너, sim 이 ep 시작 자세로 떠 있어야 한다):
  python3 scripts/sim/verify/run_noeun_s2_sim_eval.py --policy gt --tag gt --episodes <val_seen.json.gz> --only 39
"""
import argparse
import gzip
import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "realworld"))
from control_utils import discrete_step  # noqa: E402
from controllers import PID_controller  # noqa: E402
from projection import CropResize, Intrinsics, make_T, yaw_of  # noqa: E402
from run_r2r_eval import BASE_H, hab_yaw_to_world, to_world  # noqa: E402

ROOT = HERE.parents[2]  # InternNav
STOP, FORWARD, LEFT, RIGHT, STAND = 0, 1, 2, 3, -1
FORWARD_DIST, TURN_RAD = 0.25, math.radians(15.0)
GT_CLOSE_ENOUGH, GT_MAX_STEPS_PER_WP = FORWARD_DIST * 0.6, 400  # noeun_gt_replay_check.py 와 같다
OPT_TO_USD = np.diag([1.0, -1.0, -1.0])  # ROS optical (x 오른쪽, y 아래, z 전방) -> USD 카메라 (+Y 위, -Z 전방)


def R_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def hfov_crop(intr: Intrinsics, out_w, out_h, hfov_deg):
    """intr 카메라에서 hfov_deg 가 되게 주점 중심으로 잘라 out_w x out_h 로 줄이는 CropResize."""
    crop_w = int(round(2.0 * intr.fx * math.tan(math.radians(hfov_deg) / 2.0)))
    crop_h = int(round(crop_w * out_h / out_w))
    assert crop_w <= intr.width and crop_h <= intr.height, f"hfov {hfov_deg} needs {crop_w}x{crop_h} > {intr.width}x{intr.height}"
    x0, y0 = int(round(intr.cx + 0.5 - crop_w / 2.0)), int(round(intr.cy + 0.5 - crop_h / 2.0))
    return CropResize(x0, y0, crop_w, crop_h, out_w, out_h)


class SimIO(Node):
    """카메라 · odom · TF 를 받고 /cmd_vel 로 이산 동작을 걷는다."""

    def __init__(self, args):
        super().__init__("noeun_s2_sim_eval")
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.crop, self.latest, self.odom = None, None, None
        self.T_base_cam = None
        self.path = []  # (sim stamp, x, y, z)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/gdq/msg/gdq_odom", self.on_odom, 20)
        sub_c = Subscriber(self, Image, "/camera/camera/color/image_raw", qos_profile=qos_profile_sensor_data)
        sub_d = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw", qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([sub_c, sub_d], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_rgbd)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 5)
        self.pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=args.v_max, max_w=args.w_max)

    def on_info(self, msg):
        if self.crop is None:
            intr = Intrinsics.from_K(msg.k, msg.width, msg.height)
            self.crop = hfov_crop(intr, self.args.width, self.args.height, self.args.hfov)
            self.get_logger().info(f"camera {msg.width}x{msg.height} fx={intr.fx:.1f} -> crop {self.crop.crop_w}x{self.crop.crop_h}"
                                   f"@({self.crop.x0},{self.crop.y0}) -> {self.args.width}x{self.args.height} hfov {self.args.hfov}")

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        T = make_T((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        with self.lock:
            self.odom = T
            self.path.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, p.x, p.y, p.z))

    def on_rgbd(self, color, depth):
        if self.crop is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(color, "rgb8")
        d = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        if d.dtype == np.uint16:  # 16UC1 mm (physx depth)
            d = d.astype(np.float32) / 1000.0
        elif d.dtype == np.float32:  # 32FC1 m (rtx depth)
            d = d.astype(np.float32)
        else:
            assert False, f"unreachable depth dtype={d.dtype!r}"
        if self.T_base_cam is None:
            try:
                tf = self.tf_buffer.lookup_transform("base_link", depth.header.frame_id, Time())
                t, r = tf.transform.translation, tf.transform.rotation
                self.T_base_cam = make_T((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))
            except Exception:
                return
        with self.lock:
            self.latest = (time.time(), self.crop.apply_rgb(rgb), self.crop.apply_depth(d))

    # ---------------------------------------------------------------- helpers
    def robot_T(self):
        with self.lock:
            return None if self.odom is None else self.odom.copy()

    def wait_ready(self, timeout=120.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                ok = self.odom is not None and self.latest is not None
            if ok:
                return True
            time.sleep(0.2)
        return False

    def fresh_frame(self, after, timeout=10.0):
        """after(wall time) 뒤에 들어온 프레임 + 그때 base 자세."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                lt = self.latest
            if lt is not None and lt[0] > after:
                return lt[1], lt[2], self.robot_T()
            time.sleep(0.02)
        raise RuntimeError("no fresh camera frame")

    def publish(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd.publish(t)

    def execute(self, action):
        """이산 동작 하나를 걷는다 (vln_bridge_node PID 와 같은 제어). 끝난 wall 시각을 돌려준다."""
        if action == STAND:
            time.sleep(self.args.stand_s)
            return time.time()
        T = self.robot_T()
        goal = T.copy()
        yaw = yaw_of(goal)
        if action == FORWARD:
            goal[0, 3] += FORWARD_DIST * math.cos(yaw)
            goal[1, 3] += FORWARD_DIST * math.sin(yaw)
        elif action in (LEFT, RIGHT):
            dyaw = TURN_RAD if action == LEFT else -TURN_RAD
            c, s = math.cos(dyaw), math.sin(dyaw)
            goal[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ goal[:3, :3]
        else:
            assert False, f"unreachable action={action!r}"
        # 보행은 15 deg 명령에 2.5~19 deg, 0.25 m 명령에 ~0.19 m 를 간다 (2026-11-07 ep39 실측) -> 끝난 뒤 남은 오차가
        # correct_tol_* 보다 크면 같은 goal 로 다시 걷는다 (최대 correct_iters 번). 0 이면 한 번만 (vln_bridge_node 와 같음)
        for it in range(1 + self.args.correct_iters):
            self._pid_to(goal)
            e_p, e_r = self.pid.calculate_errors(self.robot_T(), goal)
            if abs(e_p) <= self.args.correct_tol_m and abs(e_r) <= math.radians(self.args.correct_tol_deg):
                break
        return time.time()

    def _pid_to(self, goal):
        prev, t_start = None, time.time()
        while time.time() - t_start < self.args.action_timeout_s:
            T = self.robot_T()
            v, w, e_p, e_r = self.pid.solve(T, goal, np.zeros(2))
            now = time.time()
            v_meas = w_meas = 0.0
            if prev is not None and now - prev[0] < 0.5:
                dt = now - prev[0]
                d = T[:2, 3] - prev[1][:2, 3]
                y0 = yaw_of(prev[1])
                v_meas = (d[0] * math.cos(y0) + d[1] * math.sin(y0)) / dt
                w_meas = math.atan2(math.sin(yaw_of(T) - y0), math.cos(yaw_of(T) - y0)) / dt
            prev = (now, T)
            v, w, done = discrete_step(v, w, e_p, e_r, v_meas, w_meas, self.args.pid_v_min, self.args.pid_w_min,
                                       lead_s=self.args.pid_lead_s)
            self.publish(max(0.0, v), w)
            if done:
                break
            time.sleep(0.1)
        self.publish(0.0, 0.0)
        time.sleep(self.args.settle_s)

    def obs(self, rgb, depth_m, T_base, instruction):
        """VLN-PE 에이전트 관측 (internvla_n1_s2_agent.step 이 읽는 키)."""
        T_cam = T_base @ self.T_base_cam
        return {
            "rgb": rgb,
            "depth": (np.clip(np.nan_to_num(depth_m, nan=0.0), 0.0, 10.0) / 10.0)[..., None].astype(np.float32),
            "globalgps": T_base[:3, 3].copy(),
            "globalrotation": R_to_quat_wxyz(T_base[:3, :3]),
            "camera_position": T_cam[:3, 3].copy(),
            "camera_orientation": R_to_quat_wxyz(T_cam[:3, :3] @ OPT_TO_USD),
            "instruction": instruction,
        }


class AgentLink:
    """agent server (scripts/eval/start_server.py) — 이미 init 된 에이전트가 있으면 다시 load 하지 않고 reset 만."""

    def __init__(self, agent_cfg):
        from internnav.configs.agent import InitRequest, ResetRequest, StepRequest
        from internnav.utils.comm_utils.client import serialize_obs

        self.ResetRequest, self.StepRequest, self.serialize = ResetRequest, StepRequest, serialize_obs
        self.InitRequest, self.cfg = InitRequest, agent_cfg
        self.base = f"http://{agent_cfg.server_host}:{agent_cfg.server_port}"
        self.name = agent_cfg.model_name

    def reset(self):
        """ep 시작: 서버에 에이전트가 있으면 reset, 없으면 init (모델 load)."""
        r = requests.post(f"{self.base}/agent/{self.name}/reset", json=self.ResetRequest(reset_index=[0]).model_dump(mode="json"))
        if r.status_code == 404:
            r = requests.post(f"{self.base}/agent/init", json=self.InitRequest(agent_config=self.cfg).model_dump(mode="json"))
            r.raise_for_status()
            self.name = r.json()["agent_name"]
        else:
            r.raise_for_status()

    def step(self, ob):
        r = requests.post(f"{self.base}/agent/{self.name}/step",
                          json=self.StepRequest(observation=self.serialize([ob])).model_dump(mode="json"), timeout=120)
        r.raise_for_status()
        return int(r.json()["action"][0]["action"][0])


def load_agent_cfg(path):
    spec = importlib.util.spec_from_file_location("eval_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.eval_cfg.agent, mod.eval_cfg.task.task_settings


def gt_actions(io, ref):
    """reference_path 를 noeun_gt_replay_check.py 와 같은 규칙으로 따라가는 행동 생성기."""
    for wp in ref[1:]:
        for _ in range(GT_MAX_STEPS_PER_WP):
            T = io.robot_T()
            dp = wp[:2] - T[:2, 3]
            if np.linalg.norm(dp) < GT_CLOSE_ENOUGH:
                break
            delta = math.atan2(dp[1], dp[0]) - yaw_of(T)
            delta = ((delta + math.pi) % (2 * math.pi)) - math.pi
            yield (LEFT if delta > 0 else RIGHT) if abs(delta) > TURN_RAD * 0.5 else FORWARD
        else:
            return  # 이 waypoint 에 못 감 -> 포기 (GT 재생과 같다)
    yield STOP


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=["gt", "model"])
    ap.add_argument("--tag", required=True, help="출력 폴더 이름 (모델/구성)")
    ap.add_argument("--episodes", required=True, help="raw_data/r2r/<split>/<split>.json.gz")
    ap.add_argument("--only", default="", help="쉼표로 episode_id (호스트 스크립트가 ep 하나씩 준다)")
    ap.add_argument("--config", default="scripts/eval/configs/noeun/h1_noeun_s2_val_seen_cfg.py", help="VLN-PE 와 같은 agent 설정")
    ap.add_argument("--out", default="scripts/sim/verify/out/noeun_s2_sim")
    ap.add_argument("--max_step", type=int, default=0, help="0 = config task_settings.max_step (VLN-PE 와 같음)")
    ap.add_argument("--success_r", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=68.7, help="60cm_15deg rig (학습 이미지) hfov")
    ap.add_argument("--stand_s", type=float, default=1.0, help="제자리(WAIT · ↓) 한 step 동안 기다리는 시간")
    ap.add_argument("--settle_s", type=float, default=0.3, help="동작 뒤 다음 관측 전 기다림")
    ap.add_argument("--action_timeout_s", type=float, default=12.0, help="vln_bridge_node discrete_block_max_s 와 같다")
    ap.add_argument("--v_max", type=float, default=0.4)
    ap.add_argument("--w_max", type=float, default=0.4)
    ap.add_argument("--pid_v_min", type=float, default=0.3)
    ap.add_argument("--pid_w_min", type=float, default=0.4)
    ap.add_argument("--pid_lead_s", type=float, default=0.45)
    ap.add_argument("--correct_iters", type=int, default=2, help="동작 뒤 오차가 크면 다시 걷는 횟수 (0 = 한 번만)")
    ap.add_argument("--correct_tol_m", type=float, default=0.05, help="다시 걷지 않는 위치 오차")
    ap.add_argument("--correct_tol_deg", type=float, default=3.0, help="다시 걷지 않는 방향 오차")
    ap.add_argument("--stand_up_s", type=float, default=5.0, help="첫 수신 뒤 일어설 때까지 기다림")
    ap.add_argument("--ready_timeout_s", type=float, default=600.0, help="sim 카메라 · odom 첫 수신 대기 (장면 로딩 ≈ 2 분)")
    ap.add_argument("--fall_tilt_deg", type=float, default=45.0, help="trunk 가 이만큼 넘게 기울면 넘어짐 (계단은 높이가 바뀌므로 높이로 보지 않는다)")
    ap.add_argument("--blocked_steps", type=int, default=10,
                    help="전진이 이 횟수만큼 못 나가면 (그 사이 회전은 세지 않음) 막힘으로 ep 을 끝낸다. 0 = 끔 (2026-11-08 GT ep7: 개찰구에 219 번 연속 막힘. 성공한 ep2 도 6 번 연속이 있어 5 는 너무 짧다)")
    ap.add_argument("--blocked_m", type=float, default=0.05, help="전진 한 번에 이보다 덜 가면 못 나간 것 (명령 0.25 m, 실제 ~0.19 m)")
    args = ap.parse_args()

    agent_cfg, task_settings = load_agent_cfg(ROOT / args.config)
    max_step = args.max_step or int(task_settings["max_step"])
    success_dz = task_settings.get("success_dz")
    eps = json.load(gzip.open(args.episodes))["episodes"]
    if args.only:
        eps = [e for e in eps if str(e["episode_id"]) in args.only.split(",")]
    assert eps, f"no episode for --only {args.only!r}"

    rclpy.init()
    io = SimIO(args)
    ex = MultiThreadedExecutor()
    ex.add_node(io)
    threading.Thread(target=ex.spin, daemon=True).start()
    assert io.wait_ready(args.ready_timeout_s),"sim 카메라 · odom 이 오지 않는다 (sim 이 떠 있나? ROS_DOMAIN_ID / FASTDDS_BUILTIN_TRANSPORTS 확인)"
    time.sleep(args.stand_up_s)  # 스폰 직후 보행 정책이 일어서는 동안
    agent = AgentLink(agent_cfg) if args.policy == "model" else None

    for ep in eps:
        eid = str(ep["episode_id"])
        d = Path(args.out) / args.tag / eid
        d.mkdir(parents=True, exist_ok=True)
        start, goal = to_world(ep["start_position"]), to_world(ep["goals"][0]["position"])
        ref = np.array([to_world(p) for p in ep["reference_path"]])
        instr = ep["instruction"]["instruction_text"]
        T0 = io.robot_T()
        start_err = float(np.hypot(*(T0[:2, 3] - start[:2])))
        with io.lock:
            io.path = []
        if args.policy == "gt":
            gen = gt_actions(io, ref)
        elif args.policy == "model":
            agent.reset()
        else:
            assert False, f"unreachable policy={args.policy!r}"

        rows, stopped, fall, t_wall, t_last = [], False, False, time.time(), time.time()
        blocked, n_noprog = False, 0  # n_noprog = 마지막으로 앞으로 나간 뒤 못 나간 전진 횟수
        f_steps = open(d / "steps.jsonl", "w")  # 걸음마다 남긴다 (ep 중간에도 어디서 막혔는지 볼 수 있게)
        for step in range(max_step):
            rgb, depth_m, T = io.fresh_frame(t_last)
            if args.policy == "gt":
                a = next(gen, None)
                if a is None:
                    break
            else:
                a = agent.step(io.obs(rgb, depth_m, T, instr))
            rows.append({"step": step, "a": a, "pos": np.round(T[:3, 3], 3).tolist(), "yaw_deg": round(math.degrees(yaw_of(T)), 1),
                         "t": round(time.time() - t_wall, 2)})
            f_steps.write(json.dumps(rows[-1]) + "\n")
            f_steps.flush()
            if a == STOP:
                stopped = True
                break
            t_last = io.execute(a)
            tilt = math.degrees(math.acos(np.clip(io.robot_T()[2, 2], -1.0, 1.0)))
            if tilt > args.fall_tilt_deg:
                fall = True
                break
            if a == FORWARD:
                moved = float(np.linalg.norm(io.robot_T()[:2, 3] - T[:2, 3]))
                n_noprog = 0 if moved >= args.blocked_m else n_noprog + 1
                rows[-1]["moved_m"] = round(moved, 3)
            if args.blocked_steps and n_noprog >= args.blocked_steps:
                blocked = True
                break
        io.publish(0.0, 0.0)
        f_steps.close()

        with io.lock:
            path = list(io.path)
        P = np.array([p[1:] for p in path]) if path else T0[:3, 3][None]
        feet = P - [0, 0, BASE_H]
        # 걸은 길이 = 걸음(동작) 사이 위치 차의 합. odom 을 다 이으면 4족 몸통 흔들림이 더해져 ~2 배가 된다 (ep39 실측)
        S = np.array([r["pos"] for r in rows] + [P[-1].tolist()])
        walked = float(np.linalg.norm(np.diff(S[:, :2], axis=0), axis=1).sum()) if len(S) > 1 else 0.0
        dxy = np.linalg.norm(feet[:, :2] - goal[:2], axis=1)
        dz = np.abs(feet[:, 2] - goal[2])
        same = dz <= success_dz if success_dz is not None else np.ones(len(dz), bool)
        ok = (dxy <= args.success_r) & same
        L = float(ep["info"]["geodesic_distance"])
        res = {"episode_id": ep["episode_id"], "key": ep.get("key"), "tag": args.tag, "policy": args.policy, "instruction": instr,
               "start_err_m": round(start_err, 3), "steps": len(rows), "stopped": stopped, "fall": fall, "blocked": blocked,
               "duration_s": round(time.time() - t_wall, 1), "geodesic_distance_m": L, "path_length_m": round(walked, 3),
               "ne_m": round(float(np.linalg.norm(feet[-1] - goal)), 3), "ne_xy_m": round(float(dxy[-1]), 3),
               "dz_m": round(float(feet[-1][2] - goal[2]), 3),
               "success": bool(stopped and not fall and ok[-1]), "oracle_success": bool(ok.any())}
        res["spl"] = round(float(res["success"]) * L / max(L, walked, 1e-9), 4)
        np.savetxt(d / "path.csv", np.array(path) if path else np.zeros((0, 4)), delimiter=",", header="stamp,x,y,z", comments="")
        with open(d / "steps.jsonl", "w") as f:  # moved_m 이 붙은 최종본으로 다시 쓴다
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (d / "episode.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print("[noeun_sim] " + json.dumps(res, ensure_ascii=False), flush=True)
    ex.shutdown()  # spin 스레드를 먼저 멈춘다 (아니면 context 종료 뒤 RCLError traceback 이 남는다)
    rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
