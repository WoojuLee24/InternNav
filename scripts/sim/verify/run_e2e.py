"""V13 / V1d end-to-end episode runner on the live sim (server must already be running).

Per episode: teleport to the start pose -> launch vln_bridge_node (subprocess) -> record odom path,
closest lidar range (/ouster/scan, collision proxy) and /vln/status until S2 says STOP or timeout ->
stop the bridge -> save episode.json (+ the bridge csv/jpg log in the same folder).

Usage (inside gd_vln container, sim + gdm running; Mode A needs ./lpp-sim, Mode B ./lpp-sim-vln):
  python3 scripts/sim/verify/run_e2e.py --mode pixel_goal --tag dualvln_bf16 --episodes scripts/sim/verify/e2e_episodes_etri_indoor.json
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[3]  # InternNav


class Recorder(Node):
    def __init__(self):
        super().__init__("vln_e2e_recorder")
        self.path, self.min_range, self.status = [], float("inf"), []
        self.create_subscription(Odometry, "/gdq/msg/gdq_odom", self.on_odom, 20)
        self.create_subscription(LaserScan, "/ouster/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, "/vln/status", lambda m: self.status.append((time.time(), m.data)), 10)
        self.tp = self.create_publisher(Pose2D, "/sim/teleport", 1)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.recording = False

    def on_odom(self, m):
        if self.recording:
            p = m.pose.pose.position
            self.path.append((time.time(), p.x, p.y))

    def on_scan(self, m):
        if not self.recording:
            return
        r = np.asarray(m.ranges, np.float32)
        r = r[np.isfinite(r) & (r > max(m.range_min, 0.25))]  # 0.25: ignore self hits on the robot body
        if r.size:
            self.min_range = min(self.min_range, float(r.min()))


def spin_for(node, secs):
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pixel_goal", "trajectory"])
    ap.add_argument("--tag", required=True, help="e.g. dualvln_bf16 (model/quant served by the running server)")
    ap.add_argument("--episodes", required=True, help="json list of {id, instruction, start:[x,y,yaw_deg], timeout_s}")
    ap.add_argument("--only", default="", help="comma separated episode ids")
    ap.add_argument("--out", default="scripts/sim/verify/out/e2e")
    ap.add_argument("--bridge_args", default="", help='extra ros args, e.g. "lookdown_ns:=/camera_lookdown/camera"')
    ap.add_argument("--inspect", action="store_true", help="also run live_inspect.py per episode (<ep dir>/inspect*)")
    args = ap.parse_args()

    eps = json.loads(Path(args.episodes).read_text())
    if args.only:
        eps = [e for e in eps if e["id"] in args.only.split(",")]
    rclpy.init()
    rec = Recorder()
    summary = []
    for ep in eps:
        d = Path(args.out) / args.tag / args.mode / ep["id"]
        d.mkdir(parents=True, exist_ok=True)
        x, y, yaw = ep["start"]
        # the first message of a fresh publisher is lost before discovery completes (e1 started at y=-6.4):
        # wait for the sim subscriber, then confirm the robot is at the start pose
        while rec.tp.get_subscription_count() == 0:
            spin_for(rec, 0.2)
        rec.cmd.publish(Twist())
        for _ in range(3):
            rec.tp.publish(Pose2D(x=float(x), y=float(y), theta=math.radians(yaw)))
            rec.path, rec.recording = [], True
            spin_for(rec, 6.0)  # settle after teleport
            rec.recording = False
            if rec.path and math.hypot(rec.path[-1][1] - x, rec.path[-1][2] - y) < 0.5:
                break
        else:
            print(f"[run_e2e] WARN teleport to {ep['start']} not confirmed", flush=True)
        rec.path, rec.min_range, rec.status = [], float("inf"), []
        rec.recording = True
        cmd = ["python3", "scripts/sim/vln_bridge_node.py", "--ros-args", "-p", "use_sim_time:=true",
               "-p", f"mode:={args.mode}", "-p", f"log_dir:={d}", "-p", f"instruction:={ep['instruction']}"]
        for kv in args.bridge_args.split():
            cmd += ["-p", kv]
        log = open(d / "bridge.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        insp = None
        if args.inspect:
            icmd = ["python3", "scripts/sim/verify/live_inspect.py", "--ros-args", "-p", "use_sim_time:=true",
                    "-p", f"out:={d}", "-p", f"mode:={args.mode}"]
            if ep.get("goal_xy") is not None:
                icmd += ["-p", f"goal:={ep['goal_xy'][0]},{ep['goal_xy'][1]}"]
            insp = subprocess.Popen(icmd, cwd=ROOT, stdout=open(d / "inspect.log", "w"), stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
        t0 = time.time()
        stopped = False
        while time.time() - t0 < ep.get("timeout_s", 150):
            spin_for(rec, 0.5)
            if any(s == "STOP" for _, s in rec.status):
                stopped = True
                spin_for(rec, 3.0)  # let the robot come to rest
                break
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        if insp is not None:  # SIGINT -> analysis + inspect.png
            os.killpg(insp.pid, signal.SIGINT)
            try:
                insp.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(insp.pid, signal.SIGKILL)
        rec.recording = False
        rec.cmd.publish(Twist())
        p = np.array([(px, py) for _, px, py in rec.path]) if rec.path else np.zeros((1, 2))
        res = {
            "id": ep["id"], "tag": args.tag, "mode": args.mode, "instruction": ep["instruction"],
            "stopped_by_model": stopped, "duration_s": round(time.time() - t0, 1),
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0,
            "start_xy": [float(x), float(y)], "final_xy": p[-1].tolist(),
            "min_scan_range_m": rec.min_range if math.isfinite(rec.min_range) else None,
        }
        goal = ep.get("goal_xy")
        if goal is not None:
            res["final_dist_to_goal_m"] = float(np.hypot(p[-1][0] - goal[0], p[-1][1] - goal[1]))
            res["success"] = bool(stopped and res["final_dist_to_goal_m"] <= ep.get("success_radius_m", 1.5))
        np.savetxt(d / "path.csv", p, delimiter=",", header="x,y", comments="")
        (d / "episode.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
        summary.append(res)
    (Path(args.out) / args.tag / args.mode / "summary.json").write_text(json.dumps(summary, indent=1))
    rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
