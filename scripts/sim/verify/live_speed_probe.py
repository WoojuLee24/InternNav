"""Live 40 s straight-run probe: robot wall speed, sim RTF, reply rate / latency from the bridge log.
Usage (gd_vln container, bridge running, web_viz on :8090): python3 scripts/sim/verify/live_speed_probe.py <bridge_log_dir> [secs]
"""
import csv, glob, json, sys, time, urllib.request
import numpy as np, rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

def post(path, d):
    urllib.request.urlopen(urllib.request.Request("http://localhost:8090" + path, data=json.dumps(d).encode(),
                                                  headers={"Content-Type": "application/json"}))

def main():
    logdir, secs = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 40
    rclpy.init(); n = Node("live_speed_probe"); rows = []
    n.create_subscription(Odometry, "/gdq/msg/gdq_odom", lambda m: rows.append(
        (time.time(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.x, m.pose.pose.position.y)), 50)
    post("/teleport", {"x": 0, "y": 0, "yaw_deg": -90}); t_end = time.time() + 7
    while time.time() < t_end: rclpy.spin_once(n, timeout_sec=0.05)
    rows.clear(); t0 = time.time()
    post("/instruction", {"text": "Walk straight down the hallway and stop at the intersection."})
    while time.time() - t0 < secs: rclpy.spin_once(n, timeout_sec=0.02)
    post("/instruction", {"text": ""})
    a = np.array(rows); d = np.hypot(np.diff(a[:, 2]), np.diff(a[:, 3])).sum()
    f = sorted(glob.glob(f"{logdir}/vln_*.csv"))[-1]
    rep = [r for r in csv.DictReader(open(f)) if float(r["wall"]) > t0]
    t_inf = np.array([float(r["t_infer"]) for r in rep]) * 1000
    walls = np.array([float(r["wall"]) for r in rep])
    print(json.dumps({"sim_RTF": round((a[-1, 1] - a[0, 1]) / (a[-1, 0] - a[0, 0]), 2),
                      "dist_m": round(float(d), 2), "wall_speed_mps": round(float(d) / secs, 3),
                      "replies": len(rep), "reply_interval_s_p50/max": [round(float(np.median(np.diff(walls))), 2), round(float(np.diff(walls).max()), 2)],
                      "server_ms_p50/p95/max": [round(float(np.percentile(t_inf, 50))), round(float(np.percentile(t_inf, 95))), round(float(t_inf.max()))],
                      "n_s2": sum(1 for r in rep if r["llm"])}))

if __name__ == "__main__":
    main()
