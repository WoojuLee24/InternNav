"""V10-5: discrete-action PID accuracy on the live sim (same gains / stop rule as vln_bridge_node).

Executes left 15°, right 15°, forward 0.25 m (x3 each) from the current pose and reports the final error.
Publishes on /cmd_vel (primary, overrides gdm). Usage:
  python3 scripts/sim/verify/verify_pid.py --ros-args -p use_sim_time:=true
"""
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "realworld"))
from control_utils import discrete_step  # noqa: E402
from controllers import PID_controller  # noqa: E402
from projection import make_T, se2, yaw_of  # noqa: E402


LEAD = float(sys.argv[sys.argv.index("--lead") + 1]) if "--lead" in sys.argv else 0.45  # = vln_bridge_node pid_lead_s


def main():
    rclpy.init()
    n = Node("vln_verify_pid")
    buf = Buffer()
    TransformListener(buf, n)
    pub = n.create_publisher(Twist, "/cmd_vel", 5)
    threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
    pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.4, max_w=0.4)

    def robot():
        t = buf.lookup_transform("map", "base_link", Time()).transform
        return make_T([t.translation.x, t.translation.y, t.translation.z],
                      [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])

    time.sleep(2.0)
    results = []
    for name, (dx, dyaw) in [("left15", (0, 15)), ("right15", (0, -15)), ("fwd0.25", (0.25, 0))] * 3:
        T0 = robot()
        goal = T0.copy()
        goal[:2, 3] += dx * np.array([math.cos(yaw_of(T0)), math.sin(yaw_of(T0))])
        goal[:3, :3] = se2(0, 0, math.radians(dyaw))[:3, :3] @ goal[:3, :3]
        t0 = time.time()
        prev = (t0, T0)
        while time.time() - t0 < 10.0:
            T = robot()
            now = time.time()
            dt = max(1e-3, now - prev[0])
            yaw0 = yaw_of(prev[1])
            d = T[:2, 3] - prev[1][:2, 3]
            v_meas = (d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0)) / dt
            w_meas = math.atan2(math.sin(yaw_of(T) - yaw0), math.cos(yaw_of(T) - yaw0)) / dt
            prev = (now, T)
            v, w, e_p, e_r = pid.solve(T, goal, np.zeros(2))
            v, w, done = discrete_step(v, w, e_p, e_r, v_meas, w_meas, lead_s=LEAD)
            if done:
                break
            tw = Twist()
            tw.linear.x, tw.angular.z = float(max(0.0, v)), float(w)
            pub.publish(tw)
            time.sleep(0.1)
        pub.publish(Twist())
        time.sleep(1.5)  # settle (sim keeps the last command 0.5 s)
        T1 = robot()
        e_yaw = math.degrees(math.atan2(math.sin(yaw_of(goal) - yaw_of(T1)), math.cos(yaw_of(goal) - yaw_of(T1))))
        moved = float(np.hypot(*(T1[:2, 3] - T0[:2, 3])))
        results.append({"cmd": name, "yaw_err_deg": e_yaw, "moved_m": moved, "pos_err_m": float(np.hypot(*(T1[:2, 3] - goal[:2, 3]))),
                        "t_s": time.time() - t0})
        print(json.dumps(results[-1]), flush=True)
    summ = {
        "turn_abs_yaw_err_max_deg": max(abs(r["yaw_err_deg"]) for r in results if "15" in r["cmd"]),
        "fwd_pos_err_max_m": max(r["pos_err_m"] for r in results if r["cmd"].startswith("fwd")),
    }
    print(json.dumps(summ), flush=True)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
