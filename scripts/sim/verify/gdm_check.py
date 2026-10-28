"""④ delivery check on the live sim: bridge (Mode A) -> /gdg/data/gp -> gdm planner -> robot (gdm_test.sh).

With stub_server.py --once the bridge sends exactly one goal. Measures
  gp      : gdm's reading of the GP (gp.odometry pose x relative pose)  vs  /vln/goal_marker  (<= 1 cm PASS)
  plan    : end of the first /gdm/local_plan_map after the GP (TF to the marker frame)  vs  goal, and GP -> plan latency
  arrive  : robot (odom) distance to the goal when the run ends, and the closest it got
  cmd_vel : publishers on /gdq/msg/cmd_vel (1 = gdm only, PASS)
Usage: python3 scripts/sim/verify/gdm_check.py --ros-args -p use_sim_time:=true -p seconds:=40.0
"""
import json
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from frame_check import tf_to_T


def pose_T(p):
    q = p.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    T = np.eye(3)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:2, 2] = [p.position.x, p.position.y]
    return T


class GdmCheck(Node):
    def __init__(self):
        super().__init__("gdm_check")
        from gd_ifc_pkg.msg import GDGGuidancePoint

        self.seconds = self.declare_parameter("seconds", 40.0).value
        self.buf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.buf, self)
        self.marker = None   # (frame, xy)
        self.goal = None     # xy in odom (gdm reading of the GP)
        self.t_gp = None
        self.res = {}
        self.robot, self.min_d = None, float("inf")
        self.create_subscription(Marker, "/vln/goal_marker", self.on_marker, 10)
        self.create_subscription(GDGGuidancePoint, "/gdg/data/gp", self.on_gp, 10)
        self.create_subscription(PathMsg, "/gdm/local_plan_map", self.on_plan, 10)
        self.create_subscription(Odometry, "/gdq/msg/gdq_odom", self.on_odom, 20)
        self.t0 = time.time()  # wall clock: sim /clock starts at 0 before the first message

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_marker(self, m):
        self.marker = (m.header.frame_id, np.array([m.pose.position.x, m.pose.position.y]))

    def on_gp(self, m):
        if not m.position or self.goal is not None:
            return
        rel = np.array([m.position[0].x, m.position[0].y, 1.0])
        self.goal = (pose_T(m.odometry.pose.pose) @ rel)[:2]
        self.res["gp_rel_xy"] = [round(float(rel[0]), 3), round(float(rel[1]), 3)]  # body frame: x fwd, y LEFT
        o = m.odometry.pose.pose
        self.res["robot_at_gp"] = [round(o.position.x, 3), round(o.position.y, 3),
                                   round(math.degrees(math.atan2(pose_T(o)[1, 0], pose_T(o)[0, 0])), 1)]
        self.t_gp = time.time()
        if self.marker is not None:
            self.res["gp_vs_marker_cm"] = float(np.hypot(*(self.goal - self.marker[1]))) * 100
        self.get_logger().info(f"GP -> goal (odom) {self.goal.round(3)} marker {self.marker}")

    def on_plan(self, m):
        if self.goal is None or "plan_latency_s" in self.res or not m.poses:
            return
        try:
            T = tf_to_T(self.buf.lookup_transform("odom", m.header.frame_id, Time(), timeout=Duration(seconds=0.2)))
        except Exception:
            return
        p = m.poses[-1].pose.position
        end = (T @ np.array([p.x, p.y, p.z, 1.0]))[:2]
        self.res["plan_latency_s"] = time.time() - self.t_gp
        self.res["plan_end_vs_goal_m"] = float(np.hypot(*(end - self.goal)))
        self.res["plan_frame"] = m.header.frame_id
        self.get_logger().info(f"first plan after GP: end {end.round(3)} ({m.header.frame_id}) "
                               f"latency {self.res['plan_latency_s']:.2f}s")

    def on_odom(self, m):
        self.robot = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
        if self.goal is not None:
            self.min_d = min(self.min_d, float(np.hypot(*(self.robot - self.goal))))

    def done(self):
        return time.time() - self.t0 > self.seconds


def main():
    rclpy.init()
    n = GdmCheck()
    while rclpy.ok() and not n.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    r = n.res
    r["cmd_vel_publishers"] = n.count_publishers("/gdq/msg/cmd_vel")
    if n.goal is not None and n.robot is not None:
        r["final_dist_m"] = float(np.hypot(*(n.robot - n.goal)))
        r["min_dist_m"] = n.min_d
    ok = r.get("gp_vs_marker_cm", 1e9) <= 1.0 and r["cmd_vel_publishers"] == 1 and "plan_latency_s" in r
    print("[gdm_check] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
          + f" -> {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
