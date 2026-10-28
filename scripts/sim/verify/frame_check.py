"""Frame-consistency check for vln_bridge_node outputs (frame_test.sh).

Uses the TRUE TF published by fake_sim (map->odom->base_link) as reference:
  goal : /vln/goal_marker expressed in base_link (via TF of its frame_id)  vs  /gdg/data/gp position
         (gdm reads it relative to the body at gp.odometry). The robot stands still in Mode A without gdm,
         so both must give the same body-frame point.
  path : /vln/trajectory expressed in base_link at its stamp  vs  the stub S1 path (x = 0.1 i, y = 0).
Prints per-sample errors and PASS/FAIL (goal <= 1 cm, path <= 1 cm).

Usage: python3 scripts/sim/verify/frame_check.py --ros-args -p kind:=goal -p seconds:=20
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

TOL = 0.01


def tf_to_T(t):
    tr, q = t.transform.translation, t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [tr.x, tr.y, tr.z]
    return T


class FrameCheck(Node):
    def __init__(self):
        super().__init__("frame_check")
        self.kind = self.declare_parameter("kind", "goal").value
        self.seconds = self.declare_parameter("seconds", 20.0).value
        self.skip = self.declare_parameter("traj_skip", 3).value
        self.buf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.buf, self)
        self.marker_base = None
        self.errs = []
        if self.kind == "goal":
            from gd_ifc_pkg.msg import GDGGuidancePoint

            self.create_subscription(Marker, "/vln/goal_marker", self.on_marker, 10)
            self.create_subscription(GDGGuidancePoint, "/gdg/data/gp", self.on_gp, 10)
        elif self.kind == "path":
            self.create_subscription(PathMsg, "/vln/trajectory", self.on_path, 10)
        else:
            assert False, f"unreachable kind={self.kind!r}"
        self.t0 = self.get_clock().now()

    def T_base(self, frame, stamp=None):
        if not self.buf.can_transform("base_link", frame, Time() if stamp is None else Time.from_msg(stamp),
                                      timeout=Duration(seconds=0.5)):
            return None  # TF not received yet (first messages)
        t = self.buf.lookup_transform("base_link", frame, Time() if stamp is None else Time.from_msg(stamp),
                                      timeout=Duration(seconds=0.5))
        return tf_to_T(t)

    def on_marker(self, m):
        p = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z, 1.0])
        T = self.T_base(m.header.frame_id)
        if T is not None:
            self.marker_base = (T @ p)[:2]

    def on_gp(self, m):
        if not m.position or self.marker_base is None:
            return
        rel = np.array([m.position[0].x, m.position[0].y])
        e = float(np.hypot(*(rel - self.marker_base)))
        self.errs.append(e)
        self.get_logger().info(f"goal: marker@base=({self.marker_base[0]:.3f}, {self.marker_base[1]:.3f}) "
                               f"gp.rel=({rel[0]:.3f}, {rel[1]:.3f}) err={e * 100:.1f} cm")

    def on_path(self, m):
        T = self.T_base(m.header.frame_id, m.header.stamp)
        if T is None:
            return
        pts = np.array([[p.pose.position.x, p.pose.position.y, 0.0, 1.0] for p in m.poses])
        base = (T @ pts.T).T[:, :2]
        ref = np.array([[0.1 * i, 0.0] for i in range(self.skip, self.skip + len(base))])
        e = float(np.max(np.hypot(*(base - ref).T)))
        self.errs.append(e)
        self.get_logger().info(f"path: first@base=({base[0, 0]:.3f}, {base[0, 1]:.3f}) max err={e * 100:.1f} cm")

    def done(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9 > self.seconds


def main():
    rclpy.init()
    n = FrameCheck()
    while rclpy.ok() and not n.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    errs = n.errs
    if not errs:
        print(f"[frame_check] {n.kind}: NO SAMPLES -> FAIL")
    else:
        ok = max(errs) <= TOL
        print(f"[frame_check] {n.kind}: n={len(errs)} max={max(errs) * 100:.1f} cm "
              f"median={float(np.median(errs)) * 100:.1f} cm -> {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
