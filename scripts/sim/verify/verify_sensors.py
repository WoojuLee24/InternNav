"""Runtime sensor/geometry checks against the running sim (no model needed).

  check:=rates   V0  topic rates + TF availability for 10 s
  check:=geom    V2/V4/V6  one synced RGB-D + /ouster/points snapshot:
                  - overlay of lidar points on the processed (cropped 640x480) image -> png
                  - depth image vs lidar z (in camera frame) at lidar pixels -> error stats (V4)
                  - grid pixels -> 3D via depth median and via ground ray -> distance to nearest lidar
                    point, binned by range (V6)
  check:=twist   V7a record odom while the robot is driven (teleop): compare twist against the
                  pose-difference velocity expressed in world vs body frame
  check:=sync    V8  tf2 lookup map<-base_link at every RGB-D stamp for 20 s

Results: scripts/sim/verify/out/sensors_<check>_<time>/ (csv/png/json).
Usage: python3 scripts/sim/verify/verify_sensors.py --ros-args -p use_sim_time:=true -p check:=geom
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projection import (  # noqa: E402
    CropResize,
    Intrinsics,
    backproject,
    make_T,
    pixel_to_base,
    project,
    ray_ground_intersection,
    sample_depth,
    transform_points,
    yaw_of,
)

OUT_ROOT = Path(__file__).resolve().parent / "out"
TOPICS = {
    "color": ("/camera/camera/color/image_raw", Image),
    "depth": ("/camera/camera/aligned_depth_to_color/image_raw", Image),
    "info": ("/camera/camera/color/camera_info", CameraInfo),
    "odom": ("/gdq/msg/gdq_odom", Odometry),
    "lidar": ("/ouster/points", PointCloud2),
}


def tf_to_T(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return make_T([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


class Verify(Node):
    def __init__(self):
        super().__init__("vln_verify_sensors")
        self.check = self.declare_parameter("check", "rates").value
        ns = self.declare_parameter("camera_ns", "/camera/camera").value  # /camera_lookdown/camera for the 30° cam
        for k, sub in (("color", "color/image_raw"), ("depth", "aligned_depth_to_color/image_raw"),
                       ("info", "color/camera_info")):
            TOPICS[k] = (f"{ns}/{sub}", TOPICS[k][1])
        self.duration = self.declare_parameter("duration", 20.0).value
        self.ground_z = self.declare_parameter("ground_z_in_base", -0.55).value
        self.out = OUT_ROOT / f"sensors_{self.check}{ns.replace('/', '_')}_{time.strftime('%y%m%d_%H%M%S')}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.tf = Buffer(cache_time=Duration(seconds=30))
        self._tfl = TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self.t_start = time.time()
        self.info = None
        self.create_subscription(CameraInfo, TOPICS["info"][0], self.on_info, qos_profile_sensor_data)
        if self.check == "rates":
            self.counts = {k: 0 for k in TOPICS}
            for k, (topic, typ) in TOPICS.items():
                self.create_subscription(typ, topic, lambda m, k=k: self.counts.__setitem__(k, self.counts[k] + 1),
                                         qos_profile_sensor_data)
            self.create_timer(1.0, self.tick_rates)
        elif self.check == "geom":
            self.lidar = None
            self.create_subscription(PointCloud2, TOPICS["lidar"][0], self.on_lidar, qos_profile_sensor_data)
            self._sync_rgbd(self.on_geom)
        elif self.check == "twist":
            self.rows = []
            self.create_subscription(Odometry, TOPICS["odom"][0], self.on_odom, 50)
            self.create_timer(1.0, self.tick_twist)
        elif self.check == "sync":
            self.rows, self.pending = [], []
            self._sync_rgbd(self.on_sync)
            self.create_timer(0.2, self.tick_sync)
        else:
            assert False, f"unreachable check={self.check!r}"
        self.done = False
        self.get_logger().info(f"check={self.check} -> {self.out}")

    def _sync_rgbd(self, cb):
        sc = Subscriber(self, Image, TOPICS["color"][0], qos_profile=qos_profile_sensor_data)
        sd = Subscriber(self, Image, TOPICS["depth"][0], qos_profile=qos_profile_sensor_data)
        self._sync = ApproximateTimeSynchronizer([sc, sd], queue_size=10, slop=0.05)
        self._sync.registerCallback(cb)

    def on_info(self, m):
        self.info = m

    def finish(self, result):
        (self.out / "result.json").write_text(json.dumps(result, indent=1, default=float))
        print(json.dumps(result, indent=1, default=float), flush=True)
        self.done = True

    # ---------------------------------------------------------------- V0
    def tick_rates(self):
        el = time.time() - self.t_start
        if el < 10.0:
            return
        res = {f"{k}_hz_wall": self.counts[k] / el for k in TOPICS}
        for frame in ("base_link", "camera_color_optical_frame", "os_sensor"):
            try:
                self.tf.lookup_transform("map", frame, Time())
                res[f"tf_map<-{frame}"] = True
            except Exception:
                res[f"tf_map<-{frame}"] = False
        self.finish(res)

    # ---------------------------------------------------------------- V2 / V4 / V6
    def on_lidar(self, m):
        self.lidar = m

    def on_geom(self, color, depth):
        if self.done or self.info is None or self.lidar is None:
            return
        import cv2

        try:
            T_base_cam = tf_to_T(self.tf.lookup_transform("base_link", depth.header.frame_id, Time()))
            T_cam_lidar = tf_to_T(self.tf.lookup_transform(depth.header.frame_id, self.lidar.header.frame_id,
                                                           Time()))
        except Exception as e:
            self.get_logger().warn(f"tf not ready: {e}", throttle_duration_sec=2.0)
            return
        rgb = self.bridge.imgmsg_to_cv2(color, "rgb8")
        d = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        d = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)
        intr_raw = Intrinsics.from_K(self.info.k, self.info.width, self.info.height)
        cr = CropResize.center(self.info.width, self.info.height, 640, 480)
        intr = cr.apply_intrinsics(intr_raw)
        rgb_p, d_p = cr.apply_rgb(rgb), cr.apply_depth(d)

        pts = point_cloud2.read_points_numpy(self.lidar, field_names=("x", "y", "z"), skip_nans=True)
        pts = pts[np.linalg.norm(pts, axis=1) > 0.2]
        p_cam = transform_points(T_cam_lidar, pts)
        uv, front = project(p_cam, intr)
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        uv_in, z_in = uv[inside], p_cam[inside, 2]

        # V2 overlay (colour = lidar range)
        vis = rgb_p.copy()
        for (u, v), z in zip(uv_in.astype(int), z_in):
            c = int(np.clip(255 * z / 8.0, 0, 255))
            cv2.circle(vis, (int(u), int(v)), 1, (255 - c, c, 0), -1)
        cv2.imwrite(str(self.out / "overlay_lidar_on_rgb.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        dvis = cv2.applyColorMap(np.clip(d_p / 8.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(self.out / "depth_proc.png"), dvis)

        # V4 depth image vs lidar (same physx collision geometry, independent sensor)
        dimg = d_p[uv_in[:, 1].astype(int), uv_in[:, 0].astype(int)]
        ok = dimg > 0.1
        err = dimg[ok] - z_in[ok]
        res = {
            "camera_K_raw": [intr_raw.fx, intr_raw.fy, intr_raw.cx, intr_raw.cy],
            "camera_K_proc": [intr.fx, intr.fy, intr.cx, intr.cy],
            "V4_n_lidar_px": int(ok.sum()),
            "V4_depth_minus_lidar_median_m": float(np.median(err)) if ok.any() else None,
            "V4_abs_err_p50_m": float(np.median(np.abs(err))) if ok.any() else None,
            "V4_abs_err_p90_m": float(np.percentile(np.abs(err), 90)) if ok.any() else None,
        }

        # V6 grid pixels -> 3D (depth / ground) -> distance to nearest lidar point (base frame)
        from scipy.spatial import cKDTree

        T_base_lidar = T_base_cam @ T_cam_lidar
        tree = cKDTree(transform_points(T_base_lidar, pts))
        rows = []
        for v in range(250, 480, 20):  # lower part of the image (floor)
            for u in range(20, 640, 40):
                p_d, m_d = pixel_to_base(u, v, d_p, intr, T_base_cam, self.ground_z)
                p_g = ray_ground_intersection(u, v, intr, T_base_cam, self.ground_z)
                rng = float(np.hypot(*p_d[:2])) if p_d is not None else float("nan")
                e_d = float(tree.query(p_d)[0]) if (p_d is not None and m_d == "depth") else float("nan")
                e_g = float(tree.query(p_g)[0]) if p_g is not None else float("nan")
                rows.append((u, v, m_d, rng, e_d, e_g))
        import csv

        with open(self.out / "V6_grid.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["u", "v", "method", "range_m", "err_depth_m", "err_ground_m"])
            w.writerows(rows)
        arr = np.array([(r[3], r[4], r[5]) for r in rows], np.float64)
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 50)):
            sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            res[f"V6_{lo}-{hi}m_n"] = int(sel.sum())
            res[f"V6_{lo}-{hi}m_depth_err_median"] = float(np.nanmedian(arr[sel, 1])) if sel.any() else None
            res[f"V6_{lo}-{hi}m_ground_err_median"] = float(np.nanmedian(arr[sel, 2])) if sel.any() else None
        res["V6_depth_fallback_ratio"] = float(np.mean([r[2] != "depth" for r in rows]))
        self.finish(res)

    # ---------------------------------------------------------------- V7a
    def on_odom(self, m):
        p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.rows.append((t, p.x, p.y, yaw, tw.linear.x, tw.linear.y, tw.angular.z))

    def tick_twist(self):
        if time.time() - self.t_start < self.duration:
            return
        a = np.array(self.rows)
        if len(a) < 20:
            self.finish({"error": "too few odom messages"})
            return
        k = 10  # difference over 10 messages (0.2 s at 50 Hz)
        dt = a[k:, 0] - a[:-k, 0]
        vw = np.column_stack([(a[k:, 1] - a[:-k, 1]) / dt, (a[k:, 2] - a[:-k, 2]) / dt])  # world, pose diff
        yaw = a[:-k, 3] + 0.5 * np.angle(np.exp(1j * (a[k:, 3] - a[:-k, 3])))
        vb = np.column_stack([np.cos(yaw) * vw[:, 0] + np.sin(yaw) * vw[:, 1],
                              -np.sin(yaw) * vw[:, 0] + np.cos(yaw) * vw[:, 1]])  # body, pose diff
        tw = 0.5 * (a[k:, 4:6] + a[:-k, 4:6])
        moving = np.linalg.norm(vw, axis=1) > 0.1
        res = {"n": int(moving.sum())}
        if moving.any():
            res["rmse_twist_vs_world_posediff"] = float(np.sqrt(np.mean(np.sum((tw - vw)[moving] ** 2, axis=1))))
            res["rmse_twist_vs_body_posediff"] = float(np.sqrt(np.mean(np.sum((tw - vb)[moving] ** 2, axis=1))))
            res["verdict"] = ("world" if res["rmse_twist_vs_world_posediff"] < res["rmse_twist_vs_body_posediff"]
                              else "body")
            # V7b: motion direction vs heading (body-frame cmd => moves along heading when vy=0)
            heading_err = np.degrees(np.arctan2(vb[moving, 1], vb[moving, 0]))
            res["V7b_motion_dir_minus_heading_median_deg"] = float(np.median(heading_err))
            res["yaw_range_deg"] = float(np.degrees(np.ptp(a[:, 3])))
        np.savetxt(self.out / "odom.csv", a, delimiter=",", header="t,x,y,yaw,tw_x,tw_y,tw_wz", comments="")
        self.finish(res)

    # ---------------------------------------------------------------- V8
    def on_sync(self, color, depth):
        # non-blocking: just record stamps; tick_sync checks them once TF has had time to arrive
        st = color.header.stamp
        cd = (st.sec + st.nanosec * 1e-9) - (depth.header.stamp.sec + depth.header.stamp.nanosec * 1e-9)
        self.pending.append((st, cd, time.time()))

    def tick_sync(self):
        now = time.time()
        keep = []
        for st, cd, tw in self.pending:
            if now - tw < 0.5:  # give TF 0.5 s (wall) to catch up, as the bridge's 0.2 s lookup timeout + queueing
                keep.append((st, cd, tw))
                continue
            ok = self.tf.can_transform("map", "base_link", Time.from_msg(st))
            latest = self.tf.get_latest_common_time("map", "base_link")
            self.rows.append((ok, latest.nanoseconds * 1e-9 - (st.sec + st.nanosec * 1e-9), cd))
        self.pending = keep
        if now - self.t_start < self.duration:
            return
        a = np.array(self.rows, dtype=np.float64) if self.rows else np.zeros((0, 3))
        self.finish({
            "n_frames": len(a),
            "tf_lookup_success": float(a[:, 0].mean()) if len(a) else None,
            "tf_latest_minus_image_stamp_median_s": float(np.nanmedian(a[:, 1])) if len(a) else None,
            "color_minus_depth_stamp_abs_max_s": float(np.abs(a[:, 2]).max()) if len(a) else None,
        })


def main():
    rclpy.init()
    n = Verify()
    while rclpy.ok() and not n.done:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
