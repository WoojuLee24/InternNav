"""③ pixel goal 3D check on the live sim: bridge's 3D goal vs lidar points ON THE SAME RAY.

For every S2 answer (/vln/s2 JSON: pixel, pixel_frame, pixel_stamp, goal_xyz in the bridge world frame):
  1. lidar cloud nearest to pixel_stamp -> world (TF at the cloud stamp) -> camera of pixel_frame (TF at pixel_stamp),
     so robot motion between the two stamps is compensated
  2. lidar points projected within --radius px of the goal pixel (processed 640x480 image, same crop/K as the bridge).
     A plane has 1/depth linear in (u, v): fit 1/z = a u + b v + c to them (2 rounds dropping residual outliers, e.g.
     a wall behind the floor) and read the lidar depth AT the goal pixel. (Taking the nearest ring instead biased the
     lidar depth short by up to 0.5 m on the floor at 4 m: rings 4-5 px apart span ~1 m there.)
  3. bridge goal -> same camera: err_z = bridge depth - lidar depth (along the optical axis), reprojection px error
No lidar point near the pixel -> "no_lidar" (counted, not judged). One png per sample: S2 image (/vln/s2_image) with
the lidar points (colour = depth) and the goal pixel / used points.
Rows go to <out>/proj.csv as they come (stop with Ctrl-C / kill).
Usage: python3 scripts/sim/verify/proj_check.py --ros-args -p use_sim_time:=true -p out:=DIR
"""
import collections
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projection import CropResize, Intrinsics, project, transform_points  # noqa: E402
from frame_check import tf_to_T  # noqa: E402

CAMS = {"fpv": "/camera/camera", "lookdown": "/camera_lookdown/camera"}


def t_of(st):
    return st.sec + st.nanosec * 1e-9


def to_time(t):
    return Time(seconds=int(t), nanoseconds=int(round((t - int(t)) * 1e9)))


class ProjCheck(Node):
    def __init__(self):
        super().__init__("proj_check")
        self.out = Path(self.declare_parameter("out", "scripts/sim/verify/out/proj").value)
        self.world = self.declare_parameter("world_frame", "odom").value
        self.radius = self.declare_parameter("radius", 15.0).value
        self.out.mkdir(parents=True, exist_ok=True)
        self.buf = Buffer(cache_time=Duration(seconds=30))
        TransformListener(self.buf, self)
        self.bridge = CvBridge()
        self.cam = {}  # name -> (frame_id, Intrinsics processed)
        for name, ns in CAMS.items():
            self.create_subscription(CameraInfo, f"{ns}/color/camera_info", lambda m, n=name: self.on_info(n, m),
                                     qos_profile_sensor_data)
        self.clouds = collections.deque(maxlen=40)
        self.create_subscription(PointCloud2, "/ouster/points", lambda m: self.clouds.append(m), qos_profile_sensor_data)
        self.pending = collections.deque(maxlen=20)  # S2 JSON waiting for TF / image
        self.imgs = {}
        self.create_subscription(String, "/vln/s2", lambda m: self.pending.append(json.loads(m.data)), 10)
        self.create_subscription(Image, "/vln/s2_image", self.on_img, 5)
        self.f = open(self.out / "proj.csv", "a", newline="")
        self.w = csv.writer(self.f)
        if self.f.tell() == 0:
            self.w.writerow(["stamp", "pixel_frame", "row", "col", "proj", "range_m", "status", "n_lidar",
                             "z_bridge", "z_lidar", "err_z_m", "reproj_px", "cloud_dt_s", "png"])
        self.create_timer(0.2, self.process)

    def on_info(self, name, m):
        if name not in self.cam:
            cr = CropResize.center(m.width, m.height, 640, 480)
            self.cam[name] = (m.header.frame_id, cr.apply_intrinsics(Intrinsics.from_K(m.k, m.width, m.height)))

    def on_img(self, m):
        self.imgs[round(t_of(m.header.stamp), 3)] = self.bridge.imgmsg_to_cv2(m, "rgb8")
        while len(self.imgs) > 20:
            self.imgs.pop(min(self.imgs))

    def T(self, target, source, t):
        return tf_to_T(self.buf.lookup_transform(target, source, to_time(t), timeout=Duration(seconds=0.0)))

    def process(self):
        for _ in range(len(self.pending)):
            j = self.pending.popleft()
            if j.get("pixel") is None or j.get("goal_xyz") is None or j.get("pixel_stamp") is None:
                continue
            if j["pixel_frame"] not in self.cam or not self.clouds:
                self.pending.append(j)
                continue
            try:
                self.check(j)
            except Exception as e:  # TF not there yet -> retry later (dropped by maxlen if it never comes)
                if "extrapolat" in str(e) or "lookup" in str(e).lower():
                    self.pending.append(j)
                else:
                    self.get_logger().warn(f"check failed: {e}")

    def check(self, j):
        row, col = j["pixel"]
        frame, intr = self.cam[j["pixel_frame"]]
        tp = j["pixel_stamp"]
        cloud = min(self.clouds, key=lambda c: abs(t_of(c.header.stamp) - tp))
        tc = t_of(cloud.header.stamp)
        T_cam_world = self.T(frame, self.world, tp)
        T_world_lidar = self.T(self.world, cloud.header.frame_id, tc)
        pts = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"), skip_nans=True)
        pts = pts[np.linalg.norm(pts, axis=1) > 0.2]
        p_cam = transform_points(T_cam_world @ T_world_lidar, pts)
        uv, front = project(p_cam, intr)
        near = front & (np.hypot(uv[:, 0] - col, uv[:, 1] - row) <= self.radius)
        p_b = transform_points(T_cam_world, np.array([j["goal_xyz"]]))[0]
        uv_b, _ = project(p_b[None], intr)
        reproj = float(np.hypot(uv_b[0, 0] - col, uv_b[0, 1] - row))
        rng = float(np.linalg.norm(p_b))
        surf, z_l, status, err = None, float("nan"), "no_lidar", float("nan")
        if near.sum() >= 3:
            A = np.column_stack([uv[near, 0], uv[near, 1], np.ones(near.sum())])
            iz = 1.0 / p_cam[near, 2]
            surf = np.ones(len(iz), bool)
            for _ in range(2):
                coef = np.linalg.lstsq(A[surf], iz[surf], rcond=None)[0]
                res = np.abs(A @ coef - iz)
                surf = res <= max(3 * np.median(res[surf]), 0.005)
            v_span = np.ptp(uv[near][surf][:, 1]) if surf.sum() >= 3 else 0.0
            iz0 = float(coef @ [col, row, 1.0])
            if surf.sum() >= 3 and v_span >= 2.0 and iz0 > 0:  # >= 2 rings, else the plane is undetermined in v
                z_l = 1.0 / iz0
                status, err = "ok", float(p_b[2] - z_l)
            else:
                status = "few_lidar"
        png = ""
        img = self.imgs.get(round(j["stamp"], 3))
        if img is not None:
            vis = img.copy()
            inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
            for (u, v), zz in zip(uv[inside].astype(int), p_cam[inside, 2]):
                c = int(np.clip(255 * zz / 8.0, 0, 255))
                cv2.circle(vis, (int(u), int(v)), 1, (255 - c, c, 0), -1)
            if surf is not None:
                for u, v in uv[near][surf].astype(int):
                    cv2.circle(vis, (int(u), int(v)), 3, (255, 0, 255), 1)
            cv2.circle(vis, (int(col), int(row)), int(self.radius), (255, 255, 255), 1)
            cv2.putText(vis, f"{j['pixel_frame']} {j['proj']} zb={p_b[2]:.2f} zl={z_l:.2f} err={err:+.2f}m",
                        (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            png = f"proj_{tp:.3f}.png"
            cv2.imwrite(str(self.out / png), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.w.writerow([f"{tp:.3f}", j["pixel_frame"], row, col, j["proj"], f"{rng:.3f}", status, int(near.sum()),
                         f"{p_b[2]:.3f}", f"{z_l:.3f}", f"{err:.3f}", f"{reproj:.2f}", f"{tc - tp:.3f}", png])
        self.f.flush()
        self.get_logger().info(f"{j['pixel_frame']} ({row},{col}) {j['proj']} range {rng:.2f} m: {status} "
                               f"err_z {err:+.3f} m reproj {reproj:.1f}px n={int(near.sum())}")


def main():
    rclpy.init()
    n = ProjCheck()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):
        pass
    n.f.close()


if __name__ == "__main__":
    main()
