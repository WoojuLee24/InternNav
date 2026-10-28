"""Minimal stand-in for sim_docker to test the bridge/server plumbing without Isaac Sim.

Publishes the same topics/frames as run_rbq10_sim.py (1280x720 rgb8 + 16UC1 mm aligned depth +
camera_info with the sim intrinsics, /gdq/msg/gdq_odom, /tf map->odom->base_link, /tf_static
base_link->...->camera_color_optical_frame). Images come from a VLN-CE episode (640x480, padded to
16:9 so the bridge's centre crop recovers the original frame), advanced as the robot moves.
Integrates a unicycle from /cmd_vel (priority) or /gdq/msg/cmd_vel, logs /gdg/data/gp.

Usage: python3 scripts/sim/verify/fake_sim.py --ros-args -p episode_dir:=data/.../17DRP5sb8fy -p episode:=0
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

FX = 1.93 * 1280 / 3.896  # sim D455 (run_rbq10_sim.py CAMERA_FOCAL / CAMERA_HORIZ_AP)
CAM_CHAIN = [  # sensor_configuration.yaml + optical
    ("base_link", "os_sensor", (0.3, 0.0, 0.05), (0, 0, 0, 1)),
    ("os_sensor", "camera_link", (0.065151, 0.002019, -0.015822), (-0.000391, 0.128428, 0.001329, 0.991718)),
    ("camera_link", "camera_color_frame", (0, 0, 0), (0, 0, 0, 1)),
    ("camera_color_frame", "camera_color_optical_frame", (0, 0, 0), (-0.5, 0.5, -0.5, 0.5)),
]


def tfmsg(parent, child, xyz, q, stamp):
    t = TransformStamped()
    t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, xyz)
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
    return t


class FakeSim(Node):
    def __init__(self):
        super().__init__("fake_sim")
        root = Path(self.declare_parameter("episode_dir", "").value)
        ep = self.declare_parameter("episode", 0).value
        # map->odom offset (x, y, yaw deg). 0 = like run_rbq10_sim.py; non-zero mimics a real robot whose
        # localization has corrected odom drift (frame-consistency test, frame_test.sh)
        self.map_odom = [float(v) for v in self.declare_parameter("map_odom", [0.0, 0.0, 0.0]).value]
        self.rgb_files = sorted((root / "videos/chunk-000/observation.images.rgb.125cm_0deg").glob(f"episode_{ep:06d}_*.jpg"),
                                key=lambda p: int(p.stem.split("_")[-1]))
        self.dep_dir = root / "videos/chunk-000/observation.images.depth.125cm_0deg"
        # optional 30° look-down camera (like sim GUIDEDOG_LOOKDOWN_CAM=1): dataset 125cm_30deg frames on
        # /camera_lookdown/camera/*, same intrinsics, camera pitched 30° down
        self.lookdown = self.declare_parameter("lookdown", False).value
        self.ld_dir = root / "videos/chunk-000/observation.images.rgb.125cm_30deg"
        self.ld_dep_dir = root / "videos/chunk-000/observation.images.depth.125cm_30deg"
        assert self.rgb_files, f"no frames under {root}"
        self.instruction = None
        for line in open(root / "meta/episodes.jsonl"):
            e = json.loads(line)
            if e["episode_index"] == ep:
                self.instruction = e["tasks"][0]
        self.get_logger().info(f"{len(self.rgb_files)} frames, instruction: {self.instruction!r}")
        self.bridge = CvBridge()
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0
        self.last_primary = 0.0
        self.dist = 0.0
        self.n_gp = 0
        self.pub_c = self.create_publisher(Image, "/camera/camera/color/image_raw", 2)
        self.pub_d = self.create_publisher(Image, "/camera/camera/aligned_depth_to_color/image_raw", 2)
        self.pub_i = self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", 2)
        self.pub_o = self.create_publisher(Odometry, "/gdq/msg/gdq_odom", 10)
        self.tfb = TransformBroadcaster(self)
        st = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()
        chain = list(CAM_CHAIN)
        if self.lookdown:
            p30 = math.radians(30.0)  # pitch down about y, then optical (z fwd, x right, y down)
            chain += [("base_link", "camera_lookdown_link", (0.365, 0.002, 0.034), (0.0, math.sin(p30 / 2), 0.0, math.cos(p30 / 2))),
                      ("camera_lookdown_link", "camera_lookdown_color_optical_frame", (0, 0, 0), (-0.5, 0.5, -0.5, 0.5))]
            self.pub_ld = [self.create_publisher(Image, "/camera_lookdown/camera/color/image_raw", 2),
                           self.create_publisher(Image, "/camera_lookdown/camera/aligned_depth_to_color/image_raw", 2),
                           self.create_publisher(CameraInfo, "/camera_lookdown/camera/color/camera_info", 2)]
        st.sendTransform([tfmsg(p, c, xyz, q, now) for p, c, xyz, q in chain])
        self.create_subscription(Twist, "/cmd_vel", lambda m: self.on_cmd(m, True), 10)
        self.create_subscription(Twist, "/gdq/msg/cmd_vel", lambda m: self.on_cmd(m, False), 10)
        try:
            from gd_ifc_pkg.msg import GDGGuidancePoint

            self.create_subscription(GDGGuidancePoint, "/gdg/data/gp", self.on_gp, 10)
        except ImportError:
            pass
        self.t_last = time.time()
        self.last_cmd = 0.0
        self.create_timer(0.02, self.tick)  # 50 Hz odom/tf
        self.create_timer(1 / 15, self.tick_cam)  # 15 Hz like RENDER_HZ

    def on_cmd(self, m, primary):
        now = time.time()
        if primary:
            self.last_primary = now
        elif now - self.last_primary < 0.5:
            return
        self.v, self.w, self.last_cmd = m.linear.x, m.angular.z, now

    def on_gp(self, m):
        self.n_gp += 1
        if m.gp_type and m.gp_type[0] == 99:
            self.get_logger().info("GP: FORCE_TO_STOP")
        elif m.position:
            self.get_logger().info(f"GP #{self.n_gp}: type={m.gp_type[0]} rel=({m.position[0].x:.2f}, {m.position[0].y:.2f})")

    def tick(self):
        now = time.time()
        dt, self.t_last = now - self.t_last, now
        if now - self.last_cmd > 0.5:  # CMD_VEL_TIMEOUT_S
            self.v = self.w = 0.0
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.w * dt
        self.dist += abs(self.v) * dt
        st = self.get_clock().now().to_msg()
        q = (0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2))
        mx, my, myaw = self.map_odom[0], self.map_odom[1], math.radians(self.map_odom[2])
        q_mo = (0.0, 0.0, math.sin(myaw / 2), math.cos(myaw / 2))
        self.tfb.sendTransform([tfmsg("map", "odom", (mx, my, 0), q_mo, st),
                                tfmsg("odom", "base_link", (self.x, self.y, 0.55), q, st)])
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = st, "odom", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = self.x, self.y, 0.55
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = q[2], q[3]
        o.twist.twist.linear.x = self.v * math.cos(self.yaw)  # world frame, like the sim
        o.twist.twist.linear.y = self.v * math.sin(self.yaw)
        o.twist.twist.angular.z = self.w
        self.pub_o.publish(o)

    def tick_cam(self):
        i = min(int(self.dist / 0.25), len(self.rgb_files) - 1)  # dataset steps are 0.25 m
        st = self.get_clock().now().to_msg()
        self.publish_cam(self.rgb_files[i], self.dep_dir, (self.pub_c, self.pub_d, self.pub_i),
                         "camera_color_optical_frame", st)
        if self.lookdown:
            self.publish_cam(self.ld_dir / self.rgb_files[i].name, self.ld_dep_dir, self.pub_ld,
                             "camera_lookdown_color_optical_frame", st)

    def publish_cam(self, rgb_file, dep_dir, pubs, frame, st):
        pub_c, pub_d, pub_i = pubs
        rgb = np.asarray(PILImage.open(rgb_file).convert("RGB").resize((960, 720)))
        dep = np.asarray(PILImage.open(dep_dir / rgb_file.name.replace(".jpg", ".png")))
        dep = np.asarray(PILImage.fromarray(dep.astype(np.int32)).resize((960, 720), PILImage.NEAREST)).astype(np.uint16)
        rgb_f = np.zeros((720, 1280, 3), np.uint8)
        dep_f = np.zeros((720, 1280), np.uint16)
        rgb_f[:, 160:1120], dep_f[:, 160:1120] = rgb, dep
        for pub, img, enc in ((pub_c, rgb_f, "rgb8"), (pub_d, dep_f, "16UC1")):
            m = self.bridge.cv2_to_imgmsg(img, enc)
            m.header.stamp, m.header.frame_id = st, frame
            pub.publish(m)
        ci = CameraInfo()
        ci.header.stamp, ci.header.frame_id = st, frame
        ci.width, ci.height, ci.distortion_model = 1280, 720, "plumb_bob"
        ci.k = [FX, 0.0, 640.0, 0.0, FX, 360.0, 0.0, 0.0, 1.0]
        ci.p = [FX, 0.0, 640.0, 0.0, 0.0, FX, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        pub_i.publish(ci)


def main():
    rclpy.init()
    rclpy.spin(FakeSim())


if __name__ == "__main__":
    main()
