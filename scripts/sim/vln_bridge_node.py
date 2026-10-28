"""ROS 2 bridge: Isaac Sim RGB-D -> InternVLA-N1 server -> gdm goal (Mode A) or cmd_vel (Mode B).

Derived from scripts/realworld/http_internvla_client.py (kept untouched). Differences:
  * images/odom matched by header.stamp (sim time), not wall-clock time.time(): odom message pose
    interpolated at the image stamp (pose_source:=odom, default, same frame as the realworld client and
    gdm /gdg/data/gp) or tf2 map<-base_link (pose_source:=tf). "map" in names below = that world frame
  * 1280x720 sim frame center-cropped to 4:3 and resized to 640x480 (training aspect),
    intrinsics transformed accordingly (projection.CropResize)
  * mode:=pixel_goal  — S2 pixel goal -> depth / ground-ray projection -> map goal -> gdm
                        (GDGGuidancePoint VIA_GP on /gdg/data/gp, or PoseStamped /goal_pose)
    mode:=trajectory  — S1 waypoints -> map -> MPC -> /gdq/msg/cmd_vel (gdm planner NOT running)
  * discrete actions -> PID; STOP -> FORCE_TO_STOP GP (Mode A) + zero cmd

Odometry twist is NOT used anywhere: sim /gdq/msg/gdq_odom twist is world-frame
(run_rbq10_sim.py publish_odom) and the real robot publishes zeros. Controllers take pose only.

Run (inside the gd_vln container, ROS_DOMAIN_ID=42):
  python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true \
      -p mode:=pixel_goal -p instruction:="walk to the door and stop"
"""
import collections
import csv
import io
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
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry, Path as PathMsg
from PIL import Image as PILImage
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "realworld"))
from control_utils import discrete_step  # noqa: E402
from controllers import Mpc_controller, PID_controller  # noqa: E402
from projection import (  # noqa: E402
    CropResize,
    Intrinsics,
    base_traj_to_pixels,
    estimate_ground_z,
    make_T,
    pixel_to_base,
    rescale_pixel,
    se2,
    transform_points,
    yaw_of,
)

ACT_STOP, ACT_FWD, ACT_LEFT, ACT_RIGHT, ACT_LOOKDOWN = 0, 1, 2, 3, 5

# control states
IDLE, MPC, PID, STOPPED = "idle", "mpc", "pid", "stopped"


def tf_to_T(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return make_T([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def interp_T(T0, T1, a):
    """Pose between T0 (a=0) and T1 (a=1): linear translation, slerp rotation."""
    from scipy.spatial.transform import Rotation, Slerp

    T = np.eye(4)
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    T[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))(a).as_matrix()
    return T


class VLNBridge(Node):
    def __init__(self):
        super().__init__("vln_bridge")
        p = self.declare_parameter
        p("mode", "pixel_goal")  # pixel_goal | trajectory
        p("instruction", "")
        p("server_url", "http://127.0.0.1:5801")
        p("dry_run", False)  # compute + visualise only, publish no goal / cmd_vel
        p("color_topic", "/camera/camera/color/image_raw")
        p("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        p("info_topic", "/camera/camera/color/camera_info")
        p("odom_topic", "/gdq/msg/gdq_odom")
        # 30° look-down camera namespace (sim GUIDEDOG_LOOKDOWN_CAM=1 -> "/camera_lookdown/camera"). Empty = off:
        # the server then re-queries "↓" with the same forward frame (realworld behaviour).
        p("lookdown_ns", "")
        # Robot pose source = the frame every goal / path / controller works in ("map" in variable names).
        # "odom": /gdq/msg/gdq_odom pose, nearest in time to the image (realworld client behaviour); gdm's
        #         /gdg/data/gp is relative to that same odom pose, so goals stay consistent when map != odom.
        # "tf":   TF map_frame <- base_frame (legacy; equal to "odom" in sim where map->odom is identity).
        p("pose_source", "odom")
        p("odom_frame", "odom")  # frame_id for published markers/paths (real robot odom header.frame_id is empty)
        p("odom_max_dt", 0.1)  # image <-> nearest odom stamp gap above which the pose counts as not synced
        p("map_frame", "map")
        p("base_frame", "base_link")
        p("proc_width", 640)
        p("proc_height", 480)
        p("ground_z_in_base", -0.55)  # fallback; sensor_configuration.yaml base_height
        # estimate the floor height from the bottom-centre depth each frame (EMA). Measured 2026-10-05 in
        # ETRI indoor sim: floor at -0.476 m in base_link (not -0.55) -> fixed value biased the ground ray ~0.2 m
        p("ground_z_auto", True)
        p("depth_window", 4)
        # S2 pixel goal coordinate space. "input": pixels of the frame sent to the server (640x480) —
        # verified: bf16 S2 outputs reach row 455 / col 491 (> 384) on VLN-CE replay (verify_quant), and the
        # VLN-CE labels (goal.125cm_30deg) are in the 640x480 frame. "s2": the 384x384 S2 resize (legacy guess).
        p("pixel_space", "input")
        p("goal_interface", "gp")  # gp (/gdg/data/gp, VIA_GP) | goal_pose (/goal_pose)
        p("goal_min_change", 0.3)
        # same goal is re-sent after this many seconds: a lost first message (publisher discovery) plus the
        # min-change dedup left gdm without any goal for a whole episode (2026-10-05 e3_right, Mode A)
        p("goal_resend_s", 5.0)
        # cap the goal distance along its direction: depth through holes of the reconstructed mesh produced
        # goals 48-49 m away (2026-10-05 e4_turnaround, Mode A)
        p("goal_max_dist", 6.0)
        p("cmd_topic", "/gdq/msg/cmd_vel")
        p("discrete_cmd_topic_mode_a", "/cmd_vel")  # overrides gdm planner while PID turns
        p("v_max", 0.4)
        p("w_max", 0.4)
        p("pid_kd_trans", 0.0)
        p("pid_kd_yaw", 0.0)
        p("pid_v_min", 0.3)  # walking-policy dead zone (gdm: rapp_vx_min_speed 0.3)
        p("pid_w_min", 0.4)  # (gdm: align_w_min 0.4)
        p("pid_lead_s", 0.45)  # coast after a zero command; stop early by |cmd| * lead (verify_pid.py)
        p("plan_period", 0.3)
        # Do not query the model while a discrete action (PID) is executing: the next answer used to cut 60 deg
        # turns off after ~2 s and the model's history assumed the turn had completed (2026-10-05 e4_turnaround).
        # Same semantics as habitat/VLN-CE: act, then observe. Upper bound in seconds.
        p("discrete_block_max_s", 12.0)
        # watchdog for a hung/dead server only. In the live ETRI sim an S2 call (↓ + look-down re-query) takes
        # 2.03 s p50 / 2.06 s p95 (2026-10-05), so 1.0 s stopped the robot on every S2 call (stop-and-go).
        p("server_timeout", 3.0)
        p("traj_skip", 3)  # drop first waypoints (realworld client does the same)
        p("log_dir", "")
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.mode = g("mode")
        assert self.mode in ("pixel_goal", "trajectory"), self.mode
        self.instruction = g("instruction")
        self.server_url = g("server_url").rstrip("/")
        self.dry_run = g("dry_run")
        self.map_frame, self.base_frame = g("map_frame"), g("base_frame")
        self.pose_source, self.odom_max_dt = g("pose_source"), g("odom_max_dt")
        if self.pose_source == "odom":
            self.world_frame = g("odom_frame")
        elif self.pose_source == "tf":
            self.world_frame = self.map_frame
        else:
            assert False, f"unreachable pose_source={self.pose_source!r}"
        self.proc_wh = (g("proc_width"), g("proc_height"))
        self.ground_z = g("ground_z_in_base")
        self.ground_z_auto = g("ground_z_auto")
        self.depth_window = g("depth_window")
        self.pixel_space = g("pixel_space")
        assert self.pixel_space in ("input", "s2"), self.pixel_space
        self.goal_interface = g("goal_interface")
        assert self.goal_interface in ("gp", "goal_pose"), self.goal_interface
        self.goal_min_change = g("goal_min_change")
        self.goal_resend_s, self.goal_max_dist = g("goal_resend_s"), g("goal_max_dist")
        self.last_goal_time = 0.0
        self.v_max, self.w_max = g("v_max"), g("w_max")
        self.plan_period = g("plan_period")
        self.discrete_block_max_s = g("discrete_block_max_s")
        self.pid_started = 0.0
        self.server_timeout = g("server_timeout")
        self.traj_skip = g("traj_skip")

        kd_t, kd_y = g("pid_kd_trans"), g("pid_kd_yaw")
        if kd_t > 0 or kd_y > 0:
            # PID derivative terms need a measured velocity; this node passes 0 because odom twist
            # is world-frame in sim and all-zero on the real robot. Wire a body-frame velocity first.
            self.get_logger().warn(
                f"pid_kd_trans={kd_t} pid_kd_yaw={kd_y} > 0 but velocity feedback is 0 "
                "(odom twist unused: world-frame in sim, zero on robot)"
            )
        self.pid = PID_controller(Kp_trans=2.0, Kd_trans=kd_t, Kp_yaw=1.5, Kd_yaw=kd_y,
                                  max_v=self.v_max, max_w=self.w_max)
        self.mpc = None
        self.pid_v_min, self.pid_w_min, self.pid_lead_s = g("pid_v_min"), g("pid_w_min"), g("pid_lead_s")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lock = threading.Lock()
        self.latest = None  # (stamp, rgb, depth_m)
        self.new_frame = False
        self.intr_raw = None
        self.crop = None
        self.intr = None
        self.T_base_cam = None
        self.odom = None
        self.odom_buf = collections.deque(maxlen=200)  # (stamp_s, 4x4 odom<-base)
        self.pose_dt = 0.0  # odom stamp - image stamp of the last world_T_base(stamp) lookup
        self.need_reset = True
        self.state = IDLE
        self.pid_goal = None  # 4x4 map pose
        self.last_reply_time = None
        self.request_in_flight_since = None
        self.last_goal_map = None
        self.episode_stopped = False
        # recent frames by request id: an async server (dual_server --mode async) answers with a pixel goal that
        # belongs to an OLDER frame (reply pixel_frame_id); it is projected with that frame's depth / pose.
        self.frame_id = 0
        self.frame_cache = collections.OrderedDict()

        self.create_subscription(CameraInfo, g("info_topic"), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 10)
        self.create_subscription(String, "/vln/instruction", self.on_instruction, 10)
        sub_c = Subscriber(self, Image, g("color_topic"), qos_profile=qos_profile_sensor_data)
        sub_d = Subscriber(self, Image, g("depth_topic"), qos_profile=qos_profile_sensor_data)
        self.lookdown_ns = g("lookdown_ns").rstrip("/")
        self.intr_ld = None
        self.T_base_cam_ld = None
        if self.lookdown_ns:
            self.create_subscription(CameraInfo, f"{self.lookdown_ns}/color/camera_info", self.on_info_ld,
                                     qos_profile_sensor_data)
            subs = [sub_c, sub_d,
                    Subscriber(self, Image, f"{self.lookdown_ns}/color/image_raw", qos_profile=qos_profile_sensor_data),
                    Subscriber(self, Image, f"{self.lookdown_ns}/aligned_depth_to_color/image_raw",
                               qos_profile=qos_profile_sensor_data)]
            self.sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.08)
        else:
            self.sync = ApproximateTimeSynchronizer([sub_c, sub_d], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_rgbd)

        # One velocity publisher per mode, so the gdm planner stays the only publisher on /gdq/msg/cmd_vel in
        # Mode A (V12; 2026-10-05 live run showed 2 publishers when this node also created it in Mode A).
        if self.mode == "pixel_goal":
            self.cmd_pub = self.create_publisher(Twist, g("discrete_cmd_topic_mode_a"), 5)  # /cmd_vel, overrides gdm
        elif self.mode == "trajectory":
            self.cmd_pub = self.create_publisher(Twist, g("cmd_topic"), 5)  # /gdq/msg/cmd_vel, gdm planner not running
        else:
            assert False, f"unreachable mode={self.mode!r}"
        self.discrete_pub = self.cmd_pub
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 5)
        self.gp_pub = None
        if self.mode == "pixel_goal" and self.goal_interface == "gp":
            try:
                from gd_ifc_pkg.msg import GDGGuidancePoint

                self.GP = GDGGuidancePoint
                self.gp_pub = self.create_publisher(GDGGuidancePoint, "/gdg/data/gp", 10)
            except ImportError:
                self.get_logger().error("gd_ifc_pkg not found -> falling back to /goal_pose (no FORCE_TO_STOP)")
                self.goal_interface = "goal_pose"
        self.vis_img_pub = self.create_publisher(Image, "/vln/pixel_goal_image", 2)  # every reply (S1 + S2)
        self.s2_img_pub = self.create_publisher(Image, "/vln/s2_image", 2)  # S2 answers only (decision frame)
        self.goal_marker_pub = self.create_publisher(Marker, "/vln/goal_marker", 5)
        self.traj_pub = self.create_publisher(PathMsg, "/vln/trajectory", 5)
        self.agree_pub = self.create_publisher(Float32MultiArray, "/vln/s1s2_agreement", 5)
        self.status_pub = self.create_publisher(String, "/vln/status", 5)
        self.s2_pub = self.create_publisher(String, "/vln/s2", 10)  # JSON per S2 answer (web_viz.py)

        self.log = None
        self.log_dir = g("log_dir")
        if g("log_dir"):
            os.makedirs(g("log_dir"), exist_ok=True)
            fn = os.path.join(g("log_dir"), time.strftime("vln_%Y%m%d_%H%M%S.csv"))
            self.log_f = open(fn, "w", newline="")
            self.log = csv.writer(self.log_f)
            self.log.writerow(["wall", "stamp", "tf_ok", "tf_dt", "kind", "action", "pixel_r", "pixel_c",
                               "proj_method", "goal_x", "goal_y", "dpx_384", "dpx_392", "dpx_proc", "t_infer", "llm",
                               "pixel_frame", "traj_end_x", "traj_end_y", "traj_start_x", "traj_start_y",
                               "traj_head_deg"])
            self.get_logger().info(f"logging to {fn}")

        threading.Thread(target=self.planning_loop, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        self.get_logger().info(
            f"vln_bridge mode={self.mode} dry_run={self.dry_run} goal_interface={self.goal_interface} "
            f"server={self.server_url} instruction={self.instruction!r}"
        )

    # ------------------------------------------------------------------ callbacks
    def on_info(self, msg):
        if self.intr_raw is not None:
            return
        self.intr_raw = Intrinsics.from_K(msg.k, msg.width, msg.height)
        self.crop = CropResize.center(msg.width, msg.height, *self.proc_wh)
        self.intr = self.crop.apply_intrinsics(self.intr_raw)
        self.get_logger().info(
            f"camera {msg.width}x{msg.height} fx={self.intr_raw.fx:.1f} cx={self.intr_raw.cx:.1f} -> crop "
            f"{self.crop.crop_w}x{self.crop.crop_h}@x0={self.crop.x0} -> {self.proc_wh} fx'={self.intr.fx:.1f}"
        )

    def on_info_ld(self, msg):
        if self.intr_ld is not None:
            return
        self.crop_ld = CropResize.center(msg.width, msg.height, *self.proc_wh)
        self.intr_ld = self.crop_ld.apply_intrinsics(Intrinsics.from_K(msg.k, msg.width, msg.height))
        self.get_logger().info(f"lookdown camera {msg.width}x{msg.height} -> fx'={self.intr_ld.fx:.1f}")

    def on_odom(self, msg):
        self.odom = msg
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if st == 0.0:  # unstamped odom -> receive time
            now = self.get_clock().now().to_msg()
            st = now.sec + now.nanosec * 1e-9
        o = msg.pose.pose
        T = make_T([o.position.x, o.position.y, o.position.z],
                   [o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w])
        with self.lock:
            self.odom_buf.append((st, T))

    def on_instruction(self, msg):
        if not msg.data.strip():  # empty instruction = stop (web_viz STOP button)
            with self.lock:
                self.instruction = ""
                self.state = STOPPED
            self.send_stop()
            self.publish_cmd(self.cmd_pub, 0.0, 0.0)
            self.status("STOP (operator)")
            self.get_logger().info("empty instruction -> stopped by operator")
            return
        with self.lock:
            self.instruction = msg.data
            self.need_reset = True
            self.episode_stopped = False
            self.last_goal_map = None
            self.state = IDLE
        self.status(f"RUN: {msg.data[:60]}")
        self.get_logger().info(f"new instruction: {msg.data!r}")

    def _depth_m(self, depth):
        d = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        if d.dtype == np.uint16:  # 16UC1 mm (sim physx depth, RealSense)
            return d.astype(np.float32) / 1000.0
        elif d.dtype == np.float32:  # 32FC1 m (sim rtx depth)
            return d.astype(np.float32)
        else:
            assert False, f"unreachable depth dtype={d.dtype!r}"

    def on_rgbd(self, color, depth, color_ld=None, depth_ld=None):
        if self.crop is None or (self.lookdown_ns and self.intr_ld is None):
            return
        rgb = self.bridge.imgmsg_to_cv2(color, "rgb8")
        d = self._depth_m(depth)
        look = None
        if color_ld is not None:
            look = (self.crop_ld.apply_rgb(self.bridge.imgmsg_to_cv2(color_ld, "rgb8")),
                    self.crop_ld.apply_depth(self._depth_m(depth_ld)), depth_ld.header.frame_id)
        with self.lock:
            self.latest = (color.header.stamp, self.crop.apply_rgb(rgb), self.crop.apply_depth(d),
                           depth.header.frame_id, look)
            self.new_frame = True

    # ------------------------------------------------------------------ helpers
    def lookup(self, target, source, stamp, timeout=0.2):
        try:
            return tf_to_T(self.tf_buffer.lookup_transform(target, source, Time.from_msg(stamp),
                                                           timeout=Duration(seconds=timeout))), True
        except Exception:
            try:  # fall back to latest
                return tf_to_T(self.tf_buffer.lookup_transform(target, source, Time())), False
            except Exception:
                return None, False

    def world_T_base(self, stamp=None, timeout=0.2):
        """(4x4 world<-base, synced) at the image stamp, or latest when stamp is None. World = self.world_frame."""
        if self.pose_source == "odom":
            t = None if stamp is None else stamp.sec + stamp.nanosec * 1e-9
            deadline = time.time() + timeout
            while True:  # wait for an odom at/after the image stamp (it can arrive later than the image)
                with self.lock:
                    buf = list(self.odom_buf)
                if t is None or (buf and buf[-1][0] >= t) or time.time() > deadline:
                    break
                time.sleep(0.005)
            if not buf:
                return None, False
            if t is None:
                return buf[-1][1], True
            # interpolate between the odoms around the image stamp (like tf2), else take the nearest one
            i = next((k for k, e in enumerate(buf) if e[0] >= t), len(buf))
            if 0 < i < len(buf):
                (t0, T0), (t1, T1) = buf[i - 1], buf[i]
                self.pose_dt = max(t - t0, t1 - t)  # gap to the farther neighbour
                return interp_T(T0, T1, (t - t0) / max(t1 - t0, 1e-9)), (t1 - t0) <= 2 * self.odom_max_dt
            st, T = buf[min(i, len(buf) - 1)]
            self.pose_dt = st - t
            return T, abs(st - t) <= self.odom_max_dt
        elif self.pose_source == "tf":
            if stamp is None:
                return self.lookup(self.map_frame, self.base_frame, Time().to_msg(), timeout=0.0)
            return self.lookup(self.map_frame, self.base_frame, stamp)
        else:
            assert False, f"unreachable pose_source={self.pose_source!r}"

    def robot_T(self):
        T, _ = self.world_T_base()
        return T

    def publish_cmd(self, pub, v, w):
        if self.dry_run:
            return
        t = Twist()
        t.linear.x = float(np.clip(v, -self.v_max, self.v_max))
        t.angular.z = float(np.clip(w, -self.w_max, self.w_max))
        pub.publish(t)

    def status(self, text):
        self.status_pub.publish(String(data=text))

    # ------------------------------------------------------------------ planning
    def planning_loop(self):
        while rclpy.ok():
            t0 = time.time()
            with self.lock:
                busy = self.state == PID and time.time() - self.pid_started < self.discrete_block_max_s
                ready = self.new_frame and self.instruction and not self.episode_stopped and not busy
                if ready:
                    stamp, rgb, depth, depth_frame, look = self.latest
                    self.new_frame = False
                    reset = self.need_reset
                    instruction = self.instruction
            if not ready:
                time.sleep(0.05)
                continue
            if self.T_base_cam is None:
                self.T_base_cam, _ = self.lookup(self.base_frame, depth_frame, stamp)
                if self.T_base_cam is None:
                    self.get_logger().warn(f"no TF {self.base_frame}<-{depth_frame} yet", throttle_duration_sec=2.0)
                    continue
            if look is not None and self.T_base_cam_ld is None:
                self.T_base_cam_ld, _ = self.lookup(self.base_frame, look[2], stamp)
                if self.T_base_cam_ld is None:
                    self.get_logger().warn(f"no TF {self.base_frame}<-{look[2]} yet", throttle_duration_sec=2.0)
                    continue
            if self.ground_z_auto:
                gz = estimate_ground_z(depth, self.intr, self.T_base_cam)
                if gz is not None:
                    self.ground_z = 0.8 * self.ground_z + 0.2 * gz
            T_map_base, tf_ok = self.world_T_base(stamp)
            if T_map_base is None:
                self.get_logger().warn(f"no {self.world_frame}<-base pose ({self.pose_source})",
                                       throttle_duration_sec=2.0)
                continue
            tf_dt = (self.pose_dt if self.pose_source == "odom" else 0.0) if tf_ok else float("nan")

            try:
                self.request_in_flight_since = time.time()
                self.frame_id += 1
                fid = self.frame_id
                self.frame_cache[fid] = (rgb, depth, T_map_base, look, stamp)
                while len(self.frame_cache) > 60:
                    self.frame_cache.popitem(last=False)
                reply = self.query(rgb, depth, instruction, reset, look, fid)
                self.request_in_flight_since = None
                self.last_reply_time = time.time()
                with self.lock:
                    self.need_reset = False if reset else self.need_reset
            except Exception as e:  # server down / timeout -> control_loop stops the robot
                self.request_in_flight_since = None
                self.get_logger().error(f"server request failed: {e}", throttle_duration_sec=2.0)
                time.sleep(0.5)
                continue
            try:
                self.handle_reply(reply, stamp, rgb, depth, T_map_base, tf_ok, tf_dt, look, fid)
            except Exception:  # keep the planning thread alive; the reply is logged for debugging
                import traceback

                self.get_logger().error(f"handle_reply failed for {reply!r:.300}\n{traceback.format_exc()}")
            time.sleep(max(0.0, self.plan_period - (time.time() - t0)))

    @staticmethod
    def _encode(rgb, depth_m):
        rgb_buf = io.BytesIO()
        PILImage.fromarray(rgb).save(rgb_buf, format="JPEG", quality=95)
        d = np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint16)  # realworld protocol (m * 1e4)
        d_buf = io.BytesIO()
        PILImage.fromarray(d).save(d_buf, format="PNG")
        return ("rgb.jpg", rgb_buf.getvalue(), "image/jpeg"), ("depth.png", d_buf.getvalue(), "image/png")

    def query(self, rgb, depth_m, instruction, reset, look=None, frame_id=None):
        data = {"reset": bool(reset), "instruction": instruction, "intrinsic": self.intr.K().tolist(),
                "frame_id": frame_id}
        files = dict(zip(("image", "depth"), self._encode(rgb, depth_m)))
        if look is not None:
            files.update(zip(("lookdown_image", "lookdown_depth"), self._encode(look[0], look[1])))
        r = requests.post(
            self.server_url + "/eval_dual",
            files=files,
            data={"json": json.dumps(data)},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def handle_reply(self, reply, stamp, rgb, depth, T_map_base, tf_ok, tf_dt, look=None, fid=None):
        s2_wh = tuple(reply.get("s2_img_size", (384, 384)))
        pixel_frame = reply.get("pixel_frame", "fpv")
        # frame the pixel goal belongs to (older than the current one with an async server)
        pfid = reply.get("pixel_frame_id")
        same_frame = pfid is None or pfid == fid or pfid not in self.frame_cache
        if same_frame:
            g_rgb, g_depth, g_T_map_base, g_look = rgb, depth, T_map_base, look
            g_stamp = stamp
        else:
            g_rgb, g_depth, g_T_map_base, g_look, g_stamp = self.frame_cache[pfid]
        # camera the pixel goal refers to (S2 "↓" answers come from the 30° look-down frame when available)
        if pixel_frame == "lookdown" and g_look is not None:
            p_rgb, p_depth, p_intr, p_T = g_look[0], g_look[1], self.intr_ld, self.T_base_cam_ld
        elif pixel_frame in ("fpv", "lookdown"):
            p_rgb, p_depth, p_intr, p_T = g_rgb, g_depth, self.intr, self.T_base_cam
        else:
            assert False, f"unreachable pixel_frame={pixel_frame!r}"
        row = col = None
        proj_method, goal_xy, dpx = "", (None, None), (float("nan"),) * 3
        traj_px = None
        traj_end = (float("nan"), float("nan"))  # S1 path end, base frame of the request (x fwd, y left)
        traj_start, traj_head = (float("nan"), float("nan")), float("nan")  # first point / first-segment heading
        goal_xyz = None
        action = None

        if "discrete_action" in reply:
            action = reply["discrete_action"]
            self.on_discrete(action, T_map_base)
        elif "trajectory" in reply:
            traj = np.asarray(reply["trajectory"], np.float64)[:, :2]
            traj_end = (float(traj[-1, 0]), float(traj[-1, 1]))
            traj_start = (float(traj[0, 0]), float(traj[0, 1]))
            seg = traj[1:] - traj[0]
            far = np.hypot(seg[:, 0], seg[:, 1]) > 0.05  # first waypoint 5 cm away defines the heading
            if far.any():
                d = seg[np.argmax(far)]
                traj_head = math.degrees(math.atan2(d[1], d[0]))
            traj_map = transform_points(T_map_base, np.column_stack([traj, np.zeros(len(traj))]))[:, :2]
            self.publish_path(traj_map[self.traj_skip:], stamp)
            # S1 path is in the CURRENT frame -> re-express it in the frame the image/pixel goal belongs to
            traj_g = transform_points(np.linalg.inv(g_T_map_base) @ T_map_base,
                                      np.column_stack([traj, np.zeros(len(traj))]))[:, :2]
            uv, front = base_traj_to_pixels(traj_g, p_intr, p_T, self.ground_z)
            traj_px = uv[front]
            if not same_frame:
                front = np.zeros(len(front), bool)  # skip the S1/S2 agreement check across frames
            if self.mode == "trajectory":
                self.on_trajectory(traj_map[self.traj_skip:])
            pg = reply.get("pixel_goal")
            if pg is not None and reply.get("is_s2_step"):
                row, col = pg
                u, v = self.pixel_to_proc(row, col, s2_wh)
                # V5: S2 pixel vs projected S1 path end, under three coordinate-space hypotheses
                # (S2 input 384x384 / Qwen smart_resize 392x392 / already in the 640x480 frame)
                if front[-1]:
                    end = uv[-1]
                    hyps = [rescale_pixel(row, col, s2_wh, self.proc_wh),
                            rescale_pixel(row, col, (392, 392), self.proc_wh),
                            (col, row)]
                    dpx = tuple(float(np.hypot(end[0] - hu, end[1] - hv)) for hu, hv in hyps)
                    self.agree_pub.publish(Float32MultiArray(data=list(dpx)))
                p_base, proj_method = pixel_to_base(u, v, p_depth, p_intr, p_T, self.ground_z,
                                                    {"radius": self.depth_window})
                if p_base is not None:
                    p_map = transform_points(g_T_map_base, p_base)[0]
                    goal_xy = (float(p_map[0]), float(p_map[1]))
                    goal_xyz = [float(v) for v in p_map[:3]]
                    self.publish_goal_marker(p_map, proj_method)
                    if self.mode == "pixel_goal":
                        self.on_pixel_goal(p_map, T_map_base)
                else:
                    self.get_logger().warn(f"pixel goal {pg} not projectable (no depth, above horizon)")
        else:
            assert False, f"unreachable reply keys={sorted(reply)!r}"

        if reply.get("llm_output"):
            self.s2_pub.publish(String(data=json.dumps({
                "stamp": stamp.sec + stamp.nanosec * 1e-9, "llm": reply.get("llm_output"), "action": action,
                "pixel": [row, col] if row is not None else None, "pixel_frame": pixel_frame,
                "proj": proj_method, "goal_xy": goal_xy if goal_xy[0] is not None else None,
                "t_infer": reply.get("t_infer"), "instruction": self.instruction, "mode": self.mode,
                "pixel_stamp": g_stamp.sec + g_stamp.nanosec * 1e-9, "goal_xyz": goal_xyz})))
        self.publish_vis(p_rgb, stamp, row, col, s2_wh, traj_px, action, reply.get("llm_output"),
                         tag="LD " if (pixel_frame == "lookdown" and look is not None) else "")
        if self.log:
            self.log.writerow([f"{time.time():.3f}", f"{stamp.sec + stamp.nanosec * 1e-9:.3f}", int(tf_ok), tf_dt,
                               "action" if action is not None else "traj", action, row, col, proj_method,
                               goal_xy[0], goal_xy[1], dpx[0], dpx[1], dpx[2], f"{reply.get('t_infer', 0):.3f}",
                               (reply.get("llm_output") or "").replace("\n", " "), pixel_frame,
                               f"{traj_end[0]:.3f}", f"{traj_end[1]:.3f}", f"{traj_start[0]:.3f}",
                               f"{traj_start[1]:.3f}", f"{traj_head:.1f}"])
            self.log_f.flush()

    def pixel_to_proc(self, row, col, s2_wh):
        if self.pixel_space == "input":
            return float(col), float(row)
        elif self.pixel_space == "s2":
            return rescale_pixel(row, col, s2_wh, self.proc_wh)
        else:
            assert False, f"unreachable pixel_space={self.pixel_space!r}"

    # ------------------------------------------------------------------ mode logic
    def on_trajectory(self, traj_map):
        if len(traj_map) < 2:
            return
        with self.lock:
            if self.mpc is None:
                self.mpc = Mpc_controller(traj_map, v_max=self.v_max, w_max=self.w_max)
            else:
                self.mpc.update_ref_traj(traj_map)
            self.state = MPC

    def on_pixel_goal(self, p_map, T_map_base):
        robot = T_map_base[:2, 3]
        d = p_map[:2] - robot
        dist = float(np.hypot(*d))
        if dist > self.goal_max_dist:
            self.get_logger().warn(f"goal {dist:.1f} m away -> clamped to {self.goal_max_dist:.1f} m")
            p_map = p_map.copy()
            p_map[:2] = robot + d / dist * self.goal_max_dist
        same = (self.last_goal_map is not None
                and np.hypot(*(p_map[:2] - self.last_goal_map[:2])) < self.goal_min_change)
        if same and time.time() - self.last_goal_time < self.goal_resend_s:
            return
        yaw = math.atan2(d[1], d[0])
        with self.lock:
            self.state = IDLE  # gdm drives
        if self.send_goal(p_map[0], p_map[1], yaw):
            self.last_goal_map, self.last_goal_time = p_map, time.time()

    def on_discrete(self, actions, T_map_base):
        if actions == [ACT_LOOKDOWN] or not actions:
            return
        if ACT_STOP in actions:
            self.get_logger().info("S2: STOP -> episode finished")
            with self.lock:
                self.state = STOPPED
                self.episode_stopped = True
            self.send_stop()
            self.publish_cmd(self.cmd_pub, 0.0, 0.0)
            self.status("STOP")
            return
        T = self.robot_T()
        if T is None:
            return
        goal = T.copy()
        for a in actions:
            yaw = yaw_of(goal)
            if a == ACT_FWD:
                goal[0, 3] += 0.25 * math.cos(yaw)
                goal[1, 3] += 0.25 * math.sin(yaw)
            elif a == ACT_LEFT:
                goal[:3, :3] = se2(0, 0, math.radians(15.0))[:3, :3] @ goal[:3, :3]
            elif a == ACT_RIGHT:
                goal[:3, :3] = se2(0, 0, math.radians(-15.0))[:3, :3] @ goal[:3, :3]
            elif a == ACT_LOOKDOWN:
                pass  # fixed-pitch camera; the server already re-queried with look_down
            else:
                assert False, f"unreachable action={a!r}"
        if self.mode == "pixel_goal":
            self.send_stop()  # halt gdm so it does not resume the previous goal after the turn
            self.last_goal_map = None
        with self.lock:
            self.pid_goal = goal
            self.state = PID
            self.pid_started = time.time()
            self.new_frame = False  # the next query uses a frame taken after the action finished

    # ------------------------------------------------------------------ gdm goal interface
    def send_goal(self, x, y, yaw):
        """Returns True if the goal was handed to a connected subscriber (else it is retried next S2 answer)."""
        if self.dry_run:
            return True
        if self.goal_interface == "goal_pose":
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.world_frame
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            q = yaw_quat(yaw)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
            if self.goal_pose_pub.get_subscription_count() == 0:
                self.get_logger().warn("no /goal_pose subscriber yet, goal kept for retry", throttle_duration_sec=2.0)
                return False
            self.goal_pose_pub.publish(ps)
        elif self.goal_interface == "gp":
            odom = self.odom
            if odom is None:
                self.get_logger().warn("no odom yet, goal dropped")
                return False
            if self.gp_pub.get_subscription_count() == 0:
                self.get_logger().warn("gdm not subscribed to /gdg/data/gp yet, goal kept for retry",
                                       throttle_duration_sec=2.0)
                return False
            # same relative-pose convention as gdm gp_simulator (calculate_relative_pose):
            # position/orientation relative to the body at gp.odometry
            o = odom.pose.pose
            T_odom = make_T([o.position.x, o.position.y, o.position.z],
                            [o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w])
            # planar like gdm (abs = odom xy + R(odom yaw) rel): the full 3D inverse tilted rel by the body
            # pitch/roll -> goal off by ~2 cm (verify/gdm_test.sh 2026-10-07)
            rel = np.linalg.inv(se2(o.position.x, o.position.y, yaw_of(T_odom))) @ se2(x, y, yaw)
            gp = self.GP()
            gp.header.stamp = self.get_clock().now().to_msg()
            gp.odometry = odom
            pt = Point(x=float(rel[0, 3]), y=float(rel[1, 3]), z=0.0)
            q = yaw_quat(yaw_of(rel))
            gp.position.append(pt)
            gp.orientation.append(Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
            gp.gp_type.append(self.GP.VIA_GP)
            gp.etype.append(self.GP.EDGE_BASIC)
            self.gp_pub.publish(gp)
        else:
            assert False, f"unreachable goal_interface={self.goal_interface!r}"
        self.get_logger().info(f"goal -> gdm ({self.goal_interface}): map ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f}")
        self.status(f"GOAL {x:.2f} {y:.2f}")
        return True

    def send_stop(self):
        if self.dry_run or self.mode != "pixel_goal":
            return
        if self.gp_pub is None or self.odom is None:
            self.get_logger().warn("FORCE_TO_STOP unavailable (no gd_ifc_pkg / odom)")
            return
        gp = self.GP()
        gp.header.stamp = self.get_clock().now().to_msg()
        gp.odometry = self.odom
        gp.gp_type.append(self.GP.FORCE_TO_STOP_GP)
        gp.etype.append(self.GP.EDGE_BASIC)
        self.gp_pub.publish(gp)

    # ------------------------------------------------------------------ control (10 Hz, wall clock)
    def control_loop(self):
        while rclpy.ok():
            time.sleep(0.1)
            with self.lock:
                state, mpc, pid_goal = self.state, self.mpc, self.pid_goal
            if state in (IDLE, STOPPED):
                continue
            assert state in (MPC, PID), f"unreachable state={state!r}"
            # server watchdog: a request hanging longer than server_timeout -> stop
            if self.request_in_flight_since and time.time() - self.request_in_flight_since > self.server_timeout \
                    and state == MPC:
                self.publish_cmd(self.cmd_pub, 0.0, 0.0)
                self.get_logger().warn("server timeout -> holding still", throttle_duration_sec=2.0)
                continue
            T = self.robot_T()
            if T is None:
                continue
            if state == MPC and mpc is not None:
                x0 = np.array([T[0, 3], T[1, 3], yaw_of(T)])
                try:
                    u, _ = mpc.solve(x0)
                    self.publish_cmd(self.cmd_pub, u[0, 0], u[0, 1])
                except Exception as e:
                    self.get_logger().warn(f"MPC failed: {e}", throttle_duration_sec=2.0)
                    self.publish_cmd(self.cmd_pub, 0.0, 0.0)
            elif state == PID and pid_goal is not None:
                # vel=0: see module docstring (odom twist unusable); Kd defaults to 0
                v, w, e_p, e_r = self.pid.solve(T, pid_goal, np.zeros(2))
                v_meas, w_meas = self.measured_velocity(T)
                v, w, done = discrete_step(v, w, e_p, e_r, v_meas, w_meas, self.pid_v_min, self.pid_w_min,
                                           lead_s=self.pid_lead_s)
                self.publish_cmd(self.discrete_pub, max(0.0, v), w)
                if done:
                    with self.lock:
                        if self.state == PID:
                            self.state = IDLE

    def measured_velocity(self, T):
        """Forward speed / yaw rate from the pose difference between control ticks (odom twist is not used)."""
        now = time.time()
        prev, self._prev_pose = getattr(self, "_prev_pose", None), (now, T)
        if prev is None or now - prev[0] > 0.5:
            return 0.0, 0.0
        dt = now - prev[0]
        d = T[:2, 3] - prev[1][:2, 3]
        yaw0 = yaw_of(prev[1])
        v = (d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0)) / dt
        w = math.atan2(math.sin(yaw_of(T) - yaw0), math.cos(yaw_of(T) - yaw0)) / dt
        return v, w

    # ------------------------------------------------------------------ visualisation
    def publish_path(self, traj_map, stamp):
        msg = PathMsg()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = stamp
        for x, y in traj_map:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.traj_pub.publish(msg)

    def publish_goal_marker(self, p_map, method):
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "vln_goal", 0, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p_map)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.a = 1.0
        m.color.g = 1.0 if method == "depth" else 0.5
        m.color.r = 0.0 if method == "depth" else 1.0
        self.goal_marker_pub.publish(m)

    def publish_vis(self, rgb, stamp, row, col, s2_wh, traj_px, action, llm, tag=""):
        save = self.log is not None and llm  # every S2 answer is saved as png next to the csv log
        if self.vis_img_pub.get_subscription_count() == 0 and self.s2_img_pub.get_subscription_count() == 0 and not save:
            return
        import cv2

        img = np.ascontiguousarray(rgb.copy())
        if traj_px is not None and len(traj_px) > 1:
            cv2.polylines(img, [traj_px.astype(np.int32).reshape(-1, 1, 2)], False, (0, 120, 255), 2)
        if row is not None:
            u, v = self.pixel_to_proc(row, col, s2_wh)  # same mapping send_goal projects
            cv2.circle(img, (int(u), int(v)), 8, (0, 255, 0), 2)
        label = f"act={action}" if action is not None else ("S2" if row is not None else "S1")
        txt = (llm or "").translate(str.maketrans({"←": "L", "→": "R", "↑": "F", "↓": "D"}))  # cv2 font is ASCII
        cv2.putText(img, tag + label + (f" {txt[:40]}" if txt else ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)
        if save:
            fn = os.path.join(self.log_dir, f"s2_{stamp.sec:06d}_{stamp.nanosec // 1000000:03d}.jpg")
            cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        msg.header.stamp = stamp
        msg.header.frame_id = "vln_proc"
        self.vis_img_pub.publish(msg)
        if llm:
            self.s2_img_pub.publish(msg)


def main():
    rclpy.init()
    node = VLNBridge()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd(node.cmd_pub, 0.0, 0.0)
        except Exception:  # context already shut down by SIGINT
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
