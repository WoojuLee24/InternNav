#!/usr/bin/env python3
"""Standalone ROS2 node that publishes /internav/occupancy_grid from depth image.

Subscribes:
  /camera/camera/color/image_raw                    (sensor_msgs/Image)
  /camera/camera/aligned_depth_to_color/image_raw   (sensor_msgs/Image)
  <odom_topic>                                       (nav_msgs/Odometry)

Publishes:
  /internav/occupancy_grid   (nav_msgs/OccupancyGrid)

Usage:
  python occgrid_publisher.py \
      --calib calib/calib_scout.txt \
      --frame_id os_sensor \
      --resolution 0.1 \
      --grid_size 100
"""

import argparse
import copy
import math
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock
from transformation import Calibration


# ── global shared state ───────────────────────────────────────────────────────
node: Node = None
calib: Calibration = None

depth_image = None          # float32 numpy array (H, W), meters
odom = None                 # [x, y, yaw]
new_depth_arrived = False

rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()


# ── helpers ───────────────────────────────────────────────────────────────────

def build_occupancy_grid(pcloud_xy, stamp, frame_id,
                         resolution=0.1, grid_size=100,
                         center_x=0.0, center_y=0.0) -> OccupancyGrid:
    """Build a ROS2 OccupancyGrid from (N, 2) XY obstacle points.

    Occupied cells → 100, unknown → -1.
    """
    half = (grid_size * resolution) / 2.0
    xmin = center_x - half
    ymin = center_y - half

    grid = -np.ones((grid_size, grid_size), dtype=np.int16)

    if len(pcloud_xy) > 0:
        pts = np.asarray(pcloud_xy, dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) > 0:
            cx = np.floor((pts[:, 0] - xmin) / resolution).astype(np.int32)
            cy = np.floor((pts[:, 1] - ymin) / resolution).astype(np.int32)
            valid = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
            grid[cy[valid], cx[valid]] = 100

    msg = OccupancyGrid()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.info.resolution = float(resolution)
    msg.info.width = int(grid_size)
    msg.info.height = int(grid_size)
    msg.info.origin = Pose()
    msg.info.origin.position.x = float(xmin)
    msg.info.origin.position.y = float(ymin)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).astype(np.int8).tolist()
    return msg


# ── mapping thread ────────────────────────────────────────────────────────────

def mapping_thread(frame_id="os_sensor", resolution=0.1, grid_size=100,
                   z_min=0.05, z_max=2.0, depth_min=0.1, depth_max=10.0,
                   rate_hz=10.0):
    global node, calib, depth_image, odom, new_depth_arrived

    while node is None or calib is None:
        time.sleep(0.1)

    node.get_logger().info(
        f"[Mapping] Thread started. frame={frame_id} "
        f"res={resolution} grid={grid_size}x{grid_size}"
    )

    desired_dt = 1.0 / rate_hz
    u_grid = None
    v_grid = None

    while True:
        t0 = time.time()

        if not new_depth_arrived:
            time.sleep(0.01)
            continue

        # ── depth 복사 ──────────────────────────────────────────────────────
        rgb_depth_rw_lock.acquire_read()
        depth = copy.deepcopy(depth_image)
        rgb_depth_rw_lock.release_read()

        new_depth_arrived = False  # reset after copy (best-effort, no lock needed for bool)

        if depth is None:
            time.sleep(0.05)
            continue

        # ── odom 복사 ───────────────────────────────────────────────────────
        odom_rw_lock.acquire_read()
        cur_odom = copy.deepcopy(odom)
        odom_rw_lock.release_read()

        # ── Depth → point cloud (camera rect frame) ─────────────────────────
        h, w = depth.shape
        if u_grid is None or u_grid.size != h * w:
            vg, ug = np.mgrid[0:h, 0:w]
            u_grid = ug.flatten()
            v_grid = vg.flatten()

        z = depth.flatten()
        valid = (z > depth_min) & (z < depth_max)
        z_v = z[valid]
        u_v = u_grid[valid]
        v_v = v_grid[valid]

        fx, fy = calib.f_u, calib.f_v
        cx_cam, cy_cam = calib.c_u, calib.c_v
        x_cam = (u_v - cx_cam) * z_v / fx
        y_cam = (v_v - cy_cam) * z_v / fy
        pcloud_cam = np.stack([x_cam, y_cam, z_v], axis=-1)  # (N, 3)

        # ── Camera rect → robot/velodyne frame ─────────────────────────────
        # project_rect_to_velo: camera(right,down,fwd) → robot(fwd,left,up)
        pcloud_robot = calib.project_rect_to_velo(pcloud_cam)  # (N, 3)

        # ── 높이 필터 (robot frame Z: 위쪽 +) ──────────────────────────────
        z_r = pcloud_robot[:, 2]
        height_mask = (z_r > z_min) & (z_r < z_max)
        pcloud_filtered = pcloud_robot[height_mask]

        if len(pcloud_filtered) == 0:
            node.get_logger().debug(
                "[Mapping] No valid obstacle points after height filter.",
                throttle_duration_sec=3.0,
            )
            time.sleep(max(0.0, desired_dt - (time.time() - t0)))
            continue

        # ── 좌표계 및 grid center 결정 ──────────────────────────────────────
        if frame_id == "camera_init" and cur_odom is not None:
            x_, y_, yaw_ = cur_odom
            cos_y, sin_y = np.cos(yaw_), np.sin(yaw_)
            R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            xy_world = (R2 @ pcloud_filtered[:, :2].T).T + np.array([x_, y_])
            center_x, center_y = x_, y_
        else:
            # os_sensor (body frame): robot is at origin
            xy_world = pcloud_filtered[:, :2]
            center_x, center_y = 0.0, 0.0

        stamp = node.get_clock().now().to_msg()

        # ── OccupancyGrid 빌드 및 발행 ──────────────────────────────────────
        occ_msg = build_occupancy_grid(
            xy_world, stamp, frame_id,
            resolution=resolution, grid_size=grid_size,
            center_x=center_x, center_y=center_y,
        )
        node.occ_grid_pub.publish(occ_msg)

        node.get_logger().debug(
            f"[Mapping] OccGrid published. frame={frame_id} pts={len(xy_world)}",
            throttle_duration_sec=2.0,
        )

        time.sleep(max(0.0, desired_dt - (time.time() - t0)))


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class OccGridNode(Node):
    def __init__(self, odom_topic: str):
        super().__init__('occgrid_publisher')
        self.get_logger().info("Initializing OccGridNode...")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── subscribers ───────────────────────────────────────────────────────
        rgb_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")
        self.sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=30, slop=0.5)
        self.sync.registerCallback(self._rgb_depth_callback)

        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_callback, qos)
        self.get_logger().info(f"Subscribing to odom: {odom_topic}")

        # ── publishers ────────────────────────────────────────────────────────
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, '/internav/occupancy_grid', 5)

        self._cv_bridge = CvBridge()
        self.get_logger().info("OccGridNode initialized.")

    def _rgb_depth_callback(self, rgb_msg, depth_msg):
        global depth_image, new_depth_arrived

        raw_depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32)
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        depth_m = raw_depth / 1000.0
        depth_m[depth_m < 0] = 0

        rgb_depth_rw_lock.acquire_write()
        depth_image = depth_m
        rgb_depth_rw_lock.release_write()

        new_depth_arrived = True

        self.get_logger().debug("Received synced RGB-Depth.", throttle_duration_sec=2.0)

    def _odom_callback(self, msg):
        global odom

        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)

        odom_rw_lock.acquire_write()
        odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        odom_rw_lock.release_write()

        self.get_logger().debug(
            f"[Odom] x:{odom[0]:.2f} y:{odom[1]:.2f} yaw:{yaw:.2f}",
            throttle_duration_sec=3.0,
        )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Publish OccupancyGrid from depth image.')
    parser.add_argument('--odom_topic', type=str, default='/gdq/msg/gdq_odom',
                        help='ROS2 odometry topic (default: /gdq/msg/gdq_odom)')
    parser.add_argument('--calib', type=str,
                        default='/home/gdr/gd_vln/workspace/src/InternNav/scripts/realworld/calib/calib_scout.txt',
                        help='Path to calibration file')
    parser.add_argument('--frame_id', type=str, default='os_sensor',
                        help='Publish frame: os_sensor (body) or camera_init (world). default: os_sensor')
    parser.add_argument('--resolution', type=float, default=0.1,
                        help='Grid resolution in meters (default: 0.1)')
    parser.add_argument('--grid_size', type=int, default=100,
                        help='Grid side length in cells (default: 100 → 10m × 10m)')
    parser.add_argument('--z_min', type=float, default=0.05,
                        help='Min obstacle height in robot frame (default: 0.05 m)')
    parser.add_argument('--z_max', type=float, default=2.0,
                        help='Max obstacle height in robot frame (default: 2.0 m)')
    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='Min valid depth in meters (default: 0.1)')
    parser.add_argument('--depth_max', type=float, default=10.0,
                        help='Max valid depth in meters (default: 10.0)')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='OccupancyGrid publish rate in Hz (default: 10)')
    args = parser.parse_args()

    calib = Calibration(args.calib)

    mapping_thread_instance = threading.Thread(
        target=mapping_thread,
        args=(args.frame_id, args.resolution, args.grid_size,
              args.z_min, args.z_max, args.depth_min, args.depth_max, args.rate),
        daemon=True,
    )

    rclpy.init()

    try:
        node = OccGridNode(odom_topic=args.odom_topic)
        mapping_thread_instance.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("[Main] Keyboard Interrupt. Shutting down...")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()
