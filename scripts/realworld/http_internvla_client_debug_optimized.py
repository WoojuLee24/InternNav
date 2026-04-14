#!/usr/bin/env python3
"""
InternVLA CLIENT (OPTIMIZED  FOR REAL-WORLD ROBOT  version 1)

This is the optimized client that connects to:
- http_internvla_server_debug_optimized.py

Optimizations:
- Uses async HTTP requests
- Optimized for high-frequency inference
- Supports MPC and PID control modes

Usage:
    python http_internvla_client_debug_optimized.py --visualize
"""

import argparse
import copy
import io
import json
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Point, Pose, PoseStamped, PointStamped, Twist, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from visualization_msgs.msg import Marker
from PIL import Image as PIL_Image
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

frame_data = {}
frame_idx = 0

from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock
from transformation import Calibration


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# Global variables
policy_init = True
mpc = None
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
calib = None

# Safety limits - can be adjusted by speed profile/flags at runtime
MAX_LINEAR_VEL = 0.6  # m/s - default FAST profile linear limit
MAX_ANGULAR_VEL = 0.5  # rad/s - default FAST profile angular limit
pid = None  # Initialized later with safety limits

SPEED_PROFILES = {
    'SAFE': (0.12, 0.20),
    'MEDIUM': (0.30, 0.35),
    'FAST': (0.60, 0.50),
    'FASTER': (0.80, 0.70),
    'FASTEST': (1.00, 0.90),
}

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

# Performance tracking
request_times = deque(maxlen=2000)
last_stats_log_time = 0.0
DEBUG_MODE = False
S2_MODE = 'async'


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url='http://127.0.0.1:5802/eval_dual'):
    global policy_init, http_idx, first_running_time, last_stats_log_time
    
    if manager:
        manager.get_logger().debug(f"[HTTP] Preparing request. idx: {http_idx + 1}, reset: {policy_init}")

    data = {"reset": policy_init, "idx": http_idx}
    data["s2_mode"] = S2_MODE
    if manager is not None and hasattr(manager, 'instruction') and manager.instruction:
        data["instruction"] = manager.instruction
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    
    try:
        response = requests.post(url, files=files, data={'json': json_data}, timeout=60)
    except Exception as e:
        if manager:
            manager.get_logger().error(f"[HTTP] Request Failed: {e}")
        return {}

    if not response.ok:
        if manager:
            manager.get_logger().error(f"[HTTP] Server error {response.status_code}: {response.text[:200]}")
        return {}

    if manager:
        manager.get_logger().debug(f"[HTTP] Response: {response.text[:200]}")
        
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    
    latency = time.time() - start
    request_times.append(latency)
    
    if manager and DEBUG_MODE and http_idx % 10 == 0:
        avg_latency = sum(request_times) / len(request_times) if request_times else 0
        hz = 1.0 / avg_latency if avg_latency > 0 else 0
        manager.get_logger().info(f"[HTTP] idx: {http_idx} | Latency: {latency*1000:.1f}ms | Avg: {hz:.1f} Hz")

    now = time.time()
    if manager and DEBUG_MODE and (now - last_stats_log_time) >= 5.0:
        stats_url = url.replace('/eval_dual', '/stats')
        try:
            stats_resp = requests.get(stats_url, timeout=3)
            if stats_resp.ok:
                stats = stats_resp.json()
                manager.get_logger().info(
                    f"[S2Stats] s2_hz={stats.get('s2_hz', 'n/a')} "
                    f"s2_avg_ms={stats.get('s2_avg_ms', 'n/a')} "
                    f"s2_updates={stats.get('s2_updates', 'n/a')} "
                    f"s2_errors={stats.get('s2_errors', 'n/a')}"
                )
        except Exception as e:
            manager.get_logger().warn(f"[S2Stats] fetch failed: {e}", throttle_duration_sec=5.0)
        last_stats_log_time = now

    try:
        return json.loads(response.text)
    except Exception as e:
        if manager:
            manager.get_logger().error(f"[HTTP] Invalid JSON response: {e}; body={response.text[:200]}")
        return {}


def apply_safety_limits(v, w):
    """Apply safety limits to velocity commands."""
    v = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, v))
    w = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, w))
    return v, w

def control_thread():
    global desired_v, desired_w, manager, pid
    
    while manager is None:
        time.sleep(0.001)
    
    pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=MAX_LINEAR_VEL, max_w=MAX_ANGULAR_VEL)
    
    manager.get_logger().info(f"[Thread] Control Thread Started. Safety limits: max_v={MAX_LINEAR_VEL}, max_w={MAX_ANGULAR_VEL}")

    last_log_time = time.time()
    loop_count = 0
    
    while True:
        global current_control_mode
        
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                try:
                    opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                    v, w = apply_safety_limits(v, w)
                    desired_v, desired_w = v, w
                    manager.move(v, 0.0, w)
                except Exception as e:
                    manager.get_logger().error(f"[MPC] Solve Error: {e}")

        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                v, w = apply_safety_limits(v, w)
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        
        loop_count += 1
        current_time = time.time()
        if DEBUG_MODE and current_time - last_log_time >= 5.0:
            manager.get_logger().info(
                f"[S1Stats] control_hz={loop_count/5.0:.0f} v={desired_v:.3f} w={desired_w:.3f} mode={current_control_mode.name}"
            )
            loop_count = 0
            last_log_time = current_time


def planning_thread(frame_id="os_sensor", traj_width=0.05, subgoal_radius=0.2):
    global trajs_in_world, manager
    
    while manager is None:
        time.sleep(0.01)
    
    manager.get_logger().info("[Thread] Planning Thread Started successfully.")

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.001
        time.sleep(0.0001)

        if not manager.new_image_arrived:
            continue
            
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            
            manager.get_logger().debug(f"[Plan] Calling dual_sys_eval... Sync Diff: {min_diff:.5f}s")
            
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                
                if DEBUG_MODE:
                    manager.get_logger().info(f"[Plan] Received Trajectory. Length: {traj_len:.4f}")
                
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                
                if DEBUG_MODE:
                    manager.get_logger().info(f"[Plan] Updated Trajs in World. Time: {time.time()}")

                manager.last_trajs_in_world = trajs_in_world
                manager.last_traj_update_time = time.time()

                _stamp = manager.get_clock().now().to_msg()
                _traj_marker = Marker()
                _traj_marker.header = Header(stamp=_stamp, frame_id=frame_id)
                _traj_marker.ns = "trajectory"
                _traj_marker.id = 0
                _traj_marker.type = Marker.LINE_STRIP
                _traj_marker.action = Marker.ADD
                _traj_marker.scale.x = traj_width
                _traj_marker.color.r = 0.0
                _traj_marker.color.g = 1.0
                _traj_marker.color.b = 0.0
                _traj_marker.color.a = 1.0
                if frame_id == "os_sensor":
                    _traj_pts = [[float(t[0]), float(t[1])] for i, t in enumerate(trajectory) if i >= 3]
                else:
                    _traj_pts = [[float(pt[0]), float(pt[1])] for pt in trajs_in_world]
                for _x, _y in _traj_pts:
                    _p = Point()
                    _p.x = _x
                    _p.y = _y
                    _p.z = 0.0
                    _traj_marker.points.append(_p)
                manager.traj_pub.publish(_traj_marker)

                if 'pixel_goal' in response:
                    manager.last_pixel_goal = response['pixel_goal']
                    if calib is not None and infer_depth is not None:
                        _row, _col = int(response['pixel_goal'][0]), int(response['pixel_goal'][1])
                        _dh, _dw = infer_depth.shape
                        if 0 <= _row < _dh and 0 <= _col < _dw:
                            _z_sub = infer_depth[_row, _col]
                            if _z_sub > 0.1:
                                _fx, _fy = calib.f_u, calib.f_v
                                _cx, _cy = calib.c_u, calib.c_v
                                _x_cam = (_col - _cx) * _z_sub / _fx
                                _y_cam = (_row - _cy) * _z_sub / _fy
                                _pc_robot = calib.project_rect_to_velo(
                                    np.array([[_x_cam, _y_cam, _z_sub]]))[0]
                                if frame_id == "os_sensor":
                                    _sub_x, _sub_y = float(_pc_robot[0]), float(_pc_robot[1])
                                else:
                                    _ox, _oy, _oyaw = odom_infer
                                    _R2 = np.array([[np.cos(_oyaw), -np.sin(_oyaw)],
                                                    [np.sin(_oyaw),  np.cos(_oyaw)]])
                                    _xy_sub = _R2 @ _pc_robot[:2] + np.array([_ox, _oy])
                                    _sub_x, _sub_y = float(_xy_sub[0]), float(_xy_sub[1])
                                _sub_msg = Marker()
                                _sub_msg.header = Header(stamp=_stamp, frame_id=frame_id)
                                _sub_msg.ns = "subgoal"
                                _sub_msg.id = 0
                                _sub_msg.type = Marker.SPHERE
                                _sub_msg.action = Marker.ADD
                                _sub_msg.pose.position.x = _sub_x
                                _sub_msg.pose.position.y = _sub_y
                                _sub_msg.pose.position.z = float(_pc_robot[2])
                                _sub_msg.pose.orientation.w = 1.0
                                _sub_msg.scale.x = subgoal_radius
                                _sub_msg.scale.y = subgoal_radius
                                _sub_msg.scale.z = subgoal_radius
                                _sub_msg.color.r = 1.0
                                _sub_msg.color.g = 0.5
                                _sub_msg.color.b = 0.0
                                _sub_msg.color.a = 1.0
                                manager.subgoal_pub.publish(_sub_msg)

                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                    if DEBUG_MODE:
                        manager.get_logger().info("[Plan] MPC Controller Initialized")
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
                
            else:
                age = time.time() - manager.last_traj_update_time
                if manager.last_trajs_in_world is not None and age <= manager.traj_hold_sec:
                    if DEBUG_MODE:
                        manager.get_logger().info(
                            f"[Plan] No new trajectory. Reusing cached trajectory ({age:.1f}s old).",
                            throttle_duration_sec=2.0,
                        )
                else:
                    manager.last_trajs_in_world = None
                    manager.last_pixel_goal = None
                    _stamp = manager.get_clock().now().to_msg()
                    _del_traj = Marker()
                    _del_traj.header = Header(stamp=_stamp, frame_id=frame_id)
                    _del_traj.ns = "trajectory"
                    _del_traj.id = 0
                    _del_traj.action = Marker.DELETE
                    manager.traj_pub.publish(_del_traj)
                    _del_sub = Marker()
                    _del_sub.header = Header(stamp=_stamp, frame_id=frame_id)
                    _del_sub.ns = "subgoal"
                    _del_sub.id = 0
                    _del_sub.action = Marker.DELETE
                    manager.subgoal_pub.publish(_del_sub)
                    if DEBUG_MODE:
                        manager.get_logger().info("[Plan] No trajectory in response. Cleared.", throttle_duration_sec=2.0)

            if 'discrete_action' in response:
                actions = response['discrete_action']
                if DEBUG_MODE:
                    manager.get_logger().info(f"[Plan] Received Discrete Actions: {actions}")

                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
        else:
            pass  # Silent skip for performance

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


def build_occupancy_grid(pcloud_xy, stamp, frame_id, resolution=0.1, grid_size=100, center_x=0.0, center_y=0.0):
    """Build a ROS2 OccupancyGrid message from (N, 2) XY points."""
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


def visualize_thread(frame_id="os_sensor"):
    global manager, calib

    while manager is None or calib is None:
        time.sleep(0.1)

    manager.get_logger().info(f"[Thread] Visualize Thread Started successfully. OccGrid frame: {frame_id}")

    u_grid = None
    v_grid = None

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.01  # 100 Hz max

        if not manager.new_vis_image_arrived:
            time.sleep(0.001)
            continue
        manager.new_vis_image_arrived = False

        rgb_depth_rw_lock.acquire_read()
        depth_image = copy.deepcopy(manager.depth_image)
        rgb_depth_rw_lock.release_read()

        if depth_image is None:
            time.sleep(0.001)
            continue

        odom_rw_lock.acquire_read()
        odom = manager.odom.copy() if manager.odom else None
        odom_rw_lock.release_read()

        h, w = depth_image.shape
        if u_grid is None or u_grid.size != h * w:
            vg, ug = np.mgrid[0:h, 0:w]
            u_grid = ug.flatten()
            v_grid = vg.flatten()

        z = depth_image.flatten()
        valid = (z > 0.1) & (z < 10.0)
        z_v = z[valid]
        u_v = u_grid[valid]
        v_v = v_grid[valid]

        fx, fy = calib.f_u, calib.f_v
        cx_cam, cy_cam = calib.c_u, calib.c_v
        x_cam = (u_v - cx_cam) * z_v / fx
        y_cam = (v_v - cy_cam) * z_v / fy
        pcloud_cam = np.stack([x_cam, y_cam, z_v], axis=-1)

        pcloud_robot = calib.project_rect_to_velo(pcloud_cam)

        z_r = pcloud_robot[:, 2]
        height_mask = (z_r > 0.05) & (z_r < 2.0)
        pcloud_filtered = pcloud_robot[height_mask]

        if len(pcloud_filtered) == 0:
            time.sleep(max(0.0, DESIRED_TIME - (time.time() - start_time)))
            continue

        if frame_id == "camera_init" and odom is not None:
            x_, y_, yaw_ = odom
            cos_y, sin_y = np.cos(yaw_), np.sin(yaw_)
            R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            xy_world = (R2 @ pcloud_filtered[:, :2].T).T + np.array([x_, y_])
            center_x, center_y = x_, y_
        else:
            xy_world = pcloud_filtered[:, :2]
            center_x, center_y = 0.0, 0.0

        stamp = manager.get_clock().now().to_msg()

        occ_msg = build_occupancy_grid(
            xy_world, stamp, frame_id,
            resolution=0.1, grid_size=100,
            center_x=center_x, center_y=center_y,
        )
        manager.occ_grid_pub.publish(occ_msg)

        time.sleep(max(0.0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self, odom_topic='/odom_bridge', instruction='', traj_hold_sec=2.0):
        super().__init__('go2_manager_optimized')
        
        self.get_logger().info("Initializing Go2Manager Node (Optimized)...")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        rgb_down_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_down_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 30, 0.5)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, qos_profile)
        self.get_logger().info(f"Subscribing to odometry topic: {odom_topic}")

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        _tf = TransformStamped()
        _tf.header.stamp = self.get_clock().now().to_msg()
        _tf.header.frame_id = 'map'
        _tf.child_frame_id = 'camera_init'
        _tf.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(_tf)
        self.get_logger().info("[TF] Published static transform: map -> camera_init (identity)")

        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, '/internav/occupancy_grid', 5)
        self.traj_pub = self.create_publisher(Marker, '/internav/trajectory', 5)
        self.dummy_path_pub = self.create_publisher(Path, '/internav/dummy_path', 5)
        self.subgoal_pub = self.create_publisher(Marker, '/internav/subgoal', 5)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.last_pixel_goal = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        self.instruction = instruction
        self.traj_hold_sec = max(0.0, float(traj_hold_sec))
        self.last_traj_update_time = 0.0
        
        self.get_logger().info("Go2Manager Node Initialized Successfully.")

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        self.get_logger().debug("Received Synced RGB-Depth", throttle_duration_sec=2.0)
        
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes.getvalue()
        self.rgb_width = raw_image.shape[1]
        self.rgb_height = raw_image.shape[0]

        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time

        self.depth_bytes = depth_bytes.getvalue()
        self.depth_width = 640
        self.depth_height = 480
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time

        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.0e9
        self.odom_queue.append((odom_stamp, copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()
            self.get_logger().info(f"[Odom] First Odom Received. Pose: ({self.odom[0]:.2f}, {self.odom[1]:.2f})")
            
        self.get_logger().debug(
            f"[Odom] x:{self.odom[0]:.2f}, y:{self.odom[1]:.2f}, yaw:{yaw:.2f}", 
            throttle_duration_sec=3.0
        )

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            self.get_logger().error("homo_goal is None! Cannot change goal.")
            raise ValueError("Please initialize homo_goal before change it!")
        
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal
        if DEBUG_MODE:
            self.get_logger().info(f"[Goal] Goal Updated incrementally. Action: {actions}")

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InternVLA CLIENT (OPTIMIZED_1 FOR REAL-WORLD ROBOT')
    
    parser.add_argument('--odom_topic', type=str, default='/gdq/msg/gdq_odom', 
                        help='ROS2 odometry topic name')
    parser.add_argument('--calib', type=str, default='scripts/realworld/calib/calib_scout.txt',
                        help='Path to calibration file')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable visualize_thread (OccupancyGrid)')
    parser.add_argument('--frame_id', type=str, default='os_sensor',
                        help='Frame ID for OccupancyGrid (e.g. os_sensor, camera_init)')
    parser.add_argument('--traj_width', type=float, default=0.1,
                        help='Trajectory marker line width in meters')
    parser.add_argument('--subgoal_radius', type=float, default=0.3,
                        help='Subgoal marker sphere radius in meters')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:5802/eval_dual',
                        help='Server URL for HTTP inference')
    parser.add_argument('--instruction', type=str, default='',
                        help='Navigation instruction to send in each request (overrides server default)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable periodic S1/S2/HTTP debug stats logging')
    parser.add_argument('--s2-mode', type=str, default='async', choices=['async', 'sync'],
                        help='Planner mode expected by server (async or sync)')
    parser.add_argument('--speed-profile', type=str, default='MEDIUM', choices=['SAFE', 'MEDIUM', 'FAST', 'FASTER', 'FASTEST'],
                        help='Safety speed profile for robot control')
    parser.add_argument('--max-linear-vel', type=float, default=None,
                        help='Override max linear velocity (m/s)')
    parser.add_argument('--max-angular-vel', type=float, default=None,
                        help='Override max angular velocity (rad/s)')
    parser.add_argument('--traj-hold-sec', type=float, default=2.0,
                        help='Keep last trajectory this long when response has no new trajectory')
    
    args = parser.parse_args()

    print("=" * 60)
    print(" InternVLA CLIENT (OPTIMIZED_1 FOR REAL-WORLD ROBOT)")
    print("=" * 60)
    print(f"Server URL: {args.server_url}")
    print(f"Odom topic: {args.odom_topic}")
    print(f"Calib: {args.calib}")
    print(f"Visualize: {args.visualize}")
    print(f"Frame ID: {args.frame_id}")
    print(f"Debug Stats: {args.debug}")
    print(f"S2 Mode: {args.s2_mode}")
    print(f"Instruction override: {'enabled' if args.instruction else 'disabled'}")
    print(f"Trajectory Hold: {args.traj_hold_sec:.1f}s")

    profile_linear, profile_angular = SPEED_PROFILES[args.speed_profile]
    MAX_LINEAR_VEL = args.max_linear_vel if args.max_linear_vel is not None else profile_linear
    MAX_ANGULAR_VEL = args.max_angular_vel if args.max_angular_vel is not None else profile_angular
    print(f"Speed Profile: {args.speed_profile}")
    print(f"Safety Limits: max_v={MAX_LINEAR_VEL:.2f} m/s, max_w={MAX_ANGULAR_VEL:.2f} rad/s")
    print("=" * 60)

    DEBUG_MODE = args.debug
    S2_MODE = args.s2_mode

    calib = Calibration(args.calib)

    dummy_odom = [0.0, 0.0, 0.0]
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread, 
                                               args=(args.frame_id, args.traj_width, args.subgoal_radius))
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    if args.visualize:
        visualize_thread_instance = threading.Thread(target=visualize_thread, args=(args.frame_id,))
        visualize_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager(
            odom_topic=args.odom_topic,
            instruction=args.instruction,
            traj_hold_sec=args.traj_hold_sec,
        )

        control_thread_instance.start()
        planning_thread_instance.start()
        if args.visualize:
            visualize_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        if manager:
            manager.get_logger().info("[Main] Keyboard Interrupt. Shutting down...")
    finally:
        if manager:
            manager.destroy_node()
        rclpy.shutdown()
