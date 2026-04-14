#!/usr/bin/env python3
"""
Ultra-High-Performance InternVLA Client

Optimized for maximum Hz (50+ Hz):
- Sends requests as fast as possible
- Non-blocking HTTP with threading
- Minimal processing overhead

Usage:
    python http_internvla_client_ultra.py --visualize
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
from geometry_msgs.msg import Point, Twist, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from visualization_msgs.msg import Marker
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image
from std_msgs.msg import Header

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
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
calib = None

rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

# Performance tracking
request_times = deque(maxlen=100)
last_request_time = 0


def dual_sys_eval(image_bytes, depth_bytes, url='http://127.0.0.1:5802/eval_dual'):
    """Send HTTP request to server."""
    global policy_init, http_idx, last_request_time
    
    data = {"reset": policy_init, "idx": http_idx}
    json_data = json.dumps(data)
    policy_init = False
    
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    
    start = time.time()
    try:
        response = requests.post(url, files=files, data={'json': json_data}, timeout=30)
        last_request_time = time.time() - start
        request_times.append(last_request_time)
    except Exception as e:
        if manager:
            manager.get_logger().error(f"[HTTP] Request Failed: {e}")
        return {}
    
    http_idx += 1
    
    if manager and http_idx % 20 == 0:
        avg_latency = sum(request_times) / len(request_times) if request_times else 0
        hz = 1.0 / avg_latency if avg_latency > 0 else 0
        manager.get_logger().info(f"[HTTP] idx: {http_idx} | Latency: {last_request_time*1000:.1f}ms | Rate: {hz:.1f} Hz")
    
    return json.loads(response.text)


def control_thread():
    """MPC/PID control thread."""
    global desired_v, desired_w, manager, mpc, current_control_mode
    
    while manager is None:
        time.sleep(0.1)
    
    manager.get_logger().info("[Thread] Control Thread Started.")

    while True:
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            
            if mpc is not None and manager is not None and odom is not None:
                try:
                    opt_u_controls, _ = mpc.solve(np.array(odom))
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
                    desired_v, desired_w = v, w
                    manager.move(v, 0.0, w)
                except Exception as e:
                    pass

        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, _, _ = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(0.05)


def planning_thread(frame_id="os_sensor", traj_width=0.05, subgoal_radius=0.2):
    """Planning thread - sends requests as fast as possible."""
    global trajs_in_world, manager, mpc, current_control_mode, calib
    
    while manager is None:
        time.sleep(0.1)
    
    manager.get_logger().info("[Thread] Planning Thread Started (Ultra-Fast Mode).")

    while True:
        if not manager.new_image_arrived:
            time.sleep(0.001)
            continue
            
        manager.new_image_arrived = False
        
        # Copy data
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        
        # Get closest odom
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
        odom_rw_lock.release_read()

        if odom_infer is None or rgb_bytes is None or depth_bytes is None:
            continue
        
        # Send request
        response = dual_sys_eval(rgb_bytes, depth_bytes)
        
        if not response:
            continue
        
        if 'trajectory' in response:
            trajectory = response['trajectory']
            trajs_in_world = []
            odom = odom_infer
            traj_len = np.linalg.norm(trajectory[-1][:2])
            
            manager.get_logger().info(f"[Plan] Trajectory received. Length: {traj_len:.4f}")
            
            for i, traj in enumerate(trajectory):
                if i < 3:
                    continue
                x_, y_, yaw_ = odom[0], odom[1], odom[2]
                w_T_b = np.array([
                    [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                    [np.sin(yaw_), np.cos(yaw_), 0, y_],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                trajs_in_world.append(w_P)
            trajs_in_world = np.array(trajs_in_world)
            
            # Publish trajectory
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
            _traj_pts = [[float(t[0]), float(t[1])] for i, t in enumerate(trajectory) if i >= 3]
            for _x, _y in _traj_pts:
                _p = Point()
                _p.x = _x
                _p.y = _y
                _p.z = 0.0
                _traj_marker.points.append(_p)
            manager.traj_pub.publish(_traj_marker)
            
            # Update MPC
            mpc_rw_lock.acquire_write()
            if mpc is None:
                mpc = Mpc_controller(np.array(trajs_in_world))
                manager.get_logger().info("[Plan] MPC Initialized")
            else:
                mpc.update_ref_traj(np.array(trajs_in_world))
            manager.request_cnt += 1
            mpc_rw_lock.release_write()
            current_control_mode = ControlMode.MPC_Mode
            
            # Publish subgoal
            if 'pixel_goal' in response and calib is not None and infer_depth is not None:
                _row, _col = int(response['pixel_goal'][0]), int(response['pixel_goal'][1])
                _dh, _dw = infer_depth.shape
                if 0 <= _row < _dh and 0 <= _col < _dw:
                    _z_sub = infer_depth[_row, _col]
                    if _z_sub > 0.1:
                        _fx, _fy = calib.f_u, calib.f_v
                        _cx, _cy = calib.c_u, calib.c_v
                        _x_cam = (_col - _cx) * _z_sub / _fx
                        _y_cam = (_row - _cy) * _z_sub / _fy
                        _pc_robot = calib.project_rect_to_velo(np.array([[_x_cam, _y_cam, _z_sub]]))[0]
                        _sub_x, _sub_y = float(_pc_robot[0]), float(_pc_robot[1])
                        
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
                        
        elif 'discrete_action' in response:
            actions = response['discrete_action']
            manager.get_logger().info(f"[Plan] Actions: {actions}")
            if actions != [5] and actions != [9]:
                manager.incremental_change_goal(actions)
                current_control_mode = ControlMode.PID_Mode
        elif response.get('status') == 'waiting':
            pass
        else:
            manager.last_trajs_in_world = None
            _stamp = manager.get_clock().now().to_msg()
            _del_traj = Marker()
            _del_traj.header = Header(stamp=_stamp, frame_id=frame_id)
            _del_traj.ns = "trajectory"
            _del_traj.id = 0
            _del_traj.action = Marker.DELETE
            manager.traj_pub.publish(_del_traj)


def visualize_thread(frame_id="os_sensor"):
    """Occupancy grid visualization thread."""
    global manager, calib

    while manager is None or calib is None:
        time.sleep(0.1)

    manager.get_logger().info(f"[Thread] Visualize Thread Started.")

    u_grid = None
    v_grid = None

    while True:
        if not manager.new_vis_image_arrived:
            time.sleep(0.01)
            continue
        manager.new_vis_image_arrived = False

        rgb_depth_rw_lock.acquire_read()
        depth_image = copy.deepcopy(manager.depth_image)
        rgb_depth_rw_lock.release_read()

        if depth_image is None:
            continue

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
            continue

        xy_world = pcloud_filtered[:, :2]
        stamp = manager.get_clock().now().to_msg()

        half = 5.0
        grid_size = 100
        resolution = 0.1
        xmin, ymin = -half, -half

        grid = -np.ones((grid_size, grid_size), dtype=np.int16)
        pts = np.asarray(xy_world, dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) > 0:
            cx = np.floor((pts[:, 0] - xmin) / resolution).astype(np.int32)
            cy = np.floor((pts[:, 1] - ymin) / resolution).astype(np.int32)
            valid = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
            grid[cy[valid], cx[valid]] = 100

        occ_msg = OccupancyGrid()
        occ_msg.header = Header(stamp=stamp, frame_id=frame_id)
        occ_msg.info.resolution = resolution
        occ_msg.info.width = grid_size
        occ_msg.info.height = grid_size
        occ_msg.info.origin.position.x = xmin
        occ_msg.info.origin.position.y = ymin
        occ_msg.info.origin.orientation.w = 1.0
        occ_msg.data = grid.reshape(-1).astype(np.int8).tolist()
        manager.occ_grid_pub.publish(occ_msg)

        time.sleep(0.05)


class Go2Manager(Node):
    def __init__(self, odom_topic='/odom_bridge'):
        super().__init__('go2_manager_ultra')
        
        self.get_logger().info("Initializing Go2Manager Node (Ultra)...")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        rgb_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")

        self.syncronizer = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 30, 0.5)
        self.syncronizer.registerCallback(self.rgb_depth_callback)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, qos_profile)
        self.get_logger().info(f"Subscribing to: {odom_topic}")

        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        _tf = TransformStamped()
        _tf.header.stamp = self.get_clock().now().to_msg()
        _tf.header.frame_id = 'map'
        _tf.child_frame_id = 'camera_init'
        _tf.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(_tf)

        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, '/internav/occupancy_grid', 5)
        self.traj_pub = self.create_publisher(Marker, '/internav/trajectory', 5)
        self.subgoal_pub = self.create_publisher(Marker, '/internav/subgoal', 5)

        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)

        self.last_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        
        self.get_logger().info("Go2Manager Initialized.")

    def rgb_depth_callback(self, rgb_msg, depth_msg):
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
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.depth_bytes = depth_bytes
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
        
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        odom_rw_lock.release_write()

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()
            self.get_logger().info(f"[Odom] First: ({self.odom[0]:.2f}, {self.odom[1]:.2f})")

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            return
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
                homo_goal[:3, :3] = np.dot(np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]), homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15)
                homo_goal[:3, :3] = np.dot(np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]), homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw
        self.control_pub.publish(request)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra-High-Performance InternVLA Client')
    
    parser.add_argument('--odom_topic', type=str, default='/gdq/msg/gdq_odom')
    parser.add_argument('--calib', type=str, default='scripts/realworld/calib/calib_scout.txt')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--frame_id', type=str, default='os_sensor')
    parser.add_argument('--traj_width', type=float, default=0.1)
    parser.add_argument('--subgoal_radius', type=float, default=0.3)
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:5802/eval_dual')
    
    args = parser.parse_args()

    print("=" * 60)
    print("ULTRA-HIGH-PERFORMANCE InternVLA CLIENT")
    print("=" * 60)
    print(f"Server: {args.server_url}")
    print(f"Odom: {args.odom_topic}")
    print(f"Visualize: {args.visualize}")
    print("=" * 60)

    calib = Calibration(args.calib)

    control_thread_inst = threading.Thread(target=control_thread)
    planning_thread_inst = threading.Thread(target=planning_thread, args=(args.frame_id, args.traj_width, args.subgoal_radius))
    control_thread_inst.daemon = True
    planning_thread_inst.daemon = True
    
    if args.visualize:
        visualize_thread_inst = threading.Thread(target=visualize_thread, args=(args.frame_id,))
        visualize_thread_inst.daemon = True
    
    rclpy.init()

    try:
        manager = Go2Manager(odom_topic=args.odom_topic)

        control_thread_inst.start()
        planning_thread_inst.start()
        if args.visualize:
            visualize_thread_inst.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager:
            manager.destroy_node()
        rclpy.shutdown()
